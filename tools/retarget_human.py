"""Human motion -> MicroDuck "Tremendous" reference channels.

Reads 3D joint positions and writes the CSV the expressive-motion env expects.

Why joint positions and not SMPL parameters
--------------------------------------------
The expressive env only needs three things about the human: the trunk's
orientation, the trunk's height, and the head's orientation. All three are
readable from joint positions alone, so no body model (SMPL / SMPL-X) is
involved -- which keeps this retargeting code clear of the SMPL-X licence and
lets ANY dataset that ships 3D joint positions be a source.

The target format
-----------------
`mjlab_microduck.tasks.expressive_motions.REF_CHANNELS`, 8 channels, all as
offsets from the standing pose:

    roll, pitch   trunk orientation (ZYX Euler of trunk_base, rad).
                  pitch > 0 = nose DOWN (bow), < 0 = chest out.
                  roll  > 0 = right side down.
    yaw           trunk yaw relative to the heading at the start of the motion.
    dz            trunk height offset (m): root z - mean foot z, minus its
                  value at the start of the motion.
    neck_pitch    > 0 = neck back / head high   (HOME 0.349, [-1.571, 1.047])
    head_pitch    > 0 = nod down (chin down)     (HOME 0.349, [-1.571, 1.571])
    head_yaw      > 0 = look LEFT                (HOME 0, +-2.967)
    head_roll     tilt                           (HOME 0, +-0.436)

The four head channels are joint offsets from the duck's HOME, exactly as the
env measures them (`_servo_joint_pos - _servo_default_joint_pos`).

The CSV is `t` + 8 values + 8 time derivatives, 50 Hz, with `#` comments.
The env reads it with `np.loadtxt(..., delimiter=",", comments="#")` and takes
column 0 as time, 1..8 as the values, 9..16 as the derivatives.

Frames
------
Everything here uses the duck's convention: x = forward, y = left, z = up,
right-handed, and the trunk's orientation as a ZYX Euler (roll about forward,
pitch about left, yaw about up). The human's trunk and head frames are built
to match, so the trunk channels transfer directly.

Usage
-----
    python retarget_human.py --joints motion.npy --joint-names names.json \
                             --out motion.csv

See `JOINT_ALIASES` for the joint names it understands and `load_joints` for
the accepted input formats.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from duck_head_ik import DuckHeadIK  # noqa: E402

REF_CHANNELS = ("roll", "pitch", "yaw", "dz",
                "neck_pitch", "head_pitch", "head_yaw", "head_roll")

# Joint names we understand, with the aliases each dataset uses for them.
JOINT_ALIASES = {
    "pelvis":     ("pelvis", "root", "hips", "hip", "pelvis_joint"),
    "head":       ("head", "head_joint"),
    "neck":       ("neck", "neck_joint", "base_of_spine", "spine", "spine1"),
    "l_hip":      ("left_hip", "l_hip", "hip_l"),
    "r_hip":      ("right_hip", "r_hip", "hip_r"),
    "l_shoulder": ("left_shoulder", "l_shoulder", "shoulder_l"),
    "r_shoulder": ("right_shoulder", "r_shoulder", "shoulder_r"),
    "l_foot":     ("left_foot", "l_foot", "foot_l", "left_ankle", "l_ankle"),
    "r_foot":     ("right_foot", "r_foot", "foot_r", "right_ankle", "r_ankle"),
    "l_ear":      ("left_ear", "l_ear", "ear_l"),
    "r_ear":      ("right_ear", "r_ear", "ear_r"),
}

# The duck's standing pelvis height (expressive_motions.STAND_HEIGHT). Used to
# report the scale, not in the maths.
TARGET_STAND_HEIGHT = 0.1152

# A human squat moves the pelvis ~0.4 m down; the duck can only drop ~15 mm.
# 0.035 puts a full human squat just inside the duck's range.
DEFAULT_HEIGHT_SCALE = 0.035

DEFAULT_XML = "src/mjlab_microduck/robot/microduck/robot_groundcontact.xml"


# ---------------------------------------------------------------- helpers

def _norm(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


def _orthonormalise(left_axis: np.ndarray, up: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a right-handed (x=forward, y=left, z=up) frame.

    `left_axis` and `up` need not be perpendicular; `up` is re-orthogonalised
    against `left_axis`. Returns (forward, left, up).
    """
    u = up - left_axis * float(np.dot(left_axis, up))
    u = _norm(u)
    fwd = np.cross(left_axis, u)        # x = left x up
    fwd = _norm(fwd)
    left = np.cross(u, fwd)             # y = up x forward
    left = _norm(left)
    return fwd, left, u


def _frames(joints: np.ndarray, idx: dict[str, int], want_ears: bool) -> tuple[np.ndarray, np.ndarray]:
    """Trunk and head rotation matrices per frame, columns = (fwd, left, up)."""
    n = joints.shape[0]
    R_t = np.zeros((n, 3, 3))
    R_h = np.zeros((n, 3, 3))
    pelvis = joints[:, idx["pelvis"]]
    ls = joints[:, idx["l_shoulder"]]
    rs = joints[:, idx["r_shoulder"]]
    head = joints[:, idx["head"]]
    neck = joints[:, idx["neck"]] if "neck" in idx else 0.5 * (ls + rs)
    have_ears = want_ears and "l_ear" in idx and "r_ear" in idx

    for i in range(n):
        # --- trunk -------------------------------------------------------
        # The up axis comes from the NECK (or the shoulder midpoint), not
        # from the head: the head moves relative to the trunk, so using it
        # would tilt the trunk frame with every nod.
        left = _norm(ls[i] - rs[i])                 # right shoulder -> left shoulder
        up_raw = _norm(neck[i] - pelvis[i])
        fwd, lft, up = _orthonormalise(left, up_raw)
        R_t[i] = np.stack([fwd, lft, up], axis=1)

        # --- head --------------------------------------------------------
        up_h_raw = _norm(head[i] - neck[i])
        if have_ears:
            left_h = _norm(joints[i, idx["l_ear"]] - joints[i, idx["r_ear"]])
        else:
            left_h = left
        fwd_h, lft_h, up_h = _orthonormalise(left_h, up_h_raw)
        R_h[i] = np.stack([fwd_h, lft_h, up_h], axis=1)
    return R_t, R_h


def zyx_euler(R: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ZYX (yaw-pitch-roll) decomposition of rotation matrices.

    Matches the env's `_ex_trunk_rpy`: roll about forward, pitch about left,
    yaw about up.
    """
    roll = np.arctan2(R[:, 2, 1], R[:, 2, 2])
    pitch = np.arcsin(np.clip(-R[:, 2, 0], -1.0, 1.0))
    yaw = np.arctan2(R[:, 1, 0], R[:, 0, 0])
    return roll, pitch, yaw


# ---------------------------------------------------------------- loading

def load_joints(path: Path, fps: float | None = None) -> tuple[np.ndarray, float, list[str] | None]:
    """Load (N, J, 3) joint positions.

    Accepted:
      .npy   -> array (N, J, 3)
      .npz   -> "joints" (plus optional "joint_names", "fps"), or the first
                3-D array with a trailing 3
      .json  -> {"joints": [...], "joint_names": [...], "fps": ...}
      .csv   -> rows of J*3 numbers
    """
    p = Path(path)
    sidecar = p.with_suffix(".joints.json")
    if sidecar.exists():
        d = json.loads(sidecar.read_text(encoding="utf-8"))
        names = [str(x) for x in d["joint_names"]]
        if fps is None:
            fps = float(d.get("fps", 30.0))
    if p.suffix == ".npy":
        return np.load(p).astype(np.float64), (fps or 30.0), names
    if p.suffix == ".npz":
        z = np.load(p, allow_pickle=True)
        names = [str(x) for x in z["joint_names"]] if "joint_names" in z else None
        if "joints" in z:
            arr = z["joints"]
        else:
            arr = next(z[k] for k in z.files if z[k].ndim == 3 and z[k].shape[-1] == 3)
        if fps is None:
            fps = float(z["fps"]) if "fps" in z else 30.0
        return np.asarray(arr, dtype=np.float64), fps, names
    if p.suffix == ".json":
        d = json.loads(p.read_text(encoding="utf-8"))
        if fps is None:
            fps = float(d.get("fps", 30.0))
        return np.asarray(d["joints"], dtype=np.float64), fps, d.get("joint_names")
    if p.suffix == ".csv":
        raw = np.loadtxt(p, delimiter=",", comments="#")
        if raw.ndim == 1:
            raw = raw[None, :]
        return raw.reshape(raw.shape[0], -1, 3).astype(np.float64), (fps or 30.0), names
    raise ValueError(f"unsupported file type: {p.suffix}")


def resolve_joints(names: list[str] | None) -> dict[str, int]:
    """Map our canonical joint names to column indices."""
    if names is None:
        raise ValueError(
            "no joint names: the retargeter needs to know which column is the "
            "pelvis, the shoulders, the ears and the feet. Pass --joint-names "
            f"with a JSON list. Accepted aliases: {JOINT_ALIASES}"
        )
    lower = [str(n).strip().lower() for n in names]
    out: dict[str, int] = {}
    for canon, aliases in JOINT_ALIASES.items():
        for a in aliases:
            if a in lower:
                out[canon] = lower.index(a)
                break
    missing = [c for c in ("pelvis", "l_shoulder", "r_shoulder", "head") if c not in out]
    if missing:
        raise ValueError(f"missing required joints {missing}; got {names}")
    if "l_ear" not in out or "r_ear" not in out:
        out.pop("l_ear", None)
        out.pop("r_ear", None)
    return out


# ---------------------------------------------------------------- resampling

def resample(x: np.ndarray, t_in: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray]:
    t_out = np.arange(t_in[0], t_in[-1] + 1e-9, 1.0 / fps)
    out = np.stack([np.interp(t_out, t_in, x[:, c]) for c in range(x.shape[1])], axis=1)
    return t_out, out


def smooth(x: np.ndarray, fps: float, window_s: float = 0.08) -> np.ndarray:
    """Zero-phase moving average: removes mocap jitter, keeps the shape."""
    w = max(1, int(round(window_s * fps)))
    if w <= 1:
        return x
    pad = w // 2
    xp = np.pad(x, ((pad, pad), (0, 0)), mode="edge")
    ker = np.ones(w) / w
    out = np.empty_like(x)
    for c in range(x.shape[1]):
        out[:, c] = np.convolve(xp[:, c], ker, mode="same")[pad: pad + x.shape[0]]
    return out


def blend_to_zero(x: np.ndarray, tail_frac: float = 0.12) -> np.ndarray:
    """Fade the last `tail_frac` of the motion linearly back to zero.

    The env wants every reference channel back at 0 before phi = 1, so the
    duck ends standing.
    """
    n = x.shape[0]
    k = max(2, int(n * (1.0 - tail_frac)))
    w = np.ones(n)
    w[k:] = np.linspace(1.0, 0.0, n - k)
    return x * w[:, None]


# ---------------------------------------------------------------- main

def retarget(joints: np.ndarray, names: list[str] | None, fps: float,
             ik: DuckHeadIK,
             height_scale: float = DEFAULT_HEIGHT_SCALE,
             smooth_s: float = 0.08,
             tail_frac: float = 0.12):
    """Human joints -> (t, values(8), info, head_joints).

    `head_joints` is the solved (neck_pitch, head_pitch, head_yaw, head_roll)
    at 50 Hz, for checking the head orientation against the truth.
    """
    idx = resolve_joints(names)
    n = joints.shape[0]
    t_in = np.arange(n) / float(fps)

    # --- trunk orientation, relative to the motion's first frame ----------
    R_t, R_h = _frames(joints, idx, want_ears=True)
    roll, pitch, yaw = zyx_euler(R_t)
    r0, p0, y0 = zyx_euler(R_t[0][None])[0][0], zyx_euler(R_t[0][None])[1][0], zyx_euler(R_t[0][None])[2][0]
    # The env measures the trunk's roll and pitch ABSOLUTELY (only the yaw is
    # taken relative to the motion's start), so the reference starts from the
    # duck's own standing attitude, not from zero.
    roll = roll - r0 + ik.trunk_rpy_home[0]
    pitch = pitch - p0 + ik.trunk_rpy_home[1]
    yaw = np.mod(yaw - y0 + math.pi, 2 * math.pi) - math.pi      # wrapped

    # --- trunk height ------------------------------------------------------
    feet = [idx[k] for k in ("l_foot", "r_foot") if k in idx]
    foot_z = joints[:, feet, 2].mean(axis=1) if feet else np.zeros(n)
    dz = (joints[:, idx["pelvis"], 2] - foot_z)
    dz = (dz - dz[0]) * height_scale

    # --- head: solve the four joints against the real robot ----------------
    # The human's head motion is measured RELATIVE TO THE HUMAN'S OWN neutral
    # head, and applied ON TOP OF the duck's standing head pose. That way the
    # reference starts at zero (the duck stands) and the human's neutral head
    # maps onto the duck's HOME head.
    R_rel = np.einsum("nji,njk->nik", R_t, R_h)      # trunk^-1 @ head
    R_delta = R_rel @ R_rel[0].T                      # head motion vs the human's neutral
    R_target = ik.R_rel_home[None] @ R_delta          # what the duck's head must be
    q_head, residuals = ik.solve_series(R_target)
    head_offsets = q_head - ik.q_home                 # channel = offset from HOME

    vals = np.concatenate([
        np.stack([roll, pitch, yaw, dz], axis=1),
        head_offsets,
    ], axis=1)

    # --- resample, smooth, normalise --------------------------------------
    t, vals = resample(vals, t_in, 50.0)
    first = vals[0].copy()
    vals = smooth(vals, 50.0, smooth_s)
    # The moving average biases the first sample (the window is padded with
    # the edge value), and the reference has to read exactly 0 at phi = 0 --
    # that is the standing pose the episode starts from.
    vals[0] = first
    vals = blend_to_zero(vals, tail_frac)

    info = {
        "frames_in": n,
        "fps_in": fps,
        "frames_out": int(len(t)),
        "duration_s": round(float(t[-1] - t[0]), 3),
        "height_scale": height_scale,
        "resolved_joints": idx,
        "used_ears": "l_ear" in idx,
        "head_ik_residual_deg": {
            "mean": round(float(np.degrees(residuals.mean())), 3),
            "max": round(float(np.degrees(residuals.max())), 2),
            "frames_over_5deg": int((np.degrees(residuals) > 5.0).sum()),
        },
        "peak": {c: round(float(np.abs(vals[:, i]).max()), 4)
                 for i, c in enumerate(REF_CHANNELS)},
    }
    return t, vals, info, q_head


def check_limits(vals: np.ndarray, ik: DuckHeadIK) -> list[str]:
    """Report channels that leave what the duck can do."""
    problems = []
    for i, c in enumerate(REF_CHANNELS):
        if c not in ik.lo_names:
            continue
        k = ik.lo_names.index(c)
        lo, hi = ik.lo[k], ik.hi[k]
        col = vals[:, i]
        if col.max() > hi or col.min() < lo:
            problems.append(
                f"{c}: range [{col.min():.3f}, {col.max():.3f}] leaves the "
                f"joint limit [{lo:.3f}, {hi:.3f}]"
            )
    return problems


def write_csv(path: Path, t: np.ndarray, vals: np.ndarray, name: str, info: dict) -> None:
    """Write the CSV the expressive env reads: t, 8 values, 8 derivatives."""
    dt = float(t[1] - t[0])
    der = np.gradient(vals, dt, axis=0)
    out = np.concatenate([t[:, None], vals, der], axis=1)
    header = (
        f"# {name}, retargeted from human joint positions "
        f"(height_scale={info['height_scale']}, {info['fps_in']:g} Hz -> 50 Hz). "
        "columns: t, " + ", ".join(REF_CHANNELS) + ", d_" + ", d_".join(REF_CHANNELS)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, out, delimiter=",", fmt="%.6f")
    body = path.read_text(encoding="utf-8")
    path.write_text(header + "\n" + body, encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--joints", required=True, type=Path,
                    help=".npy/.npz/.json/.csv of 3D joint positions (N, J, 3)")
    ap.add_argument("--joint-names", type=Path, default=None,
                    help="JSON file with a list of joint names, in column order")
    ap.add_argument("--fps", type=float, default=None, help="source frame rate")
    ap.add_argument("--out", required=True, type=Path, help="output CSV")
    ap.add_argument("--name", default="Retargeted", help="motion name for the header")
    ap.add_argument("--xml", default=DEFAULT_XML, help="the duck's MJCF (for the head IK)")
    ap.add_argument("--height-scale", type=float, default=DEFAULT_HEIGHT_SCALE)
    ap.add_argument("--smooth-s", type=float, default=0.08)
    ap.add_argument("--tail-frac", type=float, default=0.12)
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = ap.parse_args(argv)

    joints, fps, names = load_joints(args.joints, args.fps)
    if args.joint_names is not None:
        names = json.loads(args.joint_names.read_text(encoding="utf-8"))
    elif names is None:
        jn = args.joints.with_suffix(".joints.json")
        if jn.exists():
            names = json.loads(jn.read_text(encoding="utf-8"))

    ik = DuckHeadIK(args.xml)
    t, vals, info, _q_head = retarget(joints, names, fps, ik,
                                      height_scale=args.height_scale,
                                      smooth_s=args.smooth_s,
                                      tail_frac=args.tail_frac)

    print(json.dumps(info, indent=2, ensure_ascii=False))
    problems = check_limits(vals, ik)
    if problems:
        print("\nOUT OF RANGE (the env clamps and taxes these):", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
    else:
        print("\nall channels inside the duck's joint limits")

    if not args.dry_run:
        write_csv(args.out, t, vals, args.name, info)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
