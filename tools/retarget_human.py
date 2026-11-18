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
    "l_toe":      ("left_toe", "l_toe", "toe_l", "left_foot_tip"),
    "r_toe":      ("right_toe", "r_toe", "toe_r", "right_foot_tip"),
}

# The 12 step channels the expressive env reads, in the CSV's order.
STEP_COLS = ("lf_x", "lf_y", "lf_yaw", "lf_z", "rf_x", "rf_y", "rf_yaw", "rf_z",
             "l_contact", "r_contact", "root_x", "root_y")

# The duck's leg length (root z - foot z at the standing pose), and the
# practical workspace the stepping reference upstream tuned its rewards
# against: a foot lifts ~18 mm and travels ~45 mm, the root moves ~40 mm.
DUCK_LEG_LENGTH = 0.117
REFERENCE_FOOT_TRAVEL = 0.045

# A human's leg, hip joint to floor.
HUMAN_LEG_LENGTH = 0.85

# A human's foot counts as "planted" while it is within this of the ground.
DEFAULT_CONTACT_HEIGHT = 0.03

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
    if p.suffix == ".bvh":
        # A BVH stores a skeleton, not positions: this runs the forward
        # kinematics to get the world positions.
        from bvh import load_bvh

        pos, _rot, bvh_fps, bvh_names = load_bvh(p)
        return pos, (fps or bvh_fps), bvh_names
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
             tail_frac: float = 0.12,
             full_body: bool = False,
             length_scale: float | None = None,
             time_scale: float | None = None,
             contact_height: float = DEFAULT_CONTACT_HEIGHT):
    """Human joints -> (t, values(8), info, head_joints, steps).

    `head_joints` is the solved (neck_pitch, head_pitch, head_yaw, head_roll)
    at the source rate, for checking the head orientation against the truth.
    `steps` is the 12 step channels (None unless `full_body`).
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

    # --- full body: the 12 step channels, at the source rate ---------------
    steps = None
    step_info = {}
    if full_body:
        _t_steps, steps, step_info = retarget_full_body(
            joints, names, fps, float(y0),
            length_scale=length_scale,
            time_scale=time_scale,
            contact_height=contact_height,
        )
        # A single time axis for the whole CSV: the step channels and the
        # trunk channels must share it, so the motion is re-timed (slowed) for
        # both when time_scale > 1.
        time_scale = step_info["time_scale"]
        if time_scale != 1.0:
            t_in = t_in * time_scale

    # --- resample, smooth, normalise --------------------------------------
    t, vals = resample(vals, t_in, 50.0)
    first = vals[0].copy()
    vals = smooth(vals, 50.0, smooth_s)
    # The moving average biases the first sample (the window is padded with
    # the edge value), and the reference has to read exactly 0 at phi = 0 --
    # that is the standing pose the episode starts from.
    vals[0] = first
    vals = blend_to_zero(vals, tail_frac)

    if steps is not None:
        # Resample once. The contact columns are a hard 0/1, so they take the
        # NEAREST sample -- interpolating between "planted" and "swing" would
        # smear the transition and blur the schedule.
        t_out = np.arange(t_in[0], t_in[-1] + 1e-9, 1.0 / 50.0)
        out_steps = np.empty((len(t_out), steps.shape[1]))
        for c in range(steps.shape[1]):
            name = STEP_COLS[c]
            if name in ("l_contact", "r_contact"):
                j = np.clip(np.searchsorted(t_in, t_out, side="left"), 0, len(t_in) - 1)
                out_steps[:, c] = steps[j, c]
            else:
                out_steps[:, c] = np.interp(t_out, t_in, steps[:, c])
        steps = out_steps
        # A planted foot is on the ground, so its target clearance is 0. This
        # has to happen AFTER the resample: the clearance is interpolated while
        # the contact channel is nearest-neighbour, so zeroing before the
        # resample would let the two drift apart at the transitions.
        for clear_c, contact_c in (("lf_z", "l_contact"), ("rf_z", "r_contact")):
            kc = STEP_COLS.index(clear_c)
            kt = STEP_COLS.index(contact_c)
            steps[:, kc] = steps[:, kc] * (1.0 - steps[:, kt])
        # The swing clearance is measured from the foot's start height, so the
        # first frame is 0 by construction; the foot travel likewise.
        for c in ("lf_x", "lf_y", "rf_x", "rf_y", "root_x", "root_y"):
            k = STEP_COLS.index(c)
            steps[:, k] -= steps[0, k]
        step_info["resampled_frames"] = int(len(t))

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
        "yaw0": round(float(y0), 5),
    }
    if step_info:
        info["steps"] = step_info
    return t, vals, info, q_head, steps


def retarget_full_body(joints: np.ndarray, names: list[str] | None, fps: float,
                       yaw0: float,
                       length_scale: float | None = None,
                       time_scale: float | None = None,
                       contact_height: float = DEFAULT_CONTACT_HEIGHT):
    """Human motion -> the 12 step channels (STEP_COLS order).

    The channels are, per foot: the horizontal DISPLACEMENT from that foot's
    own start position, in the start heading frame; the swing clearance above
    the start height; and the contact schedule (1 = loaded/planted). Plus the
    root's horizontal displacement.

    Length and time scaling
    -----------------------
    A human's leg is ~85 cm, the duck's ~12 cm, so the duck's stride is about
    an eighth of the human's -- but a stride that short at the human's tempo
    would be a shuffle. Biped robots are dynamically similar at different
    sizes when time scales with the SQUARE ROOT of length (Froude), so:

        length_scale = duck_leg / human_leg      (k)
        time_scale   = 1 / sqrt(k)

    A 60 cm human step becomes a ~8 cm duck step taken ~2.9x longer. The
    length scale is additionally capped so the foot travel stays inside the
    workspace the stepping env's rewards were tuned against
    (REFERENCE_FOOT_TRAVEL), because a reference the policy cannot reach
    teaches it nothing.

    Pass `length_scale` / `time_scale` explicitly to override.
    """
    idx = resolve_joints(names)
    for need in ("l_foot", "r_foot"):
        if need not in idx:
            raise ValueError(
                f"full-body retargeting needs the {need} joint; add it to the "
                "joint names (see JOINT_ALIASES)"
            )
    n = joints.shape[0]
    t_in = np.arange(n) / float(fps)

    lf = joints[:, idx["l_foot"]]
    rf = joints[:, idx["r_foot"]]
    pelvis = joints[:, idx["pelvis"]]

    # --- contact schedule ---------------------------------------------------
    # The ground is the foot's own low percentile (robust to a noisy clip).
    # The threshold is RELATIVE to the foot's own height range: a fixed 3 cm
    # would swallow the start and end of every swing (a 12 cm swing is a third
    # of it), which blurs the schedule the env rewards against.
    def contact_of(foot):
        z = foot[:, 2]
        ground = float(np.percentile(z, 10))
        rng = float(z.max() - z.min())
        thr = min(max(0.05 * rng, 0.005), contact_height)
        state = np.zeros(len(z))
        planted = 0
        for i in range(len(z)):
            if planted:
                # Release late: a foot that is lifting off stays loaded a
                # little longer than one that is landing.
                if z[i] > ground + 1.15 * thr:
                    planted = 0
            else:
                if z[i] < ground + thr:
                    planted = 1
            state[i] = planted
        return state

    l_contact = contact_of(lf)
    r_contact = contact_of(rf)

    # --- horizontal displacements in the start heading frame -----------------
    c, s = math.cos(-yaw0), math.sin(-yaw0)

    def to_start(xy):
        return np.stack((c * xy[:, 0] + s * xy[:, 1],
                         -s * xy[:, 0] + c * xy[:, 1]), axis=1)

    lf_disp = to_start(lf[:, :2] - lf[0, :2])
    rf_disp = to_start(rf[:, :2] - rf[0, :2])
    root_disp = to_start(pelvis[:, :2] - pelvis[0, :2])
    max_travel = max(float(np.abs(lf_disp).max()), float(np.abs(rf_disp).max()), 1e-6)

    # --- length scale -------------------------------------------------------
    # The characteristic scale of the reference is the distance a foot travels
    # RELATIVE TO THE BODY during one stance -- the body walks on over a
    # planted foot. That is what the upstream reference encodes (its foot
    # moves ~45 mm back while the root moves ~40 mm forward). It is NOT the
    # total distance the foot covers over the motion, which for a walk grows
    # without bound.
    leg_k = DUCK_LEG_LENGTH / HUMAN_LEG_LENGTH

    def stance_travel(foot_xy, contact):
        """Median per-stance travel of the foot relative to the body (m)."""
        rel = foot_xy - pelvis[:, :2]
        travels = []
        i = 0
        n = len(contact)
        while i < n:
            if contact[i] < 0.5:
                i += 1
                continue
            j = i
            while j < n and contact[j] >= 0.5:
                j += 1
            seg = rel[i:j]
            if len(seg) >= 3:
                travels.append(float(np.linalg.norm(seg[-1] - seg[0])))
            i = j
        return float(np.median(travels)) if travels else 0.0

    l_stance = stance_travel(lf[:, :2], l_contact)
    r_stance = stance_travel(rf[:, :2], r_contact)
    stance = float(np.median([v for v in (l_stance, r_stance) if v > 0])) if max(l_stance, r_stance) > 0 else 0.0
    if stance <= 1e-6:
        # No walking in this motion (feet planted): fall back to the total
        # travel, which is then just the small in-place movement.
        stance = max_travel

    if length_scale is None:
        # Cap so the per-stance travel lands inside the workspace the stepping
        # env's rewards were tuned against.
        length_scale = min(leg_k, REFERENCE_FOOT_TRAVEL / max(stance, 1e-6))
    if time_scale is None:
        time_scale = 1.0 / math.sqrt(length_scale)

    # --- foot yaw (stored by the env but not read by any reward) ------------
    def foot_yaw(foot):
        if "l_toe" in idx and "r_toe" in idx:
            toe = joints[:, idx["l_toe" if foot is lf else "r_toe"]]
            fwd = toe[:, :2] - foot[:, :2]
        else:
            # No toe marker: use the foot's own velocity direction.
            vel = np.gradient(foot[:, :2], 1.0 / fps, axis=0)
            fwd = vel
        ang = np.arctan2(fwd[:, 1], fwd[:, 0])
        return np.mod(ang - ang[0] + math.pi, 2 * math.pi) - math.pi

    lf_yaw = foot_yaw(lf)
    rf_yaw = foot_yaw(rf)

    # --- clearance ----------------------------------------------------------
    # A PLANTED foot is on the ground: its target clearance is 0, not the few
    # millimetres it happens to float at. Only the swing has a clearance to
    # track, and that is the swing height above the ground.
    def clearance(foot, contact):
        z = foot[:, 2]
        ground = float(np.percentile(z, 10))
        c = (z - ground) * length_scale
        c = c - c[0]                       # the env measures from the start height
        c = np.maximum(c, 0.0)
        c[contact >= 0.5] = 0.0             # planted -> on the ground
        return c

    lf_clear = clearance(lf, l_contact)
    rf_clear = clearance(rf, r_contact)

    steps = np.concatenate([
        lf_disp * length_scale, lf_yaw[:, None], lf_clear[:, None],
        rf_disp * length_scale, rf_yaw[:, None], rf_clear[:, None],
        l_contact[:, None], r_contact[:, None],
        root_disp * length_scale,
    ], axis=1)

    info = {
        "length_scale": round(length_scale, 5),
        "time_scale": round(time_scale, 4),
        "contact_height_m": contact_height,
        "human_stance_travel_m": round(stance, 4),
        "human_total_foot_travel_m": round(max_travel, 4),
        "duck_total_foot_travel_m": round(max_travel * length_scale, 4),
        "duck_max_clearance_mm": round(float(max(lf_clear.max(), rf_clear.max()) * 1000), 2),
        "duck_root_travel_mm": round(float(np.abs(root_disp * length_scale).max() * 1000), 1),
        "left_planted_frac": round(float(l_contact.mean()), 3),
        "right_planted_frac": round(float(r_contact.mean()), 3),
        "left_contact_switches": int(np.sum(np.abs(np.diff(l_contact)) > 0.5)),
        "right_contact_switches": int(np.sum(np.abs(np.diff(r_contact)) > 0.5)),
    }
    if info["duck_root_travel_mm"] > REFERENCE_FOOT_TRAVEL * 1000:
        info["warning"] = (
            "the motion translates: the root travels more than the duck's step "
            "reach. The stepping env measures the foot's ABSOLUTE displacement "
            "from its start, so the reference can still be matched, but the "
            "episode gets long and the duck is really walking -- use the "
            "velocity task for locomotion and this one for in-place motion."
        )
    return t_in * time_scale, steps, info


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


def write_csv(path: Path, t: np.ndarray, vals: np.ndarray, name: str, info: dict,
              steps: np.ndarray | None = None) -> None:
    """Write the CSV the expressive env reads: t, 8 values, 8 derivatives.

    With `steps`, also the 12 step channels (foot targets, contact schedule,
    root displacement), in the order the env's STEP_COLS expects.
    """
    dt = float(t[1] - t[0])
    der = np.gradient(vals, dt, axis=0)
    out = np.concatenate([t[:, None], vals, der], axis=1)
    header = (
        f"# {name}, retargeted from human joint positions "
        f"(height_scale={info['height_scale']}, {info['fps_in']:g} Hz -> 50 Hz). "
        "columns: t, " + ", ".join(REF_CHANNELS) + ", d_" + ", d_".join(REF_CHANNELS)
    )
    if steps is not None:
        out = np.concatenate([out, steps], axis=1)
        header += ", " + ", ".join(STEP_COLS)
        s = info.get("steps", {})
        header += (
            f"  [length_scale={s.get('length_scale')}, time_scale={s.get('time_scale')}, "
            f"l_planted={s.get('left_planted_frac')}, r_planted={s.get('right_planted_frac')}]"
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
    ap.add_argument("--full-body", action="store_true",
                    help="also produce the 12 step channels (foot targets, contact "
                         "schedule, root displacement)")
    ap.add_argument("--length-scale", type=float, default=None,
                    help="foot travel scale (default: the duck/human leg ratio, "
                         "capped to the workspace the stepping env expects)")
    ap.add_argument("--time-scale", type=float, default=None,
                    help="time scale (default: Froude, 1/sqrt(length_scale))")
    ap.add_argument("--contact-height", type=float, default=DEFAULT_CONTACT_HEIGHT,
                    help="a human foot within this of the ground counts as planted (m)")
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
    t, vals, info, _q_head, steps = retarget(
        joints, names, fps, ik,
        height_scale=args.height_scale,
        smooth_s=args.smooth_s,
        tail_frac=args.tail_frac,
        full_body=args.full_body,
        length_scale=args.length_scale,
        time_scale=args.time_scale,
        contact_height=args.contact_height,
    )

    print(json.dumps(info, indent=2, ensure_ascii=False))
    problems = check_limits(vals, ik)
    if problems:
        print("\nOUT OF RANGE (the env clamps and taxes these):", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
    else:
        print("\nall channels inside the duck's joint limits")

    if not args.dry_run:
        write_csv(args.out, t, vals, args.name, info, steps)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
