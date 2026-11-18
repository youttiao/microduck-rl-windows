"""SOMA Retargeter CSV -> the MicroDuck "Tremendous" motion library.

NVIDIA's SOMA Retargeter (https://github.com/NVIDIA/soma-retargeter) turns
human BVH into a robot's joint trajectory. What it emits is a flat CSV:

    Frame, root_translateX/Y/Z, root_rotateX/Y/Z, <joint columns>

with the translation in CENTIMETRES and every angle in DEGREES, at 120 FPS.

What the MicroDuck expressive/stepping env wants is the "Tremendous" format
(mjlab_microduck.tasks.expressive_motions): 50 Hz, metres, radians, the
trunk's roll/pitch/yaw and height offset, the four head joints as offsets
from HOME, and -- for the stepping variant -- per-foot targets, a contact
schedule and the root's horizontal displacement.

This script is the bridge between the two.

Units and conventions, established by calibration against MuJoCo's own
forward kinematics (not by reading the format and hoping):

  * translation: centimetres -> metres
  * angles: degrees -> radians
  * root_rotateX/Y/Z: intrinsic ZYX Euler. The GR00T converter reads them
    as intrinsic XYZ, which puts the duck's feet 40-180 mm in the air; the
    ZYX reading lands them on the floor. `--calibrate` re-derives this by
    trying the candidates and keeping the one that puts the feet lowest.
  * the root's roll/pitch/yaw and the foot positions are NOT taken from the
    CSV: they come from MuJoCo's forward kinematics on the MicroDuck's own
    MJCF with the joint values from the CSV, so they are exactly what the
    env itself will measure.

Usage
-----
    python soma_to_motionlib.py --csv in.csv --out out.csv
    python soma_to_motionlib.py --csv in.csv --out out.csv --full-body
    python soma_to_motionlib.py --csv in.csv --out out.csv --calibrate
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from retarget_human import (  # noqa: E402
    REF_CHANNELS, STEP_COLS, blend_to_zero, smooth,
)

DEFAULT_XML = "src/mjlab_microduck/robot/microduck/robot_groundcontact.xml"

# The duck's standing head pose (microduck_constants.HOME_FRAME).
HOME_HEAD = {"neck_pitch": 0.3491, "head_pitch": 0.3491, "head_yaw": 0.0, "head_roll": 0.0}

# The duck's standing height: root z - mean foot z, minus what the servos
# sag under the weight (expressive_motions.STAND_HEIGHT).
STAND_HEIGHT = 0.1152

# The joint columns SOMA writes for MicroDuck, in the order the env expects.
MICRODUCK_JOINTS = (
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
)

# Candidate readings of root_rotateX/Y/Z, and of the world frame they are in.
_EULER_SEQS = ("xyz", "XYZ", "zyx", "ZYX")
_FRAME_SHIFTS = (0.0, 90.0, -90.0)


class DuckModel:
    """The duck's MJCF, used for the forward kinematics."""

    def __init__(self, xml: str | Path):
        import mujoco

        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(xml))
        self.data = mujoco.MjData(self.model)
        self.jid = {
            n: int(self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)])
            for n in MICRODUCK_JOINTS
        }
        self.root = int(self.model.jnt_qposadr[mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")])
        self.trunk = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
        self.foot = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, s)
                     for s in ("left_foot", "right_foot")]
        self.trunk_rpy_home = self._trunk_rpy_at_home()

    def _trunk_rpy_at_home(self):
        """The trunk's roll/pitch/yaw in the standing pose.

        The env measures the trunk's roll and pitch ABSOLUTELY, so a
        reference has to start from the duck's own standing attitude.
        """
        mj = self.mujoco
        d = self.data
        d.qpos[:] = 0.0
        d.qpos[self.root + 3] = 1.0
        legs = {"left_hip_yaw": 0.0, "left_hip_roll": -0.0873, "left_hip_pitch": -0.4579,
                "left_knee": -0.0049, "left_ankle": 0.4530,
                "right_hip_yaw": 0.0, "right_hip_roll": 0.0873, "right_hip_pitch": 0.4579,
                "right_knee": 0.0049, "right_ankle": -0.4530}
        for name, v in {**legs, **HOME_HEAD}.items():
            jid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, name)
            if jid >= 0:
                d.qpos[int(self.model.jnt_qposadr[jid])] = v
        mj.mj_forward(self.model, d)
        R = d.xmat[self.trunk].reshape(3, 3)
        return np.array([np.arctan2(R[2, 1], R[2, 2]),
                         np.arcsin(np.clip(-R[2, 0], -1.0, 1.0)),
                         np.arctan2(R[1, 0], R[0, 0])])

    def fk(self, root_xyz, root_quat_xyzw, joints_rad):
        """World foot positions and the trunk's RPY, for every frame."""
        mj = self.mujoco
        n = len(joints_rad)
        feet = np.zeros((n, 2, 3))
        rpy = np.zeros((n, 3))
        root_z = np.zeros(n)
        for f in range(n):
            d = self.data
            d.qpos[:] = 0.0
            d.qpos[self.root:self.root + 3] = root_xyz[f]
            d.qpos[self.root + 3:self.root + 7] = root_quat_xyzw[f][[3, 0, 1, 2]]  # wxyz
            for k, jn in enumerate(MICRODUCK_JOINTS):
                d.qpos[self.jid[jn]] = joints_rad[f, k]
            mj.mj_forward(self.model, d)
            feet[f, 0] = d.site_xpos[self.foot[0]]
            feet[f, 1] = d.site_xpos[self.foot[1]]
            R = d.xmat[self.trunk].reshape(3, 3)
            rpy[f] = (np.arctan2(R[2, 1], R[2, 2]),
                      np.arcsin(np.clip(-R[2, 0], -1.0, 1.0)),
                      np.arctan2(R[1, 0], R[0, 0]))
            root_z[f] = d.xpos[self.trunk][2]
        return feet, rpy, root_z


def read_soma_csv(path: Path):
    """-> (root_xyz_m (N,3), root_rot_deg (N,3), joints_rad (N,14), names)."""
    header = path.read_text(encoding="utf-8", errors="replace").splitlines()[0].split(",")
    idx = {h.strip(): i for i, h in enumerate(header)}
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data[None, :]
    tr = np.stack([data[:, idx["root_translateX"]], data[:, idx["root_translateY"]],
                   data[:, idx["root_translateZ"]]], axis=1) / 100.0
    rot = np.stack([data[:, idx["root_rotateX"]], data[:, idx["root_rotateY"]],
                    data[:, idx["root_rotateZ"]]], axis=1)
    joints = np.stack([data[:, idx[j]] for j in MICRODUCK_JOINTS], axis=1)
    return tr, rot, np.deg2rad(joints)


def quat_from(root_rot_deg, seq, shift_deg):
    """root rotation -> quaternion (xyzw), for a candidate convention."""
    from scipy.spatial.transform import Rotation as Rot

    q = Rot.from_euler(seq, root_rot_deg, degrees=True).as_quat()
    if shift_deg:
        q = Rot.from_euler("z", shift_deg, degrees=True).as_quat() * q
    return q


def shift_xy(xy, shift_deg):
    from scipy.spatial.transform import Rotation as Rot

    if not shift_deg:
        return xy
    return (Rot.from_euler("z", shift_deg, degrees=True).as_matrix() @ xy.T).T


def calibrate(duck: DuckModel, tr, rot, joints, frames=60):
    """Pick the root-rotation convention that makes the feet plant best.

    The criterion is how STEADY the lower foot is: a planted foot barely
    moves in height, so the candidate with the smallest spread of the lower
    foot's height is the one whose convention is right. (The absolute height
    of that foot is not a criterion -- the SOMA solution tends to hold the
    duck a couple of centimetres above the floor.)
    """
    n = len(tr)
    sel = np.linspace(0, n - 1, min(frames, n)).astype(int)
    best = None
    table = []
    for seq in _EULER_SEQS:
        for shift in _FRAME_SHIFTS:
            q = quat_from(rot[sel], seq, shift)
            feet, _rpy, _rz = duck.fk(shift_xy(tr[sel], shift), q, joints[sel])
            low = feet[..., 2].min(axis=1)
            score = float(low.std())
            table.append((score, seq, shift, float(low.mean())))
            if best is None or score < best[0]:
                best = (score, seq, shift)
    return {"euler": best[1], "frame_shift_deg": best[2],
            "lower_foot_height_std_mm": round(best[0] * 1000, 3),
            "candidates": [{"euler": s, "frame_shift_deg": sh,
                            "lower_foot_std_mm": round(sc * 1000, 3),
                            "lower_foot_mean_mm": round(mn * 1000, 3)}
                           for sc, s, sh, mn in sorted(table)]}


def resample(x, t_in, fps_out):
    t_out = np.arange(t_in[0], t_in[-1] + 1e-9, 1.0 / fps_out)
    out = np.stack([np.interp(t_out, t_in, x[:, c]) for c in range(x.shape[1])], axis=1)
    return t_out, out


def build(duck: DuckModel, csv: Path, fps_out=50.0, seq="zyx", shift=0.0,
          full_body=False, smooth_s=0.08, tail_frac=0.12,
          contact_height=0.012):
    tr, rot, joints = read_soma_csv(csv)
    n = len(tr)
    fps_in = 120.0
    t_in = np.arange(n) / fps_in

    q = quat_from(rot, seq, shift)
    tr_w = shift_xy(tr, shift)

    # Resample to the env's rate: the root on the time grid, the joints too.
    t, tr_w = resample(tr_w, t_in, fps_out)
    _t, joints = resample(joints, t_in, fps_out)
    _t, rot_r = resample(rot, t_in, fps_out)
    q = quat_from(rot_r, seq, shift)

    feet, rpy, root_z = duck.fk(tr_w, q, joints)

    # --- the 8 reference channels -----------------------------------------
    # Every channel is relative to the motion's start, plus the duck's own
    # standing attitude for the trunk's roll and pitch (which the env
    # measures absolutely). That way the reference reads 0 at phi = 0, which
    # is the standing pose the episode starts from.
    roll = rpy[:, 0] - rpy[0, 0] + duck.trunk_rpy_home[0]
    pitch = rpy[:, 1] - rpy[0, 1] + duck.trunk_rpy_home[1]
    yaw = np.mod(rpy[:, 2] - rpy[0, 2] + math.pi, 2 * math.pi) - math.pi
    dz = (root_z - feet[..., 2].mean(axis=1)) - (root_z[0] - feet[0, ..., 2].mean())

    head = joints[:, 5:9] - np.array([HOME_HEAD[k] for k in
                                      ("neck_pitch", "head_pitch", "head_yaw", "head_roll")])
    # The head channels are relative to the motion's start as well: the duck
    # begins standing with its head at HOME, so the reference must read 0
    # there and track the CHANGE from there.
    head = head - head[0]
    vals = np.stack([roll, pitch, yaw, dz, head[:, 0], head[:, 1], head[:, 2], head[:, 3]], axis=1)
    vals = smooth(vals, fps_out, smooth_s)
    vals[0] = np.array([roll[0], pitch[0], 0.0, dz[0], head[0, 0], head[0, 1], head[0, 2], head[0, 3]])
    vals = blend_to_zero(vals, tail_frac)

    steps = None
    step_info = {}
    if full_body:
        steps, step_info = _step_channels(feet, root_z, tr_w, rpy, contact_height)

    info = {
        "frames_in": n,
        "fps_in": fps_in,
        "frames_out": int(len(t)),
        "duration_s": round(float(t[-1]), 3),
        "root_convention": {"euler": seq, "frame_shift_deg": shift},
        "trunk_rpy_at_start_deg": [round(float(np.degrees(v)), 2) for v in rpy[0]],
        "foot_height_min_mm": round(float(feet[..., 2].min() * 1000), 2),
        "foot_height_max_mm": round(float(feet[..., 2].max() * 1000), 2),
        "peak": {c: round(float(np.abs(vals[:, i]).max()), 4)
                 for i, c in enumerate(REF_CHANNELS)},
    }
    if step_info:
        info["steps"] = step_info
    return t, vals, steps, info


def _step_channels(feet, root_z, root_xy, rpy, contact_height):
    """The 12 step channels, from the forward kinematics."""
    n = len(feet)
    yaw0 = rpy[0, 2]
    c, s = math.cos(-yaw0), math.sin(-yaw0)

    def to_start(xy):
        return np.stack((c * xy[:, 0] + s * xy[:, 1],
                         -s * xy[:, 0] + c * xy[:, 1]), axis=1)

    disp_l = to_start(feet[:, 0, :2] - feet[0, 0, :2])
    disp_r = to_start(feet[:, 1, :2] - feet[0, 1, :2])
    root_disp = to_start(root_xy - root_xy[0])

    # Contact: a foot is planted while it stays near ITS OWN ground level.
    # The level is per foot and relative (the SOMA solution tends to hold the
    # duck a couple of centimetres above the floor, and each foot has its
    # own stance height), and the threshold is a fraction of that foot's own
    # travel, so a low step still registers as a step.
    contacts = []
    for k in (0, 1):
        h = feet[:, k, 2]
        ground = float(np.percentile(h, 10))
        rng = float(h.max() - h.min())
        thr = min(max(0.15 * rng, 0.004), contact_height)
        contacts.append(((h - ground) < thr).astype(np.float64))

    # A PLANTED foot does not move: freeze its target for the whole stance.
    # The SOMA leg solution lets the stance foot creep along the floor
    # (~40 mm/s), and a reference that slides would teach the duck to slide.
    def freeze(disp, contact):
        out = disp.copy()
        i = 0
        n = len(contact)
        while i < n:
            if contact[i] < 0.5:
                i += 1
                continue
            j = i
            while j < n and contact[j] >= 0.5:
                j += 1
            # Hold the position the foot had when it touched down.
            out[i:j] = out[i]
            i = j
        return out

    disp_l = freeze(disp_l, contacts[0])
    disp_r = freeze(disp_r, contacts[1])

    def clearance(k, contact):
        h = feet[:, k, 2] - float(np.percentile(feet[:, k, 2], 10))
        cl = np.maximum(h - h[0], 0.0)
        cl[contact >= 0.5] = 0.0
        return cl

    steps = np.concatenate([
        disp_l, np.zeros((n, 1)), clearance(0, contacts[0])[:, None],
        disp_r, np.zeros((n, 1)), clearance(1, contacts[1])[:, None],
        contacts[0][:, None], contacts[1][:, None],
        root_disp,
    ], axis=1)

    info = {
        "foot_travel_mm": [round(float(np.abs(disp_l).max() * 1000), 1),
                           round(float(np.abs(disp_r).max() * 1000), 1)],
        "root_travel_mm": round(float(np.abs(root_disp).max() * 1000), 1),
        "left_planted_frac": round(float(contacts[0].mean()), 3),
        "right_planted_frac": round(float(contacts[1].mean()), 3),
        "left_switches": int(np.sum(np.abs(np.diff(contacts[0])) > 0.5)),
        "right_switches": int(np.sum(np.abs(np.diff(contacts[1])) > 0.5)),
    }
    return steps, info


def write_csv(path: Path, t, vals, steps, name, info):
    dt = float(t[1] - t[0])
    der = np.gradient(vals, dt, axis=0)
    out = np.concatenate([t[:, None], vals, der], axis=1)
    header = (f"# {name}, from the NVIDIA SOMA retargeter "
              f"(root {info['root_convention']['euler']} Euler, "
              f"frame shift {info['root_convention']['frame_shift_deg']:+g} deg). "
              "columns: t, " + ", ".join(REF_CHANNELS) + ", d_" + ", d_".join(REF_CHANNELS))
    if steps is not None:
        out = np.concatenate([out, steps], axis=1)
        header += ", " + ", ".join(STEP_COLS)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, out, delimiter=",", fmt="%.6f")
    body = path.read_text(encoding="utf-8")
    path.write_text(header + "\n" + body, encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, type=Path, help="a SOMA retargeter CSV")
    ap.add_argument("--out", required=True, type=Path, help="the Tremendous CSV to write")
    ap.add_argument("--xml", default=DEFAULT_XML)
    ap.add_argument("--fps", type=float, default=50.0)
    ap.add_argument("--euler", default=None, choices=list(_EULER_SEQS),
                    help="how to read root_rotateX/Y/Z (default: calibrate)")
    ap.add_argument("--frame-shift", type=float, default=None,
                    help="world-frame Z rotation to undo (default: calibrate)")
    ap.add_argument("--full-body", action="store_true",
                    help="also the 12 step channels (foot targets, contact schedule, root)")
    ap.add_argument("--no-calibrate", action="store_true",
                    help="skip the convention search")
    ap.add_argument("--smooth-s", type=float, default=0.08)
    ap.add_argument("--tail-frac", type=float, default=0.12)
    ap.add_argument("--contact-height", type=float, default=0.012)
    args = ap.parse_args(argv)

    duck = DuckModel(args.xml)

    seq, shift = args.euler, args.frame_shift
    if not args.no_calibrate and (seq is None or shift is None):
        tr, rot, joints = read_soma_csv(args.csv)
        cal = calibrate(duck, tr, rot, joints)
        print(json.dumps(cal, indent=2))
        seq = cal["euler"] if seq is None else seq
        shift = cal["frame_shift_deg"] if shift is None else shift

    t, vals, steps, info = build(duck, args.csv, fps_out=args.fps,
                                 seq=seq, shift=shift,
                                 full_body=args.full_body,
                                 smooth_s=args.smooth_s,
                                 tail_frac=args.tail_frac,
                                 contact_height=args.contact_height)
    print(json.dumps(info, indent=2))
    write_csv(args.out, t, vals, steps, args.csv.stem, info)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
