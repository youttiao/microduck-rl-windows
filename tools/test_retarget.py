"""Ground-truth test for the retargeter.

Builds a synthetic human whose trunk and head motion is known exactly, runs
it through `retarget_human.py`, and checks the recovered channels against the
truth. No dataset needed -- this is the unit test for the maths.

    python test_retarget.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from duck_head_ik import DuckHeadIK  # noqa: E402
from retarget_human import REF_CHANNELS, _frames, retarget  # noqa: E402

XML = "src/mjlab_microduck/robot/microduck/robot_groundcontact.xml"

# A human skeleton in the duck's convention: x = forward, y = left, z = up.
# (pelvis at the origin; the height offsets are folded into the pelvis.)
SKELETON = {
    "pelvis":     (0.00,  0.00, 0.00),
    "l_hip":      (0.00,  0.09, -0.03),
    "r_hip":      (0.00, -0.09, -0.03),
    "l_shoulder": (0.00,  0.19,  0.47),
    "r_shoulder": (0.00, -0.19,  0.47),
    "neck":       (0.00,  0.00,  0.55),
    "head":       (0.00,  0.00,  0.67),
    "l_ear":      (0.00,  0.08,  0.67),
    "r_ear":      (0.00, -0.08,  0.67),
    "l_foot":     (0.00,  0.10, -0.89),
    "r_foot":     (0.00, -0.10, -0.89),
}
ORDER = list(SKELETON)
BODY = np.array([SKELETON[k] for k in ORDER])


def rot(rpy: np.ndarray) -> np.ndarray:
    """Rotation matrix from (roll, pitch, yaw) in the duck's convention."""
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def build_motion(n=300, fps=30.0):
    """A synthetic motion with known ground truth.

    Trunk: a 0.3 rad roll and a 0.2 rad pitch oscillation, a 0.4 rad yaw
    sweep, and a 0.15 m pelvis dip (a squat).
    Head: a 0.25 rad nod and a 0.3 rad turn.
    """
    t = np.arange(n) / fps
    truth = dict(roll=[], pitch=[], yaw=[], dz=[], head_pitch=[], head_yaw=[])
    frames = np.zeros((n, len(ORDER), 3))

    for i in range(n):
        u = t[i] / t[-1]
        roll = 0.30 * np.sin(2 * np.pi * u)
        pitch = 0.20 * np.sin(2 * np.pi * u + 0.6)
        yaw = 0.40 * (u - 0.5)
        squat = 0.15 * np.sin(np.pi * u)          # dip and come back
        head_pitch = 0.25 * np.sin(2 * np.pi * u + 1.0)
        head_yaw = 0.30 * np.sin(2 * np.pi * u + 0.3)

        truth["roll"].append(roll)
        truth["pitch"].append(pitch)
        truth["yaw"].append(yaw)
        truth["dz"].append(-squat)
        truth["head_pitch"].append(head_pitch)
        truth["head_yaw"].append(head_yaw)

        Rt = rot(np.array([roll, pitch, yaw]))
        # The head turns about the neck: a rotation applied to the head's own
        # frame, then placed back at the neck.
        Rh_local = rot(np.array([0.0, head_pitch, head_yaw]))
        # Pelvis drops by the squat; the feet stay on the ground, so the
        # height offset the retargeter measures is the pelvis dip.
        pelvis = np.array([0.0, 0.0, -squat])
        base = BODY.copy()

        # Rotate the trunk part about the pelvis (the body-frame origin).
        # The neck belongs here too: it is the trunk's up axis, and the head
        # chain hangs off it.
        trunk_idx = [ORDER.index(k) for k in
                     ("pelvis", "l_hip", "r_hip", "l_shoulder", "r_shoulder", "neck")]
        base[trunk_idx] = (Rt @ base[trunk_idx].T).T + pelvis

        # The head chain rotates about the neck, in the trunk's frame. The
        # head's offsets from the neck are BODY-frame quantities; rotate them
        # and then place them at the (already transformed) neck.
        head_idx = [ORDER.index(k) for k in ("neck", "head", "l_ear", "r_ear")]
        head_local = BODY[head_idx] - BODY[ORDER.index("neck")]
        pivot = base[ORDER.index("neck")]
        base[head_idx] = (Rt @ Rh_local @ head_local.T).T + pivot

        frames[i] = base
    return frames, {k: np.array(v) for k, v in truth.items()}, fps


def main() -> int:
    ik = DuckHeadIK(XML)
    frames, truth, fps = build_motion()
    names = list(ORDER)

    t, vals, info, q_head = retarget(frames, names, fps, ik)
    got = {c: vals[:, i] for i, c in enumerate(REF_CHANNELS)}

    # The truth is sampled at the source rate; the retargeted channels are at
    # 50 Hz, so put the truth on the same grid before comparing.
    t_src = np.arange(len(truth["roll"])) / fps
    truth = {k: np.interp(t, t_src, v) for k, v in truth.items()}

    # The env measures the trunk's yaw RELATIVE to the motion's start (yaw0)
    # and the roll and pitch relative to the duck's standing attitude, so the
    # truth has to be expressed the same way.
    truth["roll"] = truth["roll"] - truth["roll"][0]
    truth["pitch"] = truth["pitch"] - truth["pitch"][0]
    truth["yaw"] = truth["yaw"] - truth["yaw"][0]

    # Compare over the middle of the motion. The window stops before the
    # tail fade (which starts at 88% of the motion) and after the first
    # frames, where the reference is still settling.
    lo, hi = int(0.10 * len(t)), int(0.85 * len(t))

    print("=== trunk channels (retrieved vs truth) ===")
    print(f"  (the duck's own standing attitude: roll {np.degrees(ik.trunk_rpy_home[0]):.2f} deg, "
          f"pitch {np.degrees(ik.trunk_rpy_home[1]):.2f} deg)")
    ok = True
    for c in ("roll", "pitch"):
        base = ik.trunk_rpy_home[0 if c == "roll" else 1]
        a, b = got[c][lo:hi], truth[c][lo:hi] + base
        err = np.abs(a - b)
        flag = "OK  " if err.max() < 0.02 else "FAIL"
        ok &= err.max() < 0.02
        print(f"  {flag} {c:10s} max err {np.degrees(err.max()):6.3f} deg   "
              f"mean {np.degrees(err.mean()):.4f} deg")
    a, b = got["yaw"][lo:hi], truth["yaw"][lo:hi]
    err = np.abs(a - b)
    flag = "OK  " if err.max() < 0.02 else "FAIL"
    ok &= err.max() < 0.02
    print(f"  {flag} {'yaw':10s} max err {np.degrees(err.max()):6.3f} deg   "
          f"mean {np.degrees(err.mean()):.4f} deg")

    print("\n=== head orientation (via the joint IK) ===")
    # The IK is free to distribute the motion across the four joints however
    # it likes -- neck_pitch and head_pitch are geared in OPPOSITE directions,
    # so the duck's nod is head_pitch - neck_pitch, and a yaw in the duck
    # ends up partly in head_roll. The thing that must match is the COMPOSED
    # head orientation, so compare that: the achieved head orientation
    # relative to the duck's HOME head, against the human's head motion
    # relative to its own neutral head.
    from scipy.spatial.transform import Rotation as Rot
    R_t, R_h = _frames(frames, {"pelvis": ORDER.index("pelvis"),
                                 "head": ORDER.index("head"),
                                 "neck": ORDER.index("neck"),
                                 "l_shoulder": ORDER.index("l_shoulder"),
                                 "r_shoulder": ORDER.index("r_shoulder"),
                                 "l_ear": ORDER.index("l_ear"),
                                 "r_ear": ORDER.index("r_ear")}, True)
    R_rel = np.einsum("nji,njk->nik", R_t, R_h)      # trunk^-1 @ head
    R_delta_h = R_rel @ R_rel[0].T
    achieved = np.stack([ik.head_orientation(q) for q in q_head])
    achieved_delta = achieved @ ik.R_rel_home.T[None]
    err = np.array([np.degrees(np.linalg.norm(Rot.from_matrix(
        achieved_delta[i] @ R_delta_h[i].T).as_rotvec())) for i in range(len(q_head))])
    flag = "OK  " if err[lo:hi].max() < 0.5 else "FAIL"
    ok &= err[lo:hi].max() < 0.5
    print(f"  {flag} {'head pose':10s} max err {err[lo:hi].max():.4f} deg   "
          f"mean {err[lo:hi].mean():.5f} deg")
    print(f"       (the IK's own residual: {info['head_ik_residual_deg']})")

    print("\n=== height channel ===")
    scale = info["height_scale"]
    a = got["dz"][lo:hi]
    b = truth["dz"][lo:hi] * scale
    err = np.abs(a - b)
    flag = "OK  " if err.max() < 0.004 else "FAIL"
    ok &= err.max() < 0.004
    print(f"  {flag} dz          max err {err.max()*1000:.3f} mm   "
          f"mean {err.mean()*1000:.4f} mm   (height_scale {scale})")

    print("\n=== IK residual ===")
    print(f"  {info['head_ik_residual_deg']}")

    print("\n=== output shape ===")
    print(f"  t: {t.shape[0]} frames, {t[1]-t[0]:.4f} s step, {t[-1]:.2f} s total")
    print(f"  values: {vals.shape[1]} channels {REF_CHANNELS}")
    print(f"  starts at zero: {np.allclose(vals[0], 0, atol=1e-6)}")
    print(f"  ends at zero:   {np.allclose(vals[-1], 0, atol=1e-3)}")

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
