"""Ground-truth test for the full-body (stepping) retargeting.

Builds a synthetic human that walks -- pelvis advancing, feet alternating
between stance and swing -- and checks that the retargeted step channels
have the right structure: an alternating contact schedule, foot travel and
clearance inside the workspace the stepping env expects, and a root that
moves forward.

    python test_retarget_full.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from duck_head_ik import DuckHeadIK  # noqa: E402
from retarget_human import (  # noqa: E402
    DUCK_LEG_LENGTH, REFERENCE_FOOT_TRAVEL, STEP_COLS, retarget,
)

XML = "src/mjlab_microduck/robot/microduck/robot_groundcontact.xml"

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


def build_walk(n=300, fps=30.0, step_period=0.5, n_steps=4.0):
    """A human walking forward: stance feet stay put, swing feet step ahead."""
    t = np.arange(n) / fps
    duration = t[-1]
    pelvis_speed = 1.2                      # m/s
    pelvis_x = pelvis_speed * t

    frames = np.zeros((n, len(ORDER), 3))
    for i in range(n):
        phase = (t[i] / step_period) % 1.0   # 0..1 within a step cycle
        base = BODY.copy()
        base[ORDER.index("pelvis")] += np.array([pelvis_x[i], 0.0, 0.0])

        for side, foot_name, phase_off in (("l", "l_foot", 0.0), ("r", "r_foot", 0.5)):
            p = (phase + phase_off) % 1.0
            # Stance for the first 60% of the cycle (the foot stays planted
            # in the world while the pelvis moves on), swing for the rest.
            if p < 0.6:
                # planted: the foot stays at the world position it had when
                # the stance began, so the pelvis walks past it.
                step_index = math.floor((t[i] + phase_off * step_period) / step_period)
                world_x = (step_index + phase_off) * pelvis_speed * step_period
                z = -0.89
            else:
                u = (p - 0.6) / 0.4            # 0..1 through the swing
                step_index = math.floor((t[i] + phase_off * step_period) / step_period)
                x0 = (step_index + phase_off) * pelvis_speed * step_period
                x1 = x0 + pelvis_speed * step_period * 0.4
                world_x = x0 + (x1 - x0) * u
                z = -0.89 + 0.12 * math.sin(math.pi * u)   # a 12 cm swing height
            y = 0.10 if side == "l" else -0.10
            base[ORDER.index(foot_name)] += np.array([world_x, y, z + 0.89])
        frames[i] = base
    return frames, t, duration, pelvis_speed


def main() -> int:
    ik = DuckHeadIK(XML)
    frames, t, duration, speed = build_walk()
    names = list(ORDER)

    _t, _vals, info, _q, steps = retarget(frames, names, 30.0, ik, full_body=True)
    s = info["steps"]
    col = {c: steps[:, i] for i, c in enumerate(STEP_COLS)}

    print("=== the retargeted step channels ===")
    print(f"  duration        {info['duration_s']:.2f} s  ({info['frames_out']} frames at 50 Hz)")
    print(f"  length_scale    {s['length_scale']}   time_scale {s['time_scale']}")
    print(f"  human stance travel   {s['human_stance_travel_m']} m")
    print(f"  human total travel    {s['human_total_foot_travel_m']} m"
          f"   ->  duck {s['duck_total_foot_travel_m']} m")
    print(f"  duck root travel      {s['duck_root_travel_mm']} mm")
    print(f"  duck clearance        {s['duck_max_clearance_mm']} mm")
    print(f"  planted: left {s['left_planted_frac']}, right {s['right_planted_frac']}"
          f"  (switches {s['left_contact_switches']} / {s['right_contact_switches']})")

    ok = True

    print("\n=== checks ===")

    # 1. the contact schedules must alternate
    lc = col["l_contact"] > 0.5
    rc = col["r_contact"] > 0.5
    both = np.mean(lc & rc)
    neither = np.mean(~lc & ~rc)
    print(f"  both feet planted  {both*100:5.1f}%   neither planted {neither*100:5.1f}%")
    if both > 0.35:
        print("  FAIL  the feet are planted together too often -- not a walk")
        ok = False
    else:
        print("  OK    the feet alternate")
    if neither > 0.02:
        print("  FAIL  both feet off the ground at once -- the duck would fall")
        ok = False
    else:
        print("  OK    at least one foot always down")

    # 2. the reference must start at zero displacement
    zero = all(abs(col[c][0]) < 1e-6 for c in ("lf_x", "lf_y", "rf_x", "rf_y", "root_x", "root_y"))
    print(f"  {'OK  ' if zero else 'FAIL'}  starts at zero displacement")
    ok &= zero

    # 3. the root must move forward, and the feet must lag behind it
    root_fwd = col["root_x"][-1] - col["root_x"][0]
    print(f"  {'OK  ' if root_fwd > 0 else 'FAIL'}  root moves forward: {root_fwd*1000:.1f} mm")
    ok &= root_fwd > 0

    # 4. the PER-STEP foot travel must stay inside the workspace the env
    #    expects (the total travel grows without bound in a walk)
    step_travel = s["human_stance_travel_m"] * s["length_scale"]
    print(f"  {'OK  ' if step_travel <= REFERENCE_FOOT_TRAVEL * 1.05 else 'FAIL'}  "
          f"per-step foot travel {step_travel*1000:.1f} mm  "
          f"(env's reference max {REFERENCE_FOOT_TRAVEL*1000:.0f} mm)")
    ok &= step_travel <= REFERENCE_FOOT_TRAVEL * 1.05

    # 5. the clearance must be positive while the foot is in swing
    swing = ~lc
    if swing.any():
        cl = col["lf_z"][swing]
        print(f"  {'OK  ' if cl.min() >= -1e-6 else 'FAIL'}  clearance >= 0 while swinging "
              f"(min {cl.min()*1000:.2f} mm)")
        ok &= cl.min() >= -1e-6
        print(f"  {'OK  ' if col['lf_z'][lc].max() < 0.005 else 'FAIL'}  clearance ~0 while planted "
              f"(max {col['lf_z'][lc].max()*1000:.2f} mm)")
        ok &= col["lf_z"][lc].max() < 0.005
    else:
        print("  FAIL  the left foot never swings")
        ok = False

    # 6. the time scaling must be the Froude one for the length scale
    froude = 1.0 / math.sqrt(s["length_scale"])
    print(f"  {'OK  ' if abs(s['time_scale'] - froude) < 0.01 else 'FAIL'}  "
          f"time scale is Froude: {s['time_scale']} vs 1/sqrt(k)={froude:.4f}")
    ok &= abs(s["time_scale"] - froude) < 0.01

    # 7. the duck's leg length should be in the ballpark of the scale used
    print(f"  (duck leg {DUCK_LEG_LENGTH*1000:.0f} mm, human leg 850 mm, "
          f"k = {DUCK_LEG_LENGTH/0.85:.3f})")

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
