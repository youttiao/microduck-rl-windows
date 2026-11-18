"""Test the BVH reader against a hand-built file.

A BVH stores a skeleton and per-frame channel values, so the world
positions have to come out of the forward kinematics. This builds a small
file whose answer is known exactly -- a root translated and rotated, with a
child at a known offset -- and checks the parsed positions.

    python test_bvh.py
"""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from bvh import load_bvh  # noqa: E402

HIERARCHY = """HIERARCHY
ROOT Hips
{
\tOFFSET 0.00 0.00 0.00
\tCHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
\tJOINT LeftUpLeg
\t{
\t\tOFFSET 0.00 0.10 0.00
\t\tCHANNELS 3 Zrotation Xrotation Yrotation
\t\tJOINT LeftFoot
\t\t{
\t\t\tOFFSET 0.00 0.00 -0.40
\t\t\tCHANNELS 3 Zrotation Xrotation Yrotation
\t\t\tJOINT LeftToeBase
\t\t\t{
\t\t\t\tOFFSET 0.00 0.00 0.10
\t\t\t\tCHANNELS 3 Zrotation Xrotation Yrotation
\t\t\t}
\t\t}
\t}
\tJOINT Spine
\t{
\t\tOFFSET 0.00 0.00 0.30
\t\tCHANNELS 3 Zrotation Xrotation Yrotation
\t\tJOINT Head
\t\t{
\t\t\tOFFSET 0.00 0.00 0.25
\t\t\tCHANNELS 3 Zrotation Xrotation Yrotation
\t\t}
\t}
}
"""


def main() -> int:
    tmp = Path(tempfile.gettempdir()) / "_bvh_test.bvh"
    # frame 0: everything zero. frame 1: the root at (1,2,3), turned 90 deg about Z.
    # The skeleton has 6 root channels and 5 child joints of 3 -> 21 numbers a frame.
    frame0 = "0 0 0 0 0 0" + " 0 0 0" * 5
    frame1 = "1 2 3 90 0 0" + " 0 0 0" * 5
    text = HIERARCHY + "MOTION\nFrames: 2\nFrame Time: 0.01\n" + frame0 + "\n" + frame1 + "\n"
    tmp.write_text(text, encoding="utf-8")

    pos, rot, fps, names = load_bvh(tmp)
    print("joints   :", names)
    print(f"frames   : {pos.shape[0]}   fps {fps}")

    ok = True
    idx = {n: i for i, n in enumerate(names)}

    def check(label, got, want, tol=1e-6):
        nonlocal ok
        good = abs(float(got) - want) < tol
        ok &= good
        print(f"  {'OK  ' if good else 'FAIL'} {label}: {got:.6f} (want {want:.6f})")

    # frame 0: everything at its offset, no rotation
    check("frame0 hips x", pos[0, idx["pelvis"], 0], 0.0)
    check("frame0 hips y", pos[0, idx["pelvis"], 1], 0.0)
    check("frame0 hips z", pos[0, idx["pelvis"], 2], 0.0)
    check("frame0 left_foot z", pos[0, idx["l_foot"], 2], -0.40)
    check("frame0 head z", pos[0, idx["head"], 2], 0.55)
    check("frame0 l_foot y", pos[0, idx["l_foot"], 1], 0.10)

    # frame 1: the root is at (1,2,3) and rotated 90 deg about Z, so every
    # child offset is turned with it. LeftUpLeg's local (0, 0.10, 0) becomes
    # (-0.10, 0, 0) in the world, and LeftFoot's local (0, 0, -0.40) is
    # carried straight down.
    check("frame1 hips x", pos[1, idx["pelvis"], 0], 1.0)
    check("frame1 hips y", pos[1, idx["pelvis"], 1], 2.0)
    check("frame1 hips z", pos[1, idx["pelvis"], 2], 3.0)
    check("frame1 l_foot x", pos[1, idx["l_foot"], 0], 0.90)
    check("frame1 l_foot y", pos[1, idx["l_foot"], 1], 2.00)
    check("frame1 l_foot z", pos[1, idx["l_foot"], 2], 2.60)
    check("frame1 head x", pos[1, idx["head"], 0], 1.0)
    check("frame1 head y", pos[1, idx["head"], 1], 2.0)
    check("frame1 head z", pos[1, idx["head"], 2], 3.55)

    # the root's rotation must be a 90 degree turn about Z
    R = rot[1, idx["pelvis"]]
    check("frame1 root rotation (cos90)", R[0, 0], 0.0, 1e-9)
    check("frame1 root rotation (sin90)", R[1, 0], 1.0, 1e-9)

    # the alias mapping must have renamed the CMU-style names
    want = {"pelvis", "l_hip", "l_foot", "l_toe", "head"}
    missing = want - set(names)
    if missing:
        print(f"  FAIL  the canonical names are missing: {missing}")
        ok = False
    else:
        print("  OK    the CMU joint names were mapped to the canonical ones")

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
