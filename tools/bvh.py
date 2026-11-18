"""BVH reader: hierarchy + motion -> world joint positions and rotations.

CMU's Motion Capture Facility publishes BVH. A BVH file stores a skeleton
as a hierarchy of OFFSETs and per-joint rotation CHANNELS, plus one row of
channel values per frame -- so the world positions have to be computed with
forward kinematics. This does that.

It also returns each joint's world ROTATION matrix, which the retargeter
uses for the foot's yaw: a BVH's foot carries a real toe joint, so the look
direction is exact rather than inferred from the foot's velocity.

The joint names come out as the file spells them ("Hips", "LeftUpLeg",
"LeftFoot", "LeftToeBase", ...), which retarget_human.JOINT_ALIASES does
not know. `BVH_ALIASES` maps them onto the retargeter's canonical names.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# BVH joint names (CMU / most mocap exports) -> the retargeter's names.
BVH_ALIASES = {
    "Hips": "pelvis",
    "LeftUpLeg": "l_hip",
    "RightUpLeg": "r_hip",
    "LeftArm": "l_shoulder",
    "RightArm": "r_shoulder",
    "LeftForeArm": "l_elbow",
    "RightForeArm": "r_elbow",
    "LeftHand": "l_hand",
    "RightHand": "r_hand",
    "Neck": "neck",
    "Head": "head",
    "LeftFoot": "l_foot",
    "RightFoot": "r_foot",
    "LeftToeBase": "l_toe",
    "RightToeBase": "r_toe",
    "LeftToeEnd": "l_toe",
    "RightToeEnd": "r_toe",
    "LeftEye": "l_eye",
    "RightEye": "r_eye",
}

# The channels a joint can have, and how many numbers each takes.
_CHANNEL_N = {"Xposition": 1, "Yposition": 1, "Zposition": 1,
              "Xrotation": 1, "Yrotation": 1, "Zrotation": 1}


class _Joint:
    __slots__ = ("name", "offset", "channels", "parent", "children")

    def __init__(self, name, offset, channels, parent):
        self.name = name
        self.offset = np.array(offset, dtype=np.float64)
        self.channels = list(channels)
        self.parent = parent
        self.children = []


def _rotation(axis: str, angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    m = np.eye(4)
    if axis == "X":
        m[1, 1], m[1, 2], m[2, 1], m[2, 2] = c, -s, s, c
    elif axis == "Y":
        m[0, 0], m[0, 2], m[2, 0], m[2, 2] = c, s, -s, c
    else:
        m[0, 0], m[0, 1], m[1, 0], m[1, 1] = c, -s, s, c
    return m


def _translation(axis: str, v: float) -> np.ndarray:
    m = np.eye(4)
    if axis == "X":
        m[0, 3] = v
    elif axis == "Y":
        m[1, 3] = v
    else:
        m[2, 3] = v
    return m


def _parse_hierarchy(lines, i):
    """Parse one joint block; returns (joint, next_line_index)."""
    tok = lines[i].split()
    name = tok[1]
    i += 1
    assert lines[i].strip().startswith("{"), "expected { after the joint name"
    i += 1
    offset = [0.0, 0.0, 0.0]
    channels = []
    children = []
    while True:
        s = lines[i].strip()
        if s == "}":
            i += 1
            break
        if s.startswith("OFFSET"):
            offset = [float(x) for x in s.split()[1:4]]
            i += 1
        elif s.startswith("CHANNELS"):
            n = int(s.split()[1])
            channels = s.split()[2:2 + n]
            i += 1
        elif s.startswith("JOINT"):
            child, i = _parse_hierarchy(lines, i)
            children.append(child)
        elif s.startswith("End"):
            i += 1
        else:
            i += 1
    joint = _Joint(name, offset, channels, None)
    joint.children = children
    return joint, i


def load_bvh(path: str | Path):
    """Parse a BVH file.

    Returns (positions (N, J, 3), rotations (N, J, 3, 3), fps, names).

    `names` is the list of joint names in column order, translated through
    BVH_ALIASES where possible (so "Hips" becomes "pelvis").
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    # --- hierarchy ---------------------------------------------------------
    i = 0
    while i < len(lines) and not lines[i].strip().upper().startswith("HIERARCHY"):
        i += 1
    i += 1
    while i < len(lines) and not lines[i].strip().upper().startswith("ROOT"):
        i += 1
    root, i = _parse_hierarchy(lines, i)

    # Flatten the tree, remembering each joint's parent.
    order = []
    stack = [(root, None)]
    while stack:
        j, parent = stack.pop(0)
        j.parent = parent
        order.append(j)
        for c in reversed(j.children):
            stack.insert(0, (c, j))
    # (the queue above gives a pre-order walk; parents come before children)

    # --- motion ------------------------------------------------------------
    while i < len(lines) and not lines[i].strip().upper().startswith("MOTION"):
        i += 1
    i += 1
    n_frames = 0
    frame_time = 1.0 / 30.0
    while i < len(lines):
        s = lines[i].strip()
        if s.lower().startswith("frames:"):
            n_frames = int(s.split(":")[1])
        elif s.lower().startswith("frame time:"):
            frame_time = float(s.split(":")[1])
        elif n_frames:
            break
        i += 1
    data = np.loadtxt(lines[i:i + n_frames]) if n_frames else np.zeros((0, 0))
    if data.ndim == 1:
        data = data[None, :]

    # --- forward kinematics ------------------------------------------------
    n = len(order)
    names = [BVH_ALIASES.get(j.name, j.name) for j in order]
    pos = np.zeros((len(data), n, 3))
    rot = np.zeros((len(data), n, 3, 3))

    # Where each joint's channels start in the data row.
    offsets = []
    col = 0
    for j in order:
        offsets.append(col)
        col += len(j.channels)

    idx = {j.name: k for k, j in enumerate(order)}
    for f in range(len(data)):
        row = data[f]
        world = {}
        for j in order:
            # The joint's own OFFSET, then its channels, in the parent's frame.
            local = np.eye(4)
            local[:3, 3] = j.offset
            p = 0
            for ch in j.channels:
                v = row[offsets[idx[j.name]] + p]
                p += 1
                if ch.endswith("position"):
                    # Position channels only exist on the root, in BVH.
                    local = local @ _translation(ch[0], v)
                else:
                    local = local @ _rotation(ch[0], np.deg2rad(v))
            if j.parent is None:
                world[j.name] = local
            else:
                world[j.name] = world[j.parent.name] @ local
            W = world[j.name]
            rot[f, idx[j.name]] = W[:3, :3]
            pos[f, idx[j.name]] = W[:3, 3]

    fps = 1.0 / frame_time if frame_time else 30.0
    return pos, rot, fps, names
