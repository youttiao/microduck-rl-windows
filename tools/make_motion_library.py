"""Build a MicroDuck motion library from human motions.

The loop the whole thing hangs on:

    human motion (joint positions)  ->  retarget  ->  Tremendous CSV
        ->  register as a DataMotion  ->  train one policy per motion
        ->  export ONNX  ->  install on the robot as a skill

This script does the first two steps for a whole directory, and prints (and
optionally applies) the registration snippet for the expressive env.

    python make_motion_library.py --src motions/ --out \
        D:/duck/repos/microduck_rl/src/mjlab_microduck/tasks/motion_data \
        --name-prefix ""

What it expects of the input
---------------------------
Each source is one human motion with 3D joint positions:
    <name>.npy            (N, J, 3) float array
    <name>.joints.json     {"joint_names": [...], "fps": 30}
or a single .npz holding "joints", "joint_names" and "fps".

See retarget_human.py for the joint names it understands.

What it produces
----------------
One CSV per motion in the expressive env's motion_data/ directory, plus the
DataMotion lines to paste into
`src/mjlab_microduck/tasks/expressive_motions.py`:

    "MyMotion": DataMotion("MyMotion", 18.25, "my_motion.csv", note="..."),

Then `uv run list-envs` shows `Mjlab-Expressive-MyMotion-MicroDuck` and
`uv run train Mjlab-Expressive-MyMotion-MicroDuck ...` trains it.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from duck_head_ik import DuckHeadIK  # noqa: E402
from retarget_human import load_joints, retarget, write_csv  # noqa: E402

# A name that is safe as a Python identifier and as a task id fragment.
SAFE = re.compile(r"[^A-Za-z0-9]+")


def slug(name: str) -> str:
    parts = [p for p in SAFE.split(name) if p]
    return "_".join(p.lower() for p in parts) or "motion"


def task_name(name: str) -> str:
    """The motion's name as it appears in a task id: `mjlab-Expressive-<Name>-MicroDuck`."""
    parts = [p for p in SAFE.split(name) if p]
    return "".join(p[:1].upper() + p[1:] for p in parts) or "Motion"


def find_sources(src: Path) -> list[Path]:
    if src.is_file():
        return [src]
    out = []
    for ext in ("*.npy", "*.npz", "*.json", "*.csv"):
        out.extend(sorted(src.glob(ext)))
    # A .npy whose names live in a sidecar .json is one motion, not two.
    json_sidecars = {p.with_suffix(".joints.json") for p in out}
    return [p for p in out if p not in json_sidecars]


def motion_name(path: Path) -> str:
    return path.stem.replace("_joints", "")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path,
                    help="a human-motion file, or a directory of them")
    ap.add_argument("--out", required=True, type=Path,
                    help="the env's motion_data/ directory")
    ap.add_argument("--xml", default="src/mjlab_microduck/robot/microduck/robot_groundcontact.xml")
    ap.add_argument("--height-scale", type=float, default=None)
    ap.add_argument("--smooth-s", type=float, default=0.08)
    ap.add_argument("--tail-frac", type=float, default=0.12)
    ap.add_argument("--register", action="store_true",
                    help="also add the DataMotion entries to expressive_motions.py")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    sources = find_sources(args.src)
    if not sources:
        print(f"no motions found under {args.src}", file=sys.stderr)
        return 1

    ik = DuckHeadIK(args.xml)
    kw = {}
    if args.height_scale is not None:
        kw["height_scale"] = args.height_scale

    entries = []
    for src in sources:
        name = motion_name(src)
        try:
            joints, fps, names = load_joints(src)
            t, vals, info, _q = retarget(joints, names, fps, ik,
                                         smooth_s=args.smooth_s,
                                         tail_frac=args.tail_frac, **kw)
        except Exception as exc:  # noqa: BLE001
            print(f"  SKIP {name}: {exc}", file=sys.stderr)
            continue

        csv_name = f"{slug(name)}.csv"
        csv_path = Path(args.out) / csv_name
        if not args.dry_run:
            write_csv(csv_path, t, vals, name, info)

        res = info["head_ik_residual_deg"]
        flag = "" if res["frames_over_5deg"] == 0 else "  <-- head unreachable in some frames"
        print(f"  {name:24s} {info['duration_s']:6.2f}s  {info['frames_out']:4d} frames  "
              f"head err {res['mean']:.3f}/{res['max']:.2f} deg{flag}")
        entries.append((task_name(name), csv_name, info))

    print()
    print("--- paste into src/mjlab_microduck/tasks/expressive_motions.py ---")
    print("from .expressive_motions import DATA_MOTIONS as _EXPRESSIVE_DATA_MOTIONS")
    print()
    for name, csv_name, info in entries:
        print(f'    "{name}": DataMotion("{name}", {info["duration_s"]:.2f}, "{csv_name}", '
              f'note="retargeted from human motion"),')
    print()
    print("--- the tasks this creates ---")
    for name, _csv, _info in entries:
        print(f"    uv run train Mjlab-Expressive-{name}-MicroDuck --env.scene.num-envs 4096")
    print()

    if args.register and not args.dry_run:
        target = Path("src/mjlab_microduck/tasks/expressive_motions.py")
        if not target.exists():
            print(f"cannot find {target} to patch; run from the repo root", file=sys.stderr)
            return 1
        src_text = target.read_text(encoding="utf-8")
        if "DATA_MOTIONS = {" not in src_text:
            print("cannot find DATA_MOTIONS in expressive_motions.py", file=sys.stderr)
            return 1
        block = "".join(
            f'    "{name}": DataMotion("{name}", {info["duration_s"]:.2f}, "{csv_name}", '
            f'note="retargeted from human motion"),\n'
            for name, csv_name, info in entries
        )
        src_text = src_text.replace("DATA_MOTIONS = {\n", "DATA_MOTIONS = {\n" + block, 1)
        target.write_text(src_text, encoding="utf-8")
        print(f"registered {len(entries)} motion(s) in {target}")
        print("now run:  uv run list-envs   to see the new tasks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
