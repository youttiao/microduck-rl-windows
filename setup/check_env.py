"""MicroDuck environment self-check: GPU / PyTorch / Warp / MuJoCo / ONNX / task registry.

Usage:
    D:\\duck\\repos\\microduck_rl\\.venv\\Scripts\\python.exe -I D:\\duck\\setup\\check_env.py
    (or: cd D:\\duck\\repos\\microduck_rl && uv run --no-sync python -I D:\\duck\\setup\\check_env.py)

Checks, in order:
  1. torch is the CUDA build and can see the GPU (and really run a kernel)
  2. warp can initialise CUDA and run a kernel on it
  3. mujoco / mujoco-warp import
  4. onnxruntime can run a real [1,61] -> [1,14] policy
  5. the task registry enumerates

Output is ASCII-only on purpose: a Windows console defaults to a legacy code
page (cp936 here) and would garble anything non-ASCII.
"""

import sys

FAILURES = []


def ok(msg):
    print(f"  [ OK ] {msg}")


def bad(msg):
    print(f"  [FAIL] {msg}")
    FAILURES.append(msg)


def warn(msg):
    print(f"  [WARN] {msg}")


print("=" * 72)
print("MicroDuck RL environment check")
print("=" * 72)

# ---------------------------------------------------------------- 1. torch
print("\n1) PyTorch / CUDA")
torch = None
try:
    import torch

    torch = torch
    ok(f"torch {torch.__version__}")
    ok(f"torch.version.cuda = {torch.version.cuda}")
    ok(f"torch.cuda.is_available() = {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        bad("CUDA unavailable -- training cannot run (MuJoCo Warp needs a GPU)")
    else:
        ok(f"CUDA device count = {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            major, minor = torch.cuda.get_device_capability(i)
            ok(
                f"  GPU {i}: {props.name}  "
                f"sm_{major}{minor}  VRAM {props.total_memory / 1024**3:.1f} GiB"
            )
            if major < 12:
                warn(
                    f"  GPU {i} is sm_{major}{minor}, below Blackwell's sm_120 -- "
                    "check that torch is a cu128-or-newer build"
                )
        x = torch.randn(512, 512, device="cuda")
        y = (x @ x).sum().item()
        torch.cuda.synchronize()
        ok(f"CUDA kernel ran (sum={y:.3f})")
except Exception as exc:  # noqa: BLE001
    bad(f"torch import/run failed: {exc}")

# ---------------------------------------------------------------- 2. warp
print("\n2) NVIDIA Warp (the layer under MuJoCo Warp)")
try:
    import warp as wp

    ok(f"warp {wp.__version__}")
    wp.init()
    devs = [str(d) for d in wp.get_devices()]
    ok(f"warp devices: {devs}")
    if torch is not None and torch.cuda.is_available():
        try:
            wp.set_device("cuda:0")
            a = wp.zeros(3, dtype=wp.float32, device="cuda:0")
            b = wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device="cuda:0")
            wp.copy(a, b)
            wp.synchronize()
            got = [round(float(x), 4) for x in a.numpy()]
            ok(f"warp CUDA kernel ran -> {got}")
            if got != [1.0, 2.0, 3.0]:
                bad(f"warp CUDA kernel produced wrong values: {got}")
        except Exception as exc:  # noqa: BLE001
            bad(f"warp CUDA kernel failed: {exc}")
    else:
        warn("no CUDA torch -- skipping the warp CUDA kernel test")
except Exception as exc:  # noqa: BLE001
    bad(f"warp import failed: {exc}")

# ---------------------------------------------------------------- 3. mujoco
print("\n3) MuJoCo / MuJoCo Warp")
for mod in ("mujoco", "mujoco_warp"):
    try:
        m = __import__(mod)
        ok(f"{mod} {getattr(m, '__version__', '?')}")
    except Exception as exc:  # noqa: BLE001
        bad(f"{mod} import failed: {exc}")

# ---------------------------------------------------------------- 4. onnx
print("\n4) ONNX inference (the deployment path)")
try:
    import os

    import numpy as np
    import onnxruntime as ort

    ok(f"onnxruntime {ort.__version__}")
    providers = ort.get_available_providers()
    ok(f"providers: {providers}")
    onnx_path = r"D:\duck\data\hf\microduck-policies\alpha_walking.onnx"
    if os.path.exists(onnx_path):
        sess = ort.InferenceSession(onnx_path, providers=providers)
        inp, out = sess.get_inputs()[0], sess.get_outputs()[0]
        ok(f"policy IO: {inp.name}{inp.shape} -> {out.name}{out.shape}")
        res = sess.run(None, {inp.name: np.zeros((1, 61), dtype=np.float32)})[0]
        ok(f"inference ran, output shape = {res.shape}")
        if list(inp.shape) != [1, 61] or list(out.shape) != [1, 14]:
            bad(f"policy violates the [1,61] -> [1,14] contract: {inp.shape} -> {out.shape}")
        if not np.isfinite(res).all():
            bad("policy output contains NaN/Inf")
    else:
        warn(f"missing {onnx_path}, skipping the policy inference check")
except Exception as exc:  # noqa: BLE001
    bad(f"onnxruntime failed: {exc}")

# ---------------------------------------------------------------- 5. tasks
print("\n5) microduck_rl task registry")
try:
    import mjlab.tasks  # noqa: F401  -- importing registers the tasks

    from mjlab.tasks.registry import list_tasks

    tasks = sorted(list_tasks())
    duck = [t for t in tasks if "MicroDuck" in t]
    ok(f"{len(tasks)} tasks registered ({len(duck)} MicroDuck)")
    for t in duck:
        print(f"         {t}")
except Exception as exc:  # noqa: BLE001
    bad(f"task registry read failed: {exc}")
    print("         (an mjlab import error here usually means mujoco/warp are broken)")

print("\n" + "=" * 72)
if FAILURES:
    print(f"RESULT: {len(FAILURES)} check(s) FAILED")
    for f in FAILURES:
        print(f"  - {f}")
    print("=" * 72)
    sys.exit(1)
print("RESULT: all checks passed")
print("=" * 72)
sys.exit(0)
