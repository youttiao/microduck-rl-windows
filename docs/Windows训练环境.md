# 在 Windows 上训练 MicroDuck

上游 `pollen-robotics/microduck_rl` 是按 Linux/macOS 写的。在 Windows 上直接 `uv sync` 会得到一个**能装上但跑不起来**的环境。这份文档说明坑在哪、怎么修，以及哪些功能在 Windows 上确实不可用。

---

## 一、最大的坑：PyPI 的 Windows torch wheel 是 CPU 版

### 现象

`uv sync` 一切正常，`.venv` 里也有 torch，但：

```python
>>> torch.__version__
'2.9.1+cpu'
>>> torch.version.cuda
None
>>> torch.cuda.device_count()
0
```

然后一跑 `uv run train <TASK>` 就死：

```
IndexError: list index out of range
  at mjlab/utils/gpu.py:70
```

`select_gpus()` 把 `torch.cuda.device_count()` 的结果映射成 GPU 列表，列表是空的，`available_gpus[i]` 直接越界。**训练第一步都进不去。**

### 为什么会这样

上游在自己的 `AGENTS.md` 里记过一个几乎一样的坑：linux-aarch64（DGX Spark / GB10）上 PyPI 的 torch wheel 是 CPU-only，所以上游用 `[tool.uv.sources]` 把 torch 指向 `download.pytorch.org/whl/cu129` 索引。注释里还写了一句：

> The marker keeps x86_64 (HF Jobs) on PyPI, where the wheel already bundles CUDA via its nvidia-*-cu12 deps — that resolution is unchanged.

**这句在 Linux 上对，在 Windows 上不对。**

- **Linux**：torch 的 PyPI wheel 是一堆元包，CUDA 靠 `nvidia-cuda-runtime-cu12`、`nvidia-cublas-cu12` 这些依赖包带进来。装上就有 CUDA。
- **Windows**：没有这套依赖机制 —— 一个 torch wheel 就是自包含的。而 PyPI 上发布的 Windows 版恰好是 **CPU** 那一个。

所以上游"x86_64 留在 PyPI 就行"的判断，是建立在 Linux 的 wheel 结构上的。

### 修复

在 `pyproject.toml` 里给 win32 加一条 CUDA 索引路由：

```toml
[tool.uv.sources]
torch = [
  { index = "pytorch-cu129", marker = "sys_platform == 'linux' and platform_machine == 'aarch64'" },
  { index = "pytorch-cu128", marker = "sys_platform == 'win32'" },
]
torchvision = [
  { index = "pytorch-cu128", marker = "sys_platform == 'win32'" },
]

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true
```

本仓库的 `patches/` 目录里就有这个改动（`git apply` 一下就行）。

### 为什么选 cu128，不是 cu129 也不是 cu130

| 候选 | 结论 |
|---|---|
| **cu129** | ❌ PyTorch **没有发布** cu129 的 Windows wheel（查 `download.pytorch.org/whl/cu129/torch/` 只有 manylinux） |
| **cu130** | ❌ CUDA 13.0 是另一个主版本。Warp 1.12.0 自带的是 **CUDA 12.9**，而 warp↔torch 之间的零拷贝互操作要求两边 CUDA 主版本一致 |
| **cu128** | ✅ 同主版本（12.x）里最接近的，也是 **Blackwell (sm_120) 的标准构建** —— RTX 50 系就是 sm_120 |

### 为什么 torchvision 也要一起路由

`torchvision` 是 `rsl-rl-lib` 的**传递依赖**，而 **uv 的 `[tool.uv.sources]` 只对直接依赖生效**（上游自己的注释就写了这一点，说 torch 必须留在 `[project.dependencies]` 里）。所以要把它也提上来：

```toml
dependencies = [
    ...
    "torch==2.9.1",
    "torchvision==0.24.1",
]
```

不这么做，`uv sync` 会把 **CPU 版 torchvision 装到 CUDA torch 旁边**。上游的注释解释了为什么版本必须用 `==` 而不是 `>=`：CUDA 索引上的构建比 PyPI 新，`>=` 会把 torch 悄悄拖到 2.13.0，一个没验证过的升级。

### 改完必须重新生成 lock

```bash
uv lock
uv sync --frozen
```

`uv.lock` 里现在有三条 torch 记录：

| 来源 | 版本 | 解析到 |
|---|---|---|
| pypi.org | `2.9.1` | linux-x86_64（**HF Jobs 用的就是这个**）+ macOS |
| download.pytorch.org/whl/cu128 | `2.9.1+cu128` | win32 |
| download.pytorch.org/whl/cu129 | `2.9.1+cu129` | linux-aarch64 |

**HF Jobs（linux-x86_64）的解析完全没变**，这是上游最在意的那条不变量。

---

## 二、不需要装系统级 CUDA Toolkit

Warp 1.12.0 的 Windows wheel **自带 CUDA 12.9 运行时**，PyTorch 的 cu128 wheel **自带 CUDA 12.8 运行时**，两边的 CUDA DLL 都在各自的 wheel 里。机器上的 NVIDIA 驱动负责往上提供兼容层。

验证：

```powershell
D:\duck\repos\microduck_rl\.venv\Scripts\python.exe -c "import warp as wp; wp.init()"
```

输出：

```
Warp 1.12.0 initialized:
   CUDA Toolkit 12.9, Driver 13.5
   Devices:
     "cpu"      : "Intel64 Family 6 Model 198 Stepping 2, GenuineIntel"
     "cuda:0"   : "NVIDIA GeForce RTX 5080 Laptop GPU" (16 GiB, sm_120, mempool enabled)
```

**所以没有"要不要装 CUDA"这个问题** —— 除非你要自己编译 CUDA kernel，那就另说。

---

## 三、CUDA Graph：最可能出问题的地方

mjlab 1.3.0 默认**开启 Warp CUDA graph capture**（`mjlab/sim/sim.py` 的 `_should_use_cuda_graph()`）：

```python
if has_mempool and driver_ver >= _GRAPH_CAPTURE_MIN_DRIVER:   # (12, 4)
    return True
```

本机两个条件都满足（`mempool enabled`、驱动 13.5），所以**默认走 graph capture 路径**。而 graph capture 在 WDDM（Windows 的显示驱动模型）上偶尔会有问题。

### 退路：关掉 mempool 就等于关掉 graph capture

在训练脚本最前面（**必须在 `wp.init()` 之前**）：

```python
import warp as wp
wp.config.enable_mempools_at_init = False
wp.init()
```

mjlab 会自动降级并打印 `[WARNING] CUDA Graphs disabled: mempool disabled`。代价是每步多一点开销（少了 graph replay 的加速），但训练能跑。

### 再不行就上 WSL2

这台机器的 NVIDIA 驱动支持 WSL2 的 CUDA 直通，Linux 是上游的主要目标平台。社区里也有现成方案：

- [`CdyDoodle/maxy_duck`](https://github.com/CdyDoodle/maxy_duck) —— Windows 原生仿真 + WSL 训练 + Web 曲线，中英双语

---

## 四、`scripts/infer_policy.py` 在 Windows 上完全用不了

这是**上游的 POSIX 限制，不是我们改坏的**。

这个脚本用 `termios` / `tty` / `select()` 做裸模式键盘控制（让���在模拟器里手动驾驶鸭子）。`termios` 是 POSIX 独有模块，Windows 上连 import 都会炸：

```
ModuleNotFoundError: No module named 'termios'
```

对应的上游测试已经加了 `skipif(sys.platform == "win32")`。

### 影响

| 功能 | Windows 上 |
|---|---|
| 训练 | ✅ 可以 |
| 导出 ONNX | ✅ 可以 |
| 出回放视频 | ✅ 可以（用 `uv run play --video True`） |
| **键盘手动驾驶模拟器** | ❌ 不行 |

想补上这个功能的话：
- 走 WSL2
- 或者用社区的 `green-creeper/microduck_genesis`（Genesis 版带键盘演练器，跨平台）
- 或者用竞技场的 `check.py` / `arena score`（在 `microduck_challenges` 环境里，跨平台）

---

## 五、上游测试里的 Windows 编码问题

上游测试用 `Path.read_text()` 不带 `encoding=`，在 Windows 上默认用系统代码页（本机 cp936 / GBK），读含 UTF-8 破折号的 `pyproject.toml` 会直接：

```
UnicodeDecodeError: 'gbk' codec can't decode byte 0x94 in position 473
```

（`0x94` 是 UTF-8 破折号 `—` 的第三个字节。**上游原版文件同样会挂**，不是我们改出来的。）

修法就是给这几处加上 `encoding="utf-8"`。

### 另一个：`train` 可执行文件的检查

`test_train_on_path_is_a_mjlab_trainer` 原来只读 `train` 可执行文件的**前 8192 字节**找 `"mjlab"`：

```python
head = Path(exe).read_bytes()[:8192]
assert b"mjlab" in head
```

Linux 上 `train` 是个带 shebang 的文本脚本，一读就有。**Windows 上是个 PE 启动器**，它把目标脚本路径放在映像**末尾**，8KB 窗口根本看不到。改成读整个文件即可（几十 KB，无所谓）。

### 改完的结果

```
272 passed, 4 skipped, 0 failed
```

---

## 六、给上游提 PR 的两个改动

`patches/` 目录里的两个 patch 可以直接提：

**`microduck_rl`**：
1. `pyproject.toml` —— win32 的 torch/torchvision CUDA 路由（本文档第一节）
2. `tests/test_aarch64_cuda_torch.py` —— 原来的两个断言把"x86_64 必须留在 PyPI"写死了，而那正是 Windows 上不成立的那条不变量。改成表达真实意图：
   - aarch64 保持 CUDA 路由
   - 每条平台路由都必须指向 PyTorch 的 CUDA 索引
   - 必须留在 PyPI 的平台（linux-x86_64、macOS）确实还留在 PyPI
   - 新增 `test_win32_resolution_uses_a_cuda_index`
3. `tests/test_hf_jobs_flag.py`、`tests/test_publish_manifest.py`、`tests/test_aarch64_cuda_torch.py` —— `read_text()` 加 `encoding="utf-8"`
4. `tests/test_hf_jobs_flag.py` —— 读完整的 `train` 可执行文件而不是前 8KB
5. `tests/test_infer_policy_bam.py` —— 加 `skipif(sys.platform == "win32")`

**`microduck_challenges`**：同样的 `pyproject.toml` 路由改动。

---

## 七、本机实测结果

| 项目 | 结果 |
|---|---|
| `uv sync --frozen` | ✅ lock 与 pyproject 一致，重装不会退回 CPU 版 |
| `torch.cuda.is_available()` | ✅ True，`2.9.1+cu128` / CUDA 12.8 / `sm_120` / 15.9 GiB |
| Warp CUDA | ✅ CUDA 12.9，设备 `cuda:0` = RTX 5080，mempool enabled，kernel 实跑通过 |
| MuJoCo Warp kernel 编译 | ✅ 全部模块为 sm_120 编译成功（首次约 40s，之后走缓存） |
| 上游冒烟测试（64 envs / 5 iters） | ✅ 退出码 0，5 iters 用时 6 秒，无 NaN，所有惩罚项 ≤ 0 |
| ONNX 导出 | ✅ `[1,61] -> [1,14]`，图结构与官方策略逐节点一致 |
| 回放视频 | ✅ 100 帧 320×240 / 2 秒 @50Hz |
| 上游回归测试 | ✅ **272 passed, 4 skipped, 0 failed** |
