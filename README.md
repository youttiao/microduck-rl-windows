# MicroDuck RL 训练环境（Windows）

在 **Windows + NVIDIA RTX 50 系** 上把 [Pollen Robotics MicroDuck](https://pollen-robotics.com/microduck/) 的官方强化学习训练栈跑起来，并训练自己的动作模型。

MicroDuck 是一只 25 cm 高、800 g、14 个 Dynamixel XL330 舵机的双足小鸭机器人（"🦆"）。官方把**训练栈、仿真器和已训好的策略全部开源了**（Apache-2.0），而且设计上就是让你自己训：

> A 25 cm open-source biped you train yourself with reinforcement learning.

本仓库是**在一台 Windows 机器上实测跑通**之后整理出来的：把上游 Linux-only 的部分补齐成 Windows 可用，附上一键搭建脚本、踩坑清单，以及一份社区生态调研。

---

## 🚨 最重要的一件事：上游在 Windows 上是坏的

直接 `uv sync` 会装上一个 **CPU 版 torch**，然后训练第一步就崩：

```
IndexError: list index out of range
  at mjlab/utils/gpu.py:70
```

原因：上游的 `AGENTS.md` 记过一个 linux-aarch64 上 PyPI 的 torch 是 CPU-only 的坑，所以用 `[tool.uv.sources]` 把 torch 指向了 CUDA 索引，注释里写"x86_64 留在 PyPI 就行"。**这个判断在 Linux 上对，在 Windows 上不对** —— Linux 的 torch 靠 `nvidia-*-cu12` 依赖包带 CUDA，Windows 的 torch wheel 是自包含的，而 PyPI 发的 Windows 版恰好是 CPU 那一个。

本仓库的 `patches/` 里有修好的版本（给 win32 加一条 cu128 索引路由）。**细节和原理见 [`docs/Windows训练环境.md`](docs/Windows训练环境.md)。**

好消息：**不需要装系统级 CUDA Toolkit。** Warp 的 Windows wheel 自带 CUDA 12.9，PyTorch cu128 wheel 自带 12.8，机器上的驱动全覆盖。

---

## ⚡ 一键搭建

```powershell
# 前提: Git / Python 3.12 / uv
winget install Git.Git Python.Python.3.12 astral-sh.uv

git clone <这个仓库>
cd microduck-rl-workspace
.\scripts\bootstrap.ps1
```

`bootstrap.ps1` 会 clone 上游两个仓库 → 打上 Windows 补丁 → 建 venv 装依赖 → 从 HuggingFace 拉官方策略和训练包 → 跑一次自检。第一次要下约 4 GB。

想装到别的盘：`.scripts\bootstrap.ps1 -Root E:\duck`

---

## 📖 文档

| 文档 | 内容 |
|---|---|
| **[`docs/工具链总览.md`](docs/工具链总览.md)** | **先读这个。** 从机器人本体 → Rust 运行时 → ONNX 契约 → 训练栈 → 仿真层 → 资产分发，一层层讲清楚这套东西怎么串起来的 |
| **[`docs/人体动作迁移.md`](docs/人体动作迁移.md)** | **把人体动作库迁到鸭子**：上游已有的（未合并的）动作跟踪环境、人体→鸭子的重定向器、动作库选型的许可问题、"一个动作一个策略再堆起来"怎么做 |
| [`docs/Windows训练环境.md`](docs/Windows训练环境.md) | Windows 上的 5 个坑、原理、修复，以及哪些功能确实不可用 |
| [`docs/训练避坑清单.md`](docs/训练避坑清单.md) | 从上游 `AGENTS.md` 提炼的中文避坑清单（奖励函数符号约定、curriculum、warm-start…）**改奖励函数前必读** |
| [`docs/社区资源调研.md`](docs/社区资源调研.md) | **688 个 microduck 仓库**的分类调研：可下载的社区策略、其它仿真器移植、硬件复刻、ROS/导航集成 |
| [`docs/community-policies.json`](docs/community-policies.json) | 45 条社区策略的机器可读清单（18 个 GitHub 源码仓 + 27 个 HF 可部署策略） |

---

## 🤖 人体动作 → 鸭子的动作策略

上游有两个**未合并**的分支带着完整的动作跟踪环境（`expressive-motions` / `dance-steps`，2026-10-30），但**没有任何人体动作库的导入器** —— 那正是缺的那块。本仓库补上了：

```
tools/
├── retarget_human.py        人体 3D 关节位置 → 上游的 "Tremendous" 参考通道 CSV
├── duck_head_ik.py          头部姿态 → 4 个颈部关节角（对着真 MJCF 数值求解，不是手推公式）
├── make_motion_library.py   批量：目录 → 一堆 CSV + 自动注册成新任务
├── soma_to_motionlib.py     NVIDIA SOMA 的 CSV → 同一格式（根旋转读法自动标定）
├── bvh.py                   BVH 解析（CMU Mocap 的格式）：层级 + 正向运动学
├── test_retarget.py         地面真值测试：躯干/头/高度三个通道
├── test_retarget_full.py    全身测试：接触表交替、单步位移、弗劳德时间缩放
└── test_bvh.py              BVH 正向运动学逐值测试
```

```powershell
# 1) 重定向 + 注册（一个动作 = 一个新任务）
uv run python tools/make_motion_library.py --src motions/ `
    --out <repo>\src\mjlab_microduck\tasks\motion_data --register

# 1b) 全身带脚步的版本（加 --full-body，注册成 StepMotion）
uv run python tools/make_motion_library.py --src motions/ --full-body --register

# 1c) 从 NVIDIA SOMA 的输出转过来
uv run python tools/soma_to_motionlib.py --csv soma.csv --out out.csv --full-body

# 2) 训练（每个动作一个任务）
uv run train Mjlab-Expressive-MyMotion-MicroDuck --env.scene.num-envs 4096

# 3) 导出 ONNX，4) 装到机器人
uv run scripts/export.py Mjlab-Expressive-MyMotion-MicroDuck --checkpoint-file <model.pt>
sudo robotctl policy add my-motion <user>/microduck-my-motion
```

**已经在本机端到端验证**（两条路都通了，训练用 GPU 跑的冒烟测试）：
- **A 自建重定向**：人体动作 → 重定向 → 注册 → `DemoSway` / `DemoWalk` 出现 → 训练退出码 0 → 导出 ONNX，**图结构与官方策略逐节点一致**
- **B NVIDIA SOMA**：SOMA CSV → 重定向 → 注册 → `SomaWalkForward` 出现 → 训练退出码 0，四个 step 项符号全对

重定向精度：躯干通道 ≤0.11°、头部姿态 0.001°、高度 0.016 mm（地面真值测试）。

> **完整训练（7000 iters @ 4096 envs）没有跑** —— 按你的要求不再用显卡训练。上面的冒烟测试是之前跑过的（5 iters / 64 envs，退出码 0），说明环境是通的；真要训的时候直接 `.\setup\train.ps1 train <TASK_ID> 4096` 就行。CPU 侧（重定向、测试、导出）全部照常可用。

**关键设计**：相位 φ 走指令通道 `[cos 2πφ, sin 2πφ, 0]`，**策略不负责计时**（上游的教训：让策略自己数拍子，十次全挂）。所以 61 维观测契约不变，训出来的策略照样能和走路/站立策略热切换。

**动作库选型有许可坑**：AMASS 系（HumanML3D / BABEL / Motion-X）是**非商用且禁止再分发**的，重定向之后公开分享仍然受限。要能公开分享就用 **OpenCap（CC BY 4.0）**、**CMU Mocap（免费无限制）**，或者文本生成动作。详见 [`docs/人体动作迁移.md`](docs/人体动作迁移.md) 第五节。


---

## 🎮 常用命令

```powershell
.\setup\train.ps1 list                                   # 列所有可训练任务 (33 个 MicroDuck)
.\setup\train.ps1 smoke Mjlab-Velocity-Flat-MicroDuck    # 冒烟测试 —— 长跑之前必做
.\setup\train.ps1 train Mjlab-Velocity-Flat-MicroDuck 4096   # 正式训练, 约 1-2 小时
.\setup\train.ps1 export Mjlab-Velocity-Flat-MicroDuck <model_N.pt>   # 导出 ONNX
.\setup\train.ps1 watch  Mjlab-Velocity-Flat-MicroDuck <wandb_run>    # 回放 (起浏览器 viewer)
.\setup\train.ps1 test                                   # 跑上游回归测试
```

竞技场挑战环境（独立 venv）：

```powershell
.\setup\challenges.ps1 sync          # 6 个现成的自定义任务模板
.\setup\challenges.ps1 list
.\setup\challenges.ps1 score <policy.onnx> got-a-talent
```

环境自检：

```powershell
D:\duck\repos\microduck_rl\.venv\Scripts\python.exe -I .\setup\check_env.py
```

---

## 🤖 想做一个自己的动作？

MicroDuck 的观测/动作契约是**固定**的：**61 维观测 → 14 维关节目标，50 Hz**。所有策略共享同一套布局，这就是为什么真机运行时能在走路/站立/把戏之间热切换。

上游反复强调的两条：

> **时间轴负责计时，策略不负责。** 早期版本让策略自己数拍子做坐下-站起，十次全挂 —— PPO 的探索噪声几秒内就把关节里的时钟冲垮了。**任何新动作都必须由指令触发。**

> **新动作需要它自己的指令通道。** 13 个指令数字是定死的。**空着的槽位：`twist[1]`、`twist[2]` 和 `body`（6 个）。**

所以路径是：

1. **先看现成的 33 个任务里有没有能改的** —— 改参数比重训快一个数量级
2. 真要新动作：**占一个空指令槽位**（比如 `body[0]`），而不是改观测布局
3. 写奖励函数 —— 照 `docs/训练避坑清单.md` 的规则来，**最要命的是符号约定**：mjlab 自带的 cost 函数返回 ≥ 0 → 权重取负；microduck 自己写的 `*_penalty` 返回 ≤ 0 → 权重取正。搞反了策略会专门去刷分。**万能检查：wandb 里每个 `Episode_Reward/<penalty>` 必须 ≤ 0**
4. 冒烟测试 → 训练 → **看视频** → 才上真机

---

## 🦆 训完怎么装到机器人

产物是一个 ONNX（形状固定 `[1,61] -> [1,14]`），真机上 `robotctl` 直接加载：

```bash
# 试自己的文件 (不用等发布, 不用重启)
sudo robotctl policy load walk /home/microduck/my_walking.onnx

# 加成一个技能, 按名字触发
sudo robotctl policy add my-move <user>/microduck-my-move
robotctl robot do my-move

# 绑到手柄按键 (X 键出厂是空的)
sudo robotctl pad bind x my-move
```

**第一次务必把鸭子放在支架上。** 完整 `robotctl` 手册在 `microduck` 仓库的 `docs/robot/cheatsheet.md`。

> ⚠️ **只能用 `scripts/export.py` 导出。** 它把观测归一化烤进 ONNX 图里 —— 手搓 checkpoint 转 ONNX，真机上策略看到的是没归一化的数，行为完全不同。

---

## ✅ 实测结果（本机，RTX 5080 Laptop）

| 项目 | 结果 |
|---|---|
| `uv sync --frozen` | ✅ lock 与 pyproject 一致，重装不会退回 CPU 版 |
| `torch.cuda.is_available()` | ✅ `2.9.1+cu128` / CUDA 12.8 / `sm_120` / 15.9 GiB |
| Warp CUDA | ✅ CUDA 12.9，`cuda:0` = RTX 5080，mempool enabled，kernel 实跑通过 |
| MuJoCo Warp kernel 编译 | ✅ 全部模块为 sm_120 编译成功 |
| **上游冒烟测试**（64 envs / 5 iters） | ✅ 退出码 0，5 iters 用时 6 秒，无 NaN，所有惩罚项 ≤ 0 |
| **ONNX 导出** | ✅ `[1,61] -> [1,14]`，图结构与官方策略**逐节点一致** |
| **回放视频** | ✅ 100 帧 320×240 / 2 秒 @50Hz |
| **上游回归测试** | ✅ **390 passed, 4 skipped, 0 failed**（含上游原有 264 个 + 表达性/踏步/ground-grasp 的 78 个 + 本仓库新加的 14 个 + Windows 兼容修复） |
| 官方资产校验 | ✅ 9 个训练包的 SHA256 **全部**与上游记录一致 |
| 补丁可应用性 | ✅ 两个 patch 在**全新的上游 clone 上** `git apply --check` 干净通过 |

---

## 📦 这个仓库里有什么

```
microduck-rl-workspace/
├── README.md                    # 你正在读的
├── docs/
│   ├── 工具链总览.md             # 从本体到训练脚本的完整拆解
│   ├── 人体动作迁移.md           # 人体动作库 → 鸭子：重定向器 + 动作库 + 堆起来
│   ├── Windows训练环境.md        # Windows 上的坑与修法
│   ├── 训练避坑清单.md           # 上游 AGENTS.md 的中文提炼
│   ├── 社区资源调研.md           # 688 个仓库的分类调研
│   └── community-policies.json   # 45 条社区策略清单
├── setup/
│   ├── env.ps1                  # 配置 PATH
│   ├── train.ps1                # 训练/冒烟/导出/回放 封装
│   ├── challenges.ps1           # 竞技场挑战环境封装
│   ├── check_env.py             # 环境自检
│   └── fetch_hf_assets.ps1      # 拉官方策略和训练包
├── tools/                       # 人体动作 → 鸭子的重定向
│   ├── retarget_human.py
│   ├── duck_head_ik.py
│   ├── make_motion_library.py
│   └── test_retarget.py
├── scripts/
│   └── bootstrap.ps1            # 一键搭建
└── patches/
    ├── microduck_rl-windows-and-expressive.patch
    └── microduck_challenges-windows-cu128.patch
```

`patches/` 里的两个 patch 是**可以直接提给上游的** —— 它们不只修好了 Windows，还把上游那条"x86_64 必须留在 PyPI"写死的测试断言改成了表达真实意图。

`microduck_rl-windows-and-expressive.patch` 是完整的一份：`origin/develop` → 本机分支，包含

- **Windows CUDA 路由**（win32 → torch/torchvision 的 cu128 索引）
- **上游未合并的 `expressive-motions` 和 `dance-steps` 两个分支**（表达性动作 + 带脚步的训练环境，17+2 个新任务）
- **三个用真实重定向做出来的样例动作**（`demo_sway` / `demo_walk` / `soma_walk_forward`），当回归测试用
- **上游测试的 Windows 兼容修复**（编码、`train` 可执行文件检查）

用法（对**全新的**上游 clone）：

```bash
git clone https://github.com/pollen-robotics/microduck_rl.git
cd microduck_rl
git apply /path/to/microduck_rl-windows-and-expressive.patch
uv sync
uv run --with pytest pytest tests/ -q     # 390 passed, 4 skipped
```

两个 patch 都验证过能在**全新的 clone 上干净应用**。

---

## 📄 许可

本仓库的脚本和文档：MIT。

**注意**：上游 `pollen-robotics/microduck_rl` 和 `pollen-robotics/microduck` 是 Apache-2.0，但**机器人的 3D 模型文件（`src/mjlab_microduck/robot/microduck/assets/*.stl`）是 CC BY-NC-SA 4.0（非商业）**。复刻、打印、改造要留意这条。
