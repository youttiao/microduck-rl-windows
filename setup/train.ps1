# MicroDuck 训练脚本助手 (Windows / PowerShell)
#
# 用法:
#   .\train.ps1 list                        # 列出所有可训练任务
#   .\train.ps1 smoke <TASK_ID>             # 冒烟测试 (64 envs / 5 iters) -- 长跑之前必做
#   .\train.ps1 train <TASK_ID> [envs] [iters]   # 正式训练
#   .\train.ps1 export <TASK_ID> <ckpt.pt>  # 从本地 checkpoint 导出 ONNX
#   .\train.ps1 export-wandb <TASK_ID> <run_path>  # 从 wandb run 导出 ONNX
#   .\train.ps1 watch <TASK_ID> <run_path>  # 可视化回放 (需要 wandb)
#   .\train.ps1 test                        # 跑上游的 CPU 回归测试
#
# 例子:
#   .\train.ps1 smoke Mjlab-Velocity-Flat-MicroDuck
#   .\train.ps1 train Mjlab-Velocity-Flat-MicroDuck 4096
#   .\train.ps1 export Mjlab-Velocity-Flat-MicroDuck D:\duck\repos\microduck_rl\logs\rsl_rl\...\model_3000.pt
#
# 说明:
#   * 默认用 tensorboard 记日志, 不需要 wandb 账号。
#   * 想要 wandb (在线看曲线 / 跨机器续训) 就设环境变量 $env:MDUCK_LOGGER = "wandb",
#     并先执行  uv run wandb login  (或 HF 账号登录)。

param(
    [Parameter(Mandatory = $true, Position = 0)][string]$Command,
    [Parameter(Position = 1)][string]$Arg1 = "",
    [Parameter(Position = 2)][string]$Arg2 = "",
    [string]$Root = $(if ($env:DUCK_ROOT) { $env:DUCK_ROOT } else { "D:\duck" })
)

$ErrorActionPreference = "Continue"

$UvDir = "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\astral-sh.uv_Microsoft.Winget.Source_8wekyb3d8bbwe"
$env:PATH = "C:\Program Files\Git\cmd;$UvDir;" + $env:PATH
$env:UV_HTTP_TIMEOUT = "600"

$Repo   = Join-Path $Root "repos\microduck_rl"
$Uv     = "$UvDir\uv.exe"
$Logger = if ($env:MDUCK_LOGGER) { $env:MDUCK_LOGGER } else { "tensorboard" }

if (-not (Test-Path "$Repo\.venv\Scripts\python.exe")) {
    Write-Host "找不到 venv: $Repo\.venv" -ForegroundColor Red
    Write-Host "先跑一次:  cd $Repo; uv sync" -ForegroundColor Yellow
    exit 1
}

Set-Location $Repo

switch ($Command.ToLower()) {

    "list" {
        & $Uv run list-envs
    }

    "smoke" {
        if (-not $Arg1) { Write-Host "用法: .\train.ps1 smoke <TASK_ID>" -ForegroundColor Yellow; exit 1 }
        Write-Host "=== 冒烟测试 $Arg1 (64 envs / 5 iters, logger=$Logger) ===" -ForegroundColor Cyan
        & $Uv run train $Arg1 --env.scene.num-envs 64 --agent.max_iterations 5 --agent.logger $Logger
    }

    "train" {
        if (-not $Arg1) { Write-Host "用法: .\train.ps1 train <TASK_ID> [num_envs] [max_iterations]" -ForegroundColor Yellow; exit 1 }
        $envs   = if ($Arg2) { $Arg2 } else { "4096" }
        $iters  = if ($env:MDUCK_ITERS) { $env:MDUCK_ITERS } else { "3000" }
        Write-Host "=== 训练 $Arg1  (envs=$envs, iters=$iters, logger=$Logger) ===" -ForegroundColor Cyan
        Write-Host "输出目录: $Repo\logs\rsl_rl\" -ForegroundColor DarkGray
        & $Uv run train $Arg1 --env.scene.num-envs $envs --agent.max_iterations $iters --agent.logger $Logger
    }

    "export" {
        if (-not $Arg1 -or -not $Arg2) { Write-Host "用法: .\train.ps1 export <TASK_ID> <checkpoint.pt>" -ForegroundColor Yellow; exit 1 }
        Write-Host "=== 导出 ONNX (归一化器会烤进图里) ===" -ForegroundColor Cyan
        & $Uv run scripts/export.py $Arg1 --checkpoint-file $Arg2
    }

    "export-wandb" {
        if (-not $Arg1 -or -not $Arg2) { Write-Host "用法: .\train.ps1 export-wandb <TASK_ID> <entity/project/run_id>" -ForegroundColor Yellow; exit 1 }
        & $Uv run scripts/export.py $Arg1 --wandb-run-path $Arg2
    }

    "watch" {
        if (-not $Arg1 -or -not $Arg2) { Write-Host "用法: .\train.ps1 watch <TASK_ID> <entity/project/run_id>" -ForegroundColor Yellow; exit 1 }
        & $Uv run play $Arg1 --wandb-run-path $Arg2
    }

    "render" {
        if (-not $Arg1) { Write-Host "用法: .\train.ps1 render <policy.onnx> [秒数]" -ForegroundColor Yellow; exit 1 }
        $secs = if ($Arg2) { $Arg2 } else { "10" }
        & $Uv run scripts/render_policy.py $Arg1 --seconds $secs --video media\output.mp4
    }

    "test" {
        & $Uv run --with pytest pytest tests/
    }

    default {
        Write-Host "未知命令 '$Command'" -ForegroundColor Red
        Write-Host "可用: list | smoke | train | export | export-wandb | watch | render | test" -ForegroundColor Yellow
        exit 1
    }
}
