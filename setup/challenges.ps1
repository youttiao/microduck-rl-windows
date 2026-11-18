# MicroDuck 官方竞技场挑战 —— 独立环境 (6 个现成的自定义任务模板)
#
# 这是一个独立的 uv 项目, 有自己的 venv 和锁文件, 和 microduck_rl 不共用。
# 用途: 照抄一个挑战目录 (sprint_2m / hill_climb_1m5 / roller_sprint_2m /
#       tug_of_war / sock_heist_6m / got_a_talent) 改成朋友想要的动作。
#
# 用法:
#   .\challenges.ps1 sync       # 安装 (首次 ~4GB, torch 走缓存)
#   .\challenges.ps1 list       # 列出挑战的任务 ID
#   .\challenges.ps1 smoke <TASK_ID>
#   .\challenges.ps1 train <TASK_ID> [envs] [iters]
#   .\challenges.ps1 score <policy.onnx> <event>   # 用 Arena 的规则打分

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

$Repo = Join-Path $Root "repos\microduck_challenges"
$Uv   = "$UvDir\uv.exe"

if (-not (Test-Path $Repo)) {
    Write-Host "找不到 $Repo" -ForegroundColor Red
    exit 1
}

Set-Location $Repo

switch ($Command.ToLower()) {

    "sync" {
        # workshop 组 = Arena 的评测器 / 规则 / 评分工具
        & $Uv sync --group workshop
    }

    "list" {
        & $Uv run list-envs
    }

    "smoke" {
        if (-not $Arg1) { Write-Host "用法: .\challenges.ps1 smoke <TASK_ID>" -ForegroundColor Yellow; exit 1 }
        & $Uv run train $Arg1 --env.scene.num-envs 2 --agent.max_iterations 2 `
            --agent.logger tensorboard --agent.run-name smoke
    }

    "train" {
        if (-not $Arg1) { Write-Host "用法: .\challenges.ps1 train <TASK_ID> [envs] [iters]" -ForegroundColor Yellow; exit 1 }
        $envs  = if ($Arg2) { $Arg2 } else { "4096" }
        $iters = if ($env:MDUCK_ITERS) { $env:MDUCK_ITERS } else { "3000" }
        & $Uv run train $Arg1 --env.scene.num-envs $envs --agent.max_iterations $iters `
            --agent.seed 1 --agent.logger tensorboard --agent.run-name $env:MDUCK_RUN
    }

    "score" {
        if (-not $Arg1 -or -not $Arg2) { Write-Host "用法: .\challenges.ps1 score <policy.onnx> <event>" -ForegroundColor Yellow; exit 1 }
        & $Uv run --group workshop arena score $Arg1 --event $Arg2
    }

    "check" {
        # got_a_talent 专用: 逐个关键帧检查, 最后给 PASS/FAIL
        if (-not $Arg1) { Write-Host "用法: .\challenges.ps1 check <policy.onnx> [timeline.json]" -ForegroundColor Yellow; exit 1 }
        if ($Arg2) {
            & $Uv run --group workshop python -m microduck_challenges.got_a_talent.check $Arg1 --timeline $Arg2
        } else {
            & $Uv run --group workshop python -m microduck_challenges.got_a_talent.check $Arg1
        }
    }

    "test" {
        & $Uv run --group workshop pytest
    }

    default {
        Write-Host "可用: sync | list | smoke | train | score | check | test" -ForegroundColor Yellow
        exit 1
    }
}
