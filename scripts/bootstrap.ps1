# MicroDuck 训练环境一键搭建 (Windows / PowerShell)
#
#   .\bootstrap.ps1                 # 装到默认位置 D:\duck
#   .\bootstrap.ps1 -Root E:\duck    # 换个盘
#
# 做这几件事:
#   1. 检查/安装 Git、Python 3.12、uv
#   2. clone 上游两个仓库 (microduck_rl / microduck_challenges)
#   3. 打上本仓库的 Windows CUDA 路由补丁
#   4. uv sync 建 venv (microduck_rl + microduck_challenges)
#   5. 从 HuggingFace 拉官方 ONNX 策略和训练包
#   6. 跑一次环境自检
#
# 幂等: 已经做过的地方会跳过。

[CmdletBinding()]
param(
    [string]$Root = "D:\duck",
    [switch]$SkipChallenges,
    [switch]$SkipAssets
)

$ErrorActionPreference = "Continue"
$ProgressPreference = "SilentlyContinue"

function Step($n, $msg) { Write-Host ""; Write-Host "[$n] $msg" -ForegroundColor Cyan }
function Ok($msg) { Write-Host "    OK  $msg" -ForegroundColor Green }
function Warn($msg) { Write-Host "    ..  $msg" -ForegroundColor DarkGray }
function Die($msg) { Write-Host "    ERR $msg" -ForegroundColor Red; exit 1 }

$RepoRoot = Split-Path -Parent $PSScriptRoot
$Patches  = Join-Path $RepoRoot "patches"

# ---------------------------------------------------------------- 1. 工具链
Step 1 "检查工具链 (git / python 3.12 / uv)"

$gitCmd = (Get-Command git -ErrorAction SilentlyContinue).Source
if (-not $gitCmd) {
    $c = "C:\Program Files\Git\cmd\git.exe"
    if (Test-Path $c) { $gitCmd = $c; $env:PATH = "C:\Program Files\Git\cmd;$env:PATH" }
}
if (-not $gitCmd) {
    Write-Host "    需要 Git: winget install Git.Git" -ForegroundColor Yellow
    Die "找不到 git"
}
Ok "git    $((& $gitCmd --version))"

$uvExe = Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" -Recurse -Filter "uv.exe" -ErrorAction SilentlyContinue |
         Select-Object -First 1 -ExpandProperty FullName
if (-not $uvExe) {
    Write-Host "    需要 uv: winget install astral-sh.uv" -ForegroundColor Yellow
    Die "找不到 uv"
}
$env:PATH = (Split-Path $uvExe -Parent) + ";" + $env:PATH
Ok "uv     $(& $uvExe --version)"

$pyOk = $false
foreach ($p in @("$env:LOCALAPPDATA\Programs\Python\Python312\python.exe")) {
    if (Test-Path $p) { $pyOk = $true; $env:PATH = (Split-Path $p -Parent) + ";" + $env:PATH }
}
if (-not $pyOk) { Warn "没找到 Python 3.12 —— uv 会自己下载一个" }
else { Ok "python $((python --version) 2>&1)" }

# uv 首次同步要下 ~4GB, 默认 30s 超时会断
$env:UV_HTTP_TIMEOUT = "600"
$env:UV_LINK_MODE   = "copy"

# ---------------------------------------------------------------- 2. clone
Step 2 "clone 上游仓库"
New-Item -ItemType Directory -Force -Path $Root | Out-Null
$reposDir = Join-Path $Root "repos"
New-Item -ItemType Directory -Force -Path $reposDir | Out-Null

function Clone-Repo($name, $url, $branch) {
    $dest = Join-Path $reposDir $name
    if (Test-Path (Join-Path $dest ".git")) { Warn "$name 已存在, 跳过 clone"; return $dest }
    & $gitCmd clone --quiet --depth 1 --branch $branch $url $dest
    if ($LASTEXITCODE -ne 0) { Die "clone $name 失败" }
    Ok "$name"
    return $dest
}

$rl = Clone-Repo "microduck_rl" "https://github.com/pollen-robotics/microduck_rl.git" "develop"
$ch = $null
if (-not $SkipChallenges) {
    $ch = Clone-Repo "microduck_challenges" "https://github.com/pollen-robotics/microduck-challenges.git" "main"
}

# ---------------------------------------------------------------- 3. 补丁
Step 3 "打 Windows CUDA 路由补丁"
function Apply-Patch($repo, $patchFile, $label) {
    if (-not (Test-Path $patchFile)) { Warn "$label: 找不到 $patchFile, 跳过"; return }
    Push-Location $repo
    & $gitCmd apply --check $patchFile 2>$null
    if ($LASTEXITCODE -eq 0) {
        & $gitCmd apply $patchFile
        Ok "$label 已应用"
    } else {
        Warn "$label 看起来已经打过, 跳过"
    }
    Pop-Location
}
Apply-Patch $rl (Join-Path $Patches "0001-microduck_rl-windows-cu128.patch")      "microduck_rl"
if ($ch) { Apply-Patch $ch (Join-Path $Patches "0001-microduck_challenges-windows-cu128.patch") "microduck_challenges" }

# ---------------------------------------------------------------- 4. venv
Step 4 "建 venv 并装依赖 (会下 ~4GB, 第一次比较久)"
Push-Location $rl
& $uvExe lock --python 3.12
& $uvExe sync --python 3.12
if ($LASTEXITCODE -ne 0) { Pop-Location; Die "microduck_rl 的 uv sync 失败" }
Ok "microduck_rl"
Pop-Location

if ($ch) {
    Push-Location $ch
    & $uvExe lock --python 3.12
    & $uvExe sync --python 3.12 --group workshop
    if ($LASTEXITCODE -ne 0) { Pop-Location; Warn "microduck_challenges 的 uv sync 失败 (不影响主环境)" }
    else { Ok "microduck_challenges" }
    Pop-Location
}

# ---------------------------------------------------------------- 5. 资产
if (-not $SkipAssets) {
    Step 5 "拉取官方 ONNX 策略和训练包"
    $assetScript = Join-Path $RepoRoot "setup\fetch_hf_assets.ps1"
    if (Test-Path $assetScript) {
        & $assetScript
        Ok "资产下载完成"
    } else {
        Warn "找不到 $assetScript, 跳过"
    }
}

# ---------------------------------------------------------------- 6. 自检
Step 6 "环境自检"
$py = Join-Path $rl ".venv\Scripts\python.exe"
$check = Join-Path $RepoRoot "setup\check_env.py"
if (Test-Path $py) {
    & $py -I $check
    if ($LASTEXITCODE -eq 0) { Ok "自检全部通过" } else { Warn "自检有失败项, 看上面的输出" }
} else {
    Warn "找不到 $py"
}

Write-Host ""
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "  搭好了。" -ForegroundColor Green
Write-Host ""
Write-Host "  列任务:      $Root\setup\train.ps1 list"
Write-Host "  冒烟测试:    $Root\setup\train.ps1 smoke Mjlab-Velocity-Flat-MicroDuck"
Write-Host "  正式训练:    $Root\setup\train.ps1 train Mjlab-Velocity-Flat-MicroDuck 4096"
Write-Host "  导出 ONNX:   $Root\setup\train.ps1 export <TASK_ID> <model_N.pt>"
Write-Host "  回放:        $Root\setup\train.ps1 watch <TASK_ID> <wandb_run_path>"
Write-Host ""
Write-Host "  文档:        $RepoRoot\docs\"
Write-Host "=" * 70 -ForegroundColor Cyan
