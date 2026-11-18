# MicroDuck 环境引导脚本 (Windows / PowerShell)
# 用法:  在 PowerShell 中执行  .  D:\duck\setup\env.ps1
#        或直接运行  D:\duck\setup\env.ps1
# 作用:  把 git / uv / python 加进 PATH, 并设置 uv 的网络超时环境变量

$GitDir     = "C:\Program Files\Git\cmd"
$PythonDir  = "$env:LOCALAPPDATA\Programs\Python\Python312"
$PythonScr  = "$env:LOCALAPPDATA\Programs\Python\Python312\Scripts"
$UvDir      = "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\astral-sh.uv_Microsoft.Winget.Source_8wekyb3d8bbwe"

$env:PATH = "$GitDir;$PythonDir;$PythonScr;$UvDir;$env:PATH"

# uv 首次同步要下载 ~2GB 的 CUDA wheel, 默认 30s 超时会中断
$env:UV_HTTP_TIMEOUT = "600"
$env:UV_LINK_MODE   = "copy"

# 训练日志目录 (mjlab 默认写 logs/, 这里指到 D:\duck\runs)
$env:MJLAB_LOG_DIR  = "D:\duck\runs"

function Get-DuckPython {
    # microduck_rl 的解释器
    "D:\duck\repos\microduck_rl\.venv\Scripts\python.exe"
}

function Get-DuckUv {
    "$UvDir\uv.exe"
}

Write-Host ""
Write-Host "  git    : $((git --version) 2>&1)"
Write-Host "  uv     : $((& $UvDir\uv.exe --version) 2>&1)"
Write-Host "  python : $((python --version) 2>&1)"
Write-Host "  duck   : $(Get-DuckPython)"
Write-Host ""
