# 下载人体动作库到 D:\duck\data\
#
#   .\fetch_motions.ps1                 # 默认：CMU Mocap
#   .\fetch_motions.ps1 -Source cmu     # 只下 CMU
#   .\fetch_motions.ps1 -Source amass    # 只下 AMASS（需要先注册）
#   .\fetch_motions.ps1 -Source all
#
# 两个来源的许可完全不同，务必看清楚：
#
#   CMU Graphics Lab Motion Capture Facility
#     http://mocap.cs.cmu.edu/subjects.html
#     BVH 格式，**免费下载、无任何使用限制**，可以公开分享。
#     本仓库的 tools\bvh.py 能直接读。
#
#   AMASS (Archive of Motion Capture as Surface Shapes)
#     https://amass.is.tue.mpg.de/
#     质量最好（40+ 小时、11k+ 动作），但是：
#       - **需要注册账号**并同意许可才能下载
#       - 许可为 **学术非商用**，且 **禁止再分发**
#       - 所以即使重定向成鸭子动作，**公开分享仍然受限**
#     下载的是 SMPL 参数，要算关节位置需要 SMPL-X 身体模型；
#     用 HumanML3D 那种已经算好关节位置的版本就不需要。
#
# 想公开分享动作库，就用 CMU（或者自己录，走 OpenCap 那套手机管线）。

[CmdletBinding()]
param(
    [ValidateSet("cmu", "amass", "all")]
    [string]$Source = "cmu",
    [string]$Root = "D:\duck",
    [string[]]$CmuSubjects = @("01", "05", "49", "91"),
    [string]$AmassSubset = "CMU"
)

$ErrorActionPreference = "Continue"

# --- CMU -------------------------------------------------------------------
# The CMU Graphics Lab Motion Capture Facility, free and unrestricted.
# File layout: http://mocap.cs.cmu.edu/subjects/<subject>/<subject>_<take>.bvh
function Get-Cmu {
    $dest = Join-Path $Root "data\cmu_mocap"
    New-Item -ItemType Directory -Force -Path $dest | Out-Null
    foreach ($s in $CmuSubjects) {
        $take = 1
        while ($take -le 3) {
            $name = "{0}_{1:00}.bvh" -f $s, $take
            $url = "http://mocap.cs.cmu.edu/subjects/$s/$name"
            $out = Join-Path $dest $name
            if (Test-Path $out) { Write-Host "  skip $name"; $take++; continue }
            try {
                Invoke-WebRequest -Uri $url -OutFile $out -UseBasicParsing -TimeoutSec 180
                $kb = [math]::Round((Get-Item $out).Length / 1KB, 1)
                Write-Host "  OK   $name  $kb KB"
            } catch {
                Write-Host "  FAIL $name  ($($_.Exception.Message))"
                if (Test-Path $out) { Remove-Item $out -ErrorAction SilentlyContinue }
            }
            $take++
        }
    }
    Write-Host "CMU motions -> $dest"
}

# --- AMASS -----------------------------------------------------------------
# AMASS: register at https://amass.is.tue.mpg.de/ first, then download the
# archives by hand and drop the .npz files under data/amass/<subset>/.
# The layout the retargeter expects is a .npz with "joints"/"joint_names"/"fps",
# OR the raw AMASS SMPL .npz plus the SMPL-X body model.
function Get-Amass {
    $dest = Join-Path $Root "data\amass\$AmassSubset"
    New-Item -ItemType Directory -Force -Path $dest | Out-Null
    Write-Host ""
    Write-Host "AMASS needs an account:" -ForegroundColor Yellow
    Write-Host "  1. register at https://amass.is.tue.mpg.de/ and accept the licence" -ForegroundColor Yellow
    Write-Host "  2. download the '$AmassSubset' archive (a .zip of per-take .npz files)" -ForegroundColor Yellow
    Write-Host "  3. unzip it into $dest" -ForegroundColor Yellow
    Write-Host "  4. for joint POSITIONS (no SMPL-X body model needed), use the" -ForegroundColor Yellow
    Write-Host "     HumanML3D-style release instead: https://github.com/EricGuo5513/HumanML3D" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Then convert each motion to joint positions and retarget:" -ForegroundColor Yellow
    Write-Host "  uv run python tools\retarget_human.py --joints <motion.npz> --out <out.csv>" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "REMINDER: AMASS is non-commercial and forbids redistribution." -ForegroundColor Red
    Write-Host "         Retargeting does not clear that. For a shareable library use CMU." -ForegroundColor Red
}

switch ($Source) {
    "cmu" { Get-Cmu }
    "amass" { Get-Amass }
    "all" { Get-Cmu; Get-Amass }
}
