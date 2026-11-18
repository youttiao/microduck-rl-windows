# 从 HuggingFace 下载 Pollen Robotics 官方发布的模型资产
#
#   .\fetch_hf_assets.ps1                  # 装到 <Root>\data\hf\
#   .\fetch_hf_assets.ps1 -Root E:\duck     # 换个根目录
#
# 拉两样东西:
#   - microduck-policies : 10 个官方 ONNX 策略 + manifest.json (schema 2)
#   - microduck-training: 9 个训练包 (.pt checkpoint + wandb 配置 + provenance)
#                          —— warm-start 用, 可以从官方策略继续训
#
# 校验: 每个 .pt 的 SHA256 会和上游 bundle.json 里记录的 checkpoint_sha256
#       逐个比对, 对不上会报出来。

[CmdletBinding()]
param(
    [string]$Root = "D:\duck"
)

$ErrorActionPreference = "Continue"
$base = Join-Path $Root "data\hf"

$repos = @(
  @{ repo = "pollen-robotics/microduck-policies";  dir = "microduck-policies"  },
  @{ repo = "pollen-robotics/microduck-training"; dir = "microduck-training" }
)

foreach ($r in $repos) {
  $dest = Join-Path $base $r.dir
  Write-Host "=== $($r.repo) -> $dest ===" -ForegroundColor Cyan

  $api = "https://huggingface.co/api/models/$($r.repo)/tree/main?recursive=true"
  $tree = Invoke-RestMethod -Uri $api -TimeoutSec 60
  $files = $tree | Where-Object { $_.type -eq "file" -and $_.path -notlike ".gitattributes" }

  foreach ($f in $files) {
    $out = Join-Path $dest ($f.path -replace '/', '\')
    $outDir = Split-Path $out -Parent
    if (-not (Test-Path $outDir)) { New-Item -ItemType Directory -Force -Path $outDir | Out-Null }
    if (Test-Path $out) { Write-Host "  skip  $($f.path)"; continue }
    try {
      $url = "https://huggingface.co/$($r.repo)/resolve/main/$($f.path)"
      Invoke-WebRequest -Uri $url -OutFile $out -UseBasicParsing -TimeoutSec 300
      Write-Host ("  OK    {0,-58} {1} KB" -f $f.path, [math]::Round((Get-Item $out).Length / 1KB, 1))
    } catch {
      Write-Host ("  FAIL  {0,-58} {1}" -f $f.path, $_.Exception.Message)
    }
  }
}

# ---------------------------------------------------------------- 校验
Write-Host ""
Write-Host "=== 校验训练包的 SHA256 ===" -ForegroundColor Cyan
$trDir = Join-Path $base "microduck-training\bundles"
$checked = 0; $bad = 0
Get-ChildItem $trDir -Directory -ErrorAction SilentlyContinue | ForEach-Object {
  $exp = $_.Name
  Get-ChildItem $_.FullName -Directory | ForEach-Object {
    $bundle = Join-Path $_.FullName "bundle.json"
    if (Test-Path $bundle) {
      $j = Get-Content $bundle -Raw | ConvertFrom-Json
      $pt = Get-ChildItem $_.FullName -Filter "model_*.pt" | Select-Object -First 1
      if ($pt) {
        $actual = (Get-FileHash $pt.FullName -Algorithm SHA256).Hash.ToLower()
        $checked++
        if ($actual -eq $j.checkpoint_sha256) {
          Write-Host ("  MATCH    {0}/{1}" -f $exp, $_.Name) -ForegroundColor Green
        } else {
          $bad++
          Write-Host ("  MISMATCH {0}/{1}" -f $exp, $_.Name) -ForegroundColor Red
          Write-Host ("           expected {0}" -f $j.checkpoint_sha256)
          Write-Host ("           actual   {0}" -f $actual)
        }
      }
    }
  }
}
Write-Host ""
Write-Host "校验完成: $checked 个包, $bad 个不匹配" -ForegroundColor $(if ($bad -eq 0) { "Green" } else { "Yellow" })
