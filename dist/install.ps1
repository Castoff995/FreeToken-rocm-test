# Windows ROCm port - one-shot installer.
# Run with the Python interpreter you intend to serve from:
#   powershell -ExecutionPolicy Bypass -File dist\install.ps1 [-Python py -3.12]
# Assumes: AMD GPU present, driver installed, VS Build Tools (vcvarsall) available.
#
# IMPORTANT: every AMD wheel must come from the SAME nightly build stamp
# (rocm-sdk, torch rocmNNN, amd-torch-device-<gfx> ...), otherwise you get
# version-mismatch warnings and possible ABI breakage.

param(
    [string]$Py = "python",           # e.g. -Py "py -3.12"
    [string]$Arch = "gfx1201",        # device arch extra(s), comma-sep: device-gfx1031,device-gfx1201
    [string]$WheelDir = ""            # defaults to <repo>\rocm-wheels\<tag>
)
$ErrorActionPreference = "Stop"
$REPO = Split-Path -Parent $PSScriptRoot
$INDEX = "https://rocm.nightlies.amd.com/whl-multi-arch/"
$tag = ($Py -replace '\s', '') + "-" + $Arch.Replace(',', '+')
if (-not $WheelDir) { $WheelDir = Join-Path $REPO "rocm-wheels\$tag" }

Write-Host "== FreeToken Windows/ROCm installer ==" -ForegroundColor Cyan

if (-not $env:HIP_PATH) {
    Write-Warning "HIP_PATH is not set. Install the AMD ROCm runtime (TheRock build), set HIP_PATH to its root, then re-run."
}
& $Py --version

# 1a. Download the AMD wheel set (same nightly stamp guaranteed by the index)
New-Item -ItemType Directory -Force -Path $WheelDir | Out-Null
Write-Host "Downloading AMD wheels for '$Arch' into $WheelDir ..."
& $Py -m pip download --index-url $INDEX -d $WheelDir "rocm[libraries,devel,$Arch]"
if ($LASTEXITCODE -ne 0) { throw "pip download failed" }

# 1b. torch ROCm build + device module land in their own folders on the index
& $Py -m pip download --index-url "$INDEX/torch/" -d "$WheelDir\torch" torch --no-deps
& $Py -m pip download --index-url "$INDEX/amd-torch-device-$($Arch.Split(',')[0])/" -d "$WheelDir\torch" "amd-torch-device-$($Arch.Split(',')[0])"
if ($LASTEXITCODE -ne 0) { Write-Warning "torch wheel download failed - check folder names on $INDEX" }

# 2. Install AMD stack from the downloaded wheels (--no-deps keeps the stack consistent)
Write-Host "Installing AMD wheels from $WheelDir ..."
& $Py -m pip install (Get-ChildItem $WheelDir -Recurse -Filter *.whl | ForEach-Object { $_.FullName }) --no-deps --force-reinstall

# 3. Triton (AMD backend), tvm-ffi, misc runtime deps (plain PyPI)
& $Py -m pip install "triton-windows>=3.7.1" apache-tvm-ffi==0.1.13.post3 msgpack pyzmq psutil requests aiohttp partial_json_parser gguf

# 4. FreeToken without CUDA extensions
$env:FREETOKEN_SKIP_CUDA_EXT = "1"
& $Py -m pip install -e "$REPO" --no-deps --no-build-isolation
Remove-Item Env:FREETOKEN_SKIP_CUDA_EXT

# 5. Patch tvm-ffi / triton / uvicorn for Windows+ROCm
& $Py "$REPO\dist\patch_upstream.py"

# 6. Smoke test
Write-Host "`n== Smoke test ==" -ForegroundColor Cyan
& $Py -c "import torch; print('torch:', torch.__version__, '| hip:', torch.version.hip); print('gpu:', torch.cuda.get_device_name(0))"

Write-Host @"

Done. Start the server with:
    powershell -File dist\run-server.ps1 -Model <path-to-model>
Then open the chat UI at http://localhost:1420
"@ -ForegroundColor Green
