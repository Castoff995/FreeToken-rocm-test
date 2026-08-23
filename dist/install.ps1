# ============================================================
#  FreeToken for Windows + AMD GPUs - one-command installer
#
#  What this does (all automatic):
#    1. makes a private Python environment (.venv) so nothing
#       else on your PC gets touched
#    2. installs the AMD GPU torch/ROCm wheels you downloaded
#    3. installs the engine + its small helper packages
#    4. patches 3 upstream bugs (automatic, safe to re-run)
#
#  Run it from inside the cloned repo folder:
#    powershell -ExecutionPolicy Bypass -File dist\install.ps1
#
#  Need help? Check PORT_REQUIREMENTS.md first.
# ============================================================
param(
    [string]$Py = "py -3.12",         # leave as-is if you installed Python 3.12 normally
    [string]$Arch = "gfx1201",        # your GPU family: gfx1201 = RX 9070 XT
    [string]$WheelDir = ""            # folder holding AMD .whl files (see below)
)
$ErrorActionPreference = "Stop"
$REPO = Split-Path -Parent $PSScriptRoot
$INDEX = "https://rocm.nightlies.amd.com/whl-multi-arch/"
$tag = ($Py -replace '\s', '') + "-" + $Arch.Replace(',', '+')
if (-not $WheelDir) { $WheelDir = Join-Path $REPO "rocm-wheels\$tag" }
$VENV = Join-Path $REPO ".venv"
$PYEXE = "$VENV\Scripts\python.exe"
$PIP   = "$VENV\Scripts\python.exe -m pip"

Write-Host ""
Write-Host "  FreeToken installer for Windows + AMD" -ForegroundColor Cyan
Write-Host "  --------------------------------------"

# ---- Step 1: private python environment -------------------------------
Write-Host "`n[1/5] Creating a private Python environment (.venv) ..." -ForegroundColor Yellow
Invoke-Expression "$Py -m venv `"$VENV`""

# ---- Step 2: AMD GPU wheels -------------------------------------------
Write-Host "[2/5] AMD GPU wheels (torch / ROCm) ..." -ForegroundColor Yellow
if (-not (Test-Path "$WheelDir") -or -not (Get-ChildItem "$WheelDir" -Filter *.whl -ErrorAction SilentlyContinue)) {
    Write-Host "      downloading AMD wheels (~2 GB, one time only) ..." -ForegroundColor Gray
    New-Item -ItemType Directory -Force -Path $WheelDir | Out-Null
    Invoke-Expression "$Py -m pip download --index-url $INDEX -d `"$WheelDir`" `"rocm[libraries,devel,device-$Arch]`""
}
$PIP install (Get-ChildItem $WheelDir -Recurse -Filter *.whl | ForEach-Object { $_.FullName }) --no-deps --force-reinstall

# ---- Step 3: engine + helpers -----------------------------------------
Write-Host "[3/5] Installing FreeToken + helpers ..." -ForegroundColor Yellow
$PIP install "triton-windows>=3.7.1" apache-tvm-ffi==0.1.13.post3 msgpack pyzmq psutil requests aiohttp partial_json_parser gguf
$env:FREETOKEN_SKIP_CUDA_EXT = "1"
$PIP install -e "$REPO" --no-deps --no-build-isolation
Remove-Item Env:FREETOKEN_SKIP_CUDA_EXT

# ---- Step 4: upstream patches -----------------------------------------
Write-Host "[4/5] Applying 3 small compatibility patches ..." -ForegroundColor Yellow
& $PYEXE "$REPO\dist\patch_upstream.py"

# ---- Step 5: verify -----------------------------------------------------
Write-Host "[5/5] Checking your GPU ..." -ForegroundColor Yellow
& $PYEXE -c "import torch; print('      torch', torch.__version__, '| HIP', torch.version.hip); print('      GPU:', torch.cuda.get_device_name(0))"

Write-Host ""
Write-Host "  All done! To chat with a model:" -ForegroundColor Green
Write-Host "    powershell -File dist\run-server.ps1 -Model <path-to-your-model>" -ForegroundColor White
Write-Host "  then open http://localhost:1420 in your browser.`n" -ForegroundColor Green
