# Offline installer that runs INSIDE the portable bundle folder.
#   powershell -ExecutionPolicy Bypass -File install.ps1
# Installs pip into the bundled embeddable Python, then all wheels, then FreeToken.
# Requires: VS Build Tools (vcvarsall) and an AMD GPU + driver.

$ErrorActionPreference = "Stop"
$PY = "$PSScriptRoot\python\python.exe"
$ROCM_INDEX = "https://rocm.nightlies.amd.com/whl-multi-arch/"

Write-Host "== FreeToken portable bundle installer ==" -ForegroundColor Cyan

# 0. ROCm runtime check (the one thing we cannot bundle: driver + HIP runtime)
if (-not $env:HIP_PATH) {
    Write-Warning @"
HIP_PATH is not set. This bundle needs the AMD ROCm runtime (TheRock build) on disk.
Download/extract it, then either set HIP_PATH machine-wide or pass -RocmPath later to run-server.ps1.
"@
}

# 1. Bootstrap pip into the embeddable python
& $PY "$PSScriptRoot\python\get-pip.py" --no-warn-script-location

# 2. Plain-PyPI wheels first (build tools for the AMD stack resolution)
& $PY -m pip install --no-index --find-links "$PSScriptRoot\pypi-wheels" setuptools wheel ninja

# 3. AMD stack + torch from bundled wheels (--no-deps keeps nightly stamps consistent)
& $PY -m pip install (Get-ChildItem "$PSScriptRoot\rocm-wheels" -Recurse -Filter *.whl | ForEach-Object { $_.FullName }) --no-deps

# 4. Runtime deps from bundled PyPI wheels
& $PY -m pip install --no-index --find-links "$PSScriptRoot\pypi-wheels" `
    "triton-windows>=3.7.1" apache-tvm-ffi==0.1.13.post3 msgpack pyzmq psutil requests aiohttp `
    partial_json_parser gguf

# 5. FreeToken source (bundled) without CUDA extensions
$env:FREETOKEN_SKIP_CUDA_EXT = "1"
& $PY -m pip install --no-deps --no-build-isolation -e "$PSScriptRoot\freetoken"
Remove-Item Env:FREETOKEN_SKIP_CUDA_EXT

# 6. Upstream patches (tvm-ffi / triton / uvicorn)
& $PY "$PSScriptRoot\patch_upstream.py"

# 7. Smoke test
& $PY -c "import torch; print('torch:', torch.__version__, '| hip:', torch.version.hip); print('gpu:', torch.cuda.get_device_name(0))"

Write-Host @"

Portable install complete. Start serving:
    powershell -File run-server.ps1 -Model <path-to-model>
Then open http://localhost:1420
"@ -ForegroundColor Green
