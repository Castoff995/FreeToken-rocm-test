# Assemble a portable, self-contained distribution zip:
#   python embeddable + pip bootstrap + all wheels (AMD nightly + PyPI) + repo + scripts
# Usage:
#   powershell -File dist\make-bundle.ps1 [-Arch gfx1201] [-PythonVer 3.12.10]
# Output: dist\freetoken-rocm-port-<arch>.zip
# The end user only needs: this zip, VS Build Tools, and an AMD GPU+driver.
param(
    [string]$Arch = "gfx1201",
    [string]$PythonVer = "3.12.10",
    [string]$OutDir = "$PSScriptRoot"
)
$ErrorActionPreference = "Stop"
$REPO = Split-Path -Parent $PSScriptRoot
$INDEX = "https://rocm.nightlies.amd.com/whl-multi-arch/"
$stage = Join-Path $env:TEMP "freetoken-bundle"
if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }
New-Item -ItemType Directory -Force -Path $stage | Out-Null

Write-Host "== 1. Python $PythonVer embeddable =="
$pyzip = "$stage\python.zip"
Invoke-WebRequest "https://www.python.org/ftp/python/$PythonVer/python-$PythonVer-embed-amd64.zip" -OutFile $pyzip
Expand-Archive $pyzip -DestinationPath "$stage\python"
Remove-Item $pyzip
# enable site-packages + pip inside the embeddable
$pth = Get-ChildItem "$stage\python" -Filter "python*._pth" | Select-Object -First 1
Set-Content $pth.FullName "python312.zip`n.`nLib\\site-packages`nimport site`n"
Invoke-WebRequest "https://bootstrap.pypa.io/get-pip.py" -OutFile "$stage\python\get-pip.py"

Write-Host "== 2. Download wheels =="
$amdWheels = "$stage\rocm-wheels"
New-Item -ItemType Directory -Force -Path $amdWheels | Out-Null
# AMD stack: same nightly stamp guaranteed by the index (multi-arch extras OK)
pip download --index-url $INDEX -d $amdWheels "rocm[libraries,devel,device-$Arch]"
# torch ROCm + device module live in per-package folders on the index
pip download --index-url "$INDEX/torch/" -d $amdWheels\torch torch --no-deps
pip download --index-url "$INDEX/amd-torch-device-$Arch/" -d $amdWheels\torch "amd-torch-device-$Arch"
# plain-PyPI runtime deps
pip download -d "$stage\pypi-wheels" `
    "triton-windows>=3.7.1" apache-tvm-ffi==0.1.13.post3 msgpack pyzmq psutil requests aiohttp `
    partial_json_parser gguf setuptools wheel ninja

Write-Host "== 3. Repo snapshot (source, no git/VCS junk) =="
git -C $REPO archive --format=zip --output="$stage\repo.zip" HEAD
Expand-Archive "$stage\repo.zip" -DestinationPath "$stage\freetoken"

Write-Host "== 4. Offline installer entry point =="
Move-Item "$REPO\dist\install-offline.ps1" "$stage\install.ps1" -ErrorAction SilentlyContinue
Copy-Item "$REPO\dist\run-server.ps1","$REPO\dist\stop-server.ps1","$REPO\dist\patch_upstream.py" $stage

Write-Host "== 5. Zip it =="
$zip = Join-Path $OutDir "freetoken-rocm-port-$Arch.zip"
Compress-Archive -Path "$stage\*" -DestinationPath $zip -Force
Remove-Item $stage -Recurse -Force
"{0:N1} MB -> {1}" -f ((Get-Item $zip).Length / 1MB), $zip
