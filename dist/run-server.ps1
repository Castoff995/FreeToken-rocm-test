# Start the FreeToken engine + web UI on Windows/ROCm.
# Usage: powershell -File dist\run-server.ps1 -Model G:\path\to\model [-Arch gfx1201] [-RocmPath G:\ROCM10RT-gfx1201]
param(
    [Parameter(Mandatory = $true)][string]$Model,
    [string]$Arch = "gfx1201",
    [string]$RocmPath = "",
    [int]$Port = 1919,
    [string[]]$ExtraArgs = @()
)
$ErrorActionPreference = "Stop"
$REPO = Split-Path -Parent $PSScriptRoot
$LogDir = "$env:TEMP\opencode"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

if (-not $RocmPath) {
    if ($env:HIP_PATH) { $RocmPath = $env:HIP_PATH }
    else { throw "Set -RocmPath or the HIP_PATH environment variable to your ROCm runtime root." }
}

# Find vcvarsall (MSVC CRT for JIT DLL linking) - optional but recommended
$vcvars = Get-ChildItem "C:\Program Files\Microsoft Visual Studio" -Recurse -Filter vcvarsall.bat -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName

$modelName = Split-Path $Model -Leaf
$cmd = @"
$(if ($vcvars) { "call `"$vcvars`" x64 >nul" })
set HIP_PATH=$RocmPath
set TVM_FFI_ROCM_ARCH_LIST=$Arch
set ROCM_SDK_TARGET_FAMILY=$Arch
set TRITON_OVERRIDE_ARCH=$Arch
set "CC=$RocmPath\lib\llvm\bin\clang.EXE"
cd /d %TEMP%
ft serve --model "$Model" --server-port $Port $($ExtraArgs -join ' ') > "$LogDir\ft_serve.log" 2> "$LogDir\ft_err.log"
"@
$runner = Join-Path $env:TEMP "freetoken_serve.cmd"
Set-Content $runner $cmd -Encoding ASCII

Start-Process cmd.exe -ArgumentList "/c", $runner -WindowStyle Hidden
Write-Host "Server starting: model '$modelName' on port $Port" -ForegroundColor Cyan
Write-Host "Logs: $LogDir\ft_serve.log / ft_err.log"

# Wait for readiness
for ($i = 0; $i -lt 120; $i++) {
    Start-Sleep 5
    try {
        $r = Invoke-RestMethod "http://127.0.0.1:$Port/v1/models" -TimeoutSec 3
        Write-Host "`nAPI ready at http://127.0.0.1:$Port  (model: $($r.data[0].id))" -ForegroundColor Green
        # launch web UI
        $webui = Join-Path $REPO "webui"
        Start-Process cmd.exe -ArgumentList "/c", "cd /d `"$webui`" && python -m http.server 1420" -WindowStyle Hidden
        Write-Host "Web chat UI:  http://localhost:1420" -ForegroundColor Green
        exit 0
    } catch { Write-Host "  waiting... ($($i*5)s)" -NoNewline }
}
Write-Host "`nTimed out waiting for the API - check $LogDir\ft_err.log" -ForegroundColor Red
