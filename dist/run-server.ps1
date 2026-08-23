# ============================================================
#  FreeToken server launcher (Windows + AMD)
#
#  Usage:
#    powershell -File dist\run-server.ps1 -Model <path-to-model>
#
#  Optional switches:
#    -Arch gfx1201          your GPU family
#    -RocmPath <folder>     where the AMD ROCm runtime lives
#                           (not needed if HIP_PATH is already set)
#    -Port 1919             API port
#    -KVPages 4096          cap KV cache (needed for big dense models)
#    -ExtraArgs "--flag"    anything else for `ft serve`
#
#  When it says READY, open http://localhost:1420 and chat.
# ============================================================
param(
    [Parameter(Mandatory = $true)][string]$Model,
    [string]$Arch = "gfx1201",
    [string]$RocmPath = "",
    [int]$Port = 1919,
    [int]$KVPages = 0,
    [string[]]$ExtraArgs = @()
)
$ErrorActionPreference = "Stop"
$REPO = Split-Path -Parent $PSScriptRoot
$LogDir = "$env:TEMP\freetoken-logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

if (-not $RocmPath) {
    if ($env:HIP_PATH) { $RocmPath = $env:HIP_PATH }
    else { throw "Where is the AMD ROCm runtime? Pass -RocmPath or set HIP_PATH once:`n       [Environment]::SetEnvironmentVariable('HIP_PATH','C:\\ROCm\\...','User')" }
}
if ($KVPages -gt 0) { $ExtraArgs += "--num-pages $KVPages" }

# engine binary: prefer the repo venv this installer created, fall back to PATH
$ft = Join-Path $REPO ".venv\Scripts\ft.exe"
if (-not (Test-Path $ft)) { $ft = "ft" }

$vcvars = Get-ChildItem "C:\Program Files\Microsoft Visual Studio" -Recurse -Filter vcvarsall.bat -ErrorAction SilentlyContinue |
          Select-Object -First 1 -ExpandProperty FullName

$cmd = @"
$(if ($vcvars) { "call `"$vcvars`" x64 >nul" })
set HIP_PATH=$RocmPath
set TVM_FFI_ROCM_ARCH_LIST=$Arch
set TRITON_OVERRIDE_ARCH=$Arch
set ROCM_SDK_TARGET_FAMILY=$Arch
set "CC=$RocmPath\lib\llvm\bin\clang.EXE"
cd /d %TEMP%
"$ft" serve --model "$Model" --server-port $Port $($ExtraArgs -join ' ') > "$LogDir\serve.log" 2> "$LogDir\serve_err.log"
"@
$runner = Join-Path $env:TEMP "freetoken_serve.cmd"
Set-Content $runner $cmd -Encoding ASCII

Write-Host "Starting '$(Split-Path $Model -Leaf)' on port $Port ..." -ForegroundColor Cyan
Start-Process cmd.exe -ArgumentList "/c", $runner -WindowStyle Hidden

for ($i = 1; $i -le 120; $i++) {
    Start-Sleep 5
    if (Select-String -Path "$LogDir\serve.log" -Pattern "ready to serve" -ErrorAction SilentlyContinue) {
        # web UI next to the API
        Start-Process cmd.exe -ArgumentList "/c", "cd /d `"$REPO\webui`" && python -m http.server 1420" -WindowStyle Hidden
        Write-Host ""
        Write-Host "  READY! Open http://localhost:1420 in your browser." -ForegroundColor Green
        Write-Host "  Logs: $LogDir\serve.log / serve_err.log"
        exit 0
    }
    if (Select-String -Path "$LogDir\serve.log","$LogDir\serve_err.log" -Pattern "AssertionError|Traceback|exited during load" -ErrorAction SilentlyContinue) {
        Write-Host "`n  The server hit an error while loading. Last lines:" -ForegroundColor Red
        Get-Content "$LogDir\serve_err.log","$LogDir\serve.log" -Tail 6 -ErrorAction SilentlyContinue
        exit 1
    }
    Write-Host "." -NoNewline
}
Write-Host "`n  Timed out after 10 min - see $LogDir\serve_err.log" -ForegroundColor Red
