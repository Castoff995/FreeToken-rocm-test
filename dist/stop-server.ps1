# Stop any running FreeToken engine / web UI.
Get-CimInstance Win32_Process -Filter "Name='ft.exe' or Name like '%python%'" |
    Where-Object { $_.CommandLine -match "serve|http.server 1420|freetoken" } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force; "stopped $($_.ProcessId) $($_.Name)" }
