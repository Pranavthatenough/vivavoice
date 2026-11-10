# Starts VivaVoice and opens it in your browser. From the project folder:
#     .\scripts\start.ps1
# Press Ctrl+C in this window to stop it.

Set-Location (Split-Path $PSScriptRoot -Parent)

if (-not (Test-Path "venv\Scripts\python.exe")) {
    Write-Host "VivaVoice isn't set up yet. Run .\scripts\setup.ps1 first." -ForegroundColor Red
    exit 1
}

# Make sure Ollama is running.
try {
    Invoke-RestMethod "http://127.0.0.1:11434/api/version" -TimeoutSec 3 | Out-Null
} catch {
    Write-Host "Starting Ollama..."
    Start-Process ollama -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 4
}

# Open the browser as soon as the server answers (in the background, so the server can start).
Start-Job -ScriptBlock {
    for ($i = 0; $i -lt 120; $i++) {
        try {
            Invoke-RestMethod "http://127.0.0.1:8000/api/voice" -TimeoutSec 2 | Out-Null
            Start-Process "http://127.0.0.1:8000"
            return
        } catch { Start-Sleep -Seconds 1 }
    }
} | Out-Null

Write-Host "Starting VivaVoice. The browser opens when it's ready; press Ctrl+C here to stop." -ForegroundColor Cyan
& "venv\Scripts\python.exe" backend\app.py