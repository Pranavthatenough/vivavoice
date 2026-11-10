# VivaVoice setup for Windows. Run once, from the project folder:
#     .\scripts\setup.ps1
# It checks your tools, creates the Python environment, installs the libraries,
# downloads the AI models and loads the sample notes. Safe to run again.

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

function Step($text) { Write-Host "`n$text" -ForegroundColor Cyan }
function Fail($text) { Write-Host "`n$text" -ForegroundColor Red; exit 1 }

Step "1/6  Checking Python and Ollama"
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Fail "Python isn't installed or isn't on PATH. Install Python 3.11 or newer from python.org (tick 'Add Python to PATH')."
}
$ver = (& python -c "import sys; print('%d.%d' % sys.version_info[:2])")
if ([version]$ver -lt [version]"3.11") { Fail "Python $ver found; VivaVoice needs 3.11 or newer." }
Write-Host "Python $ver"
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    Fail "Ollama isn't installed. Get it from https://ollama.com, then run this script again."
}

Step "2/6  Checking Windows' memory settings"
# If the page file is off, Windows can refuse memory even when Task Manager shows plenty free.
# This caused most of the "failed to allocate" errors while building VivaVoice.
$os = Get-CimInstance Win32_OperatingSystem
$ramGB = [math]::Round($os.TotalVisibleMemorySize / 1MB, 1)
$commitGB = [math]::Round($os.TotalVirtualMemorySize / 1MB, 1)
if ($commitGB - $ramGB -lt 2) {
    Write-Host "Warning: your page file looks off (memory limit $commitGB GB with $ramGB GB of RAM)." -ForegroundColor Yellow
    Write-Host "Models may fail to load. Fix: View advanced system settings > Performance Settings > Advanced >" -ForegroundColor Yellow
    Write-Host "Virtual memory > Change > tick 'Automatically manage paging file size', then restart." -ForegroundColor Yellow
} else {
    Write-Host "OK: memory limit $commitGB GB with $ramGB GB of RAM."
}

Step "3/6  Creating the Python environment"
if (-not (Test-Path "venv\Scripts\python.exe")) {
    & python -m venv venv
    if ($LASTEXITCODE -ne 0) { Fail "Couldn't create the virtual environment." }
}
$py = (Resolve-Path "venv\Scripts\python.exe").Path

Step "4/6  Installing the libraries (a few minutes the first time)"
& $py -m pip install --upgrade pip --quiet
& $py -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { Fail "Installing the libraries failed. Scroll up for the error." }

Step "5/6  Downloading the AI models (about 2.5 GB the first time)"
try {
    Invoke-RestMethod "http://127.0.0.1:11434/api/version" -TimeoutSec 3 | Out-Null
} catch {
    Write-Host "Starting Ollama..."
    Start-Process ollama -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 4
}
foreach ($model in @("llama3.2:3b", "nomic-embed-text")) {
    & ollama pull $model
    if ($LASTEXITCODE -ne 0) { Fail "Couldn't download $model. Check your internet connection and try again." }
}

Step "6/6  Loading the sample notes"
Get-ChildItem "sample_notes" -Filter *.pdf | ForEach-Object {
    & $py backend\rag.py ingest $_.FullName
    if ($LASTEXITCODE -ne 0) { Write-Host "Couldn't load $($_.Name); you can upload it later in the app." -ForegroundColor Yellow }
}

Write-Host "`nSetup finished. Start VivaVoice with:  .\scripts\start.ps1" -ForegroundColor Green
Write-Host "The first viva also downloads the speech model (about 480 MB)."