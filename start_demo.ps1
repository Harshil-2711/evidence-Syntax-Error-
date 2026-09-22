```powershell
# start_demo.ps1
# Launches backend + opens the frontend for the hackathon demo.
#
# Usage: .\start_demo.ps1
#
# What it does:
#   1. Starts the FastAPI backend on http://localhost:8080
#   2. Starts a simple HTTP server for the frontend on http://localhost:5173
#   3. Opens the frontend in your default browser

Write-Host ""
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "  Evidence-Gated Self-Healing Agent - Demo Launcher" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""

$root = $PSScriptRoot

# Check Python
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "[ERROR] Python not found. Install Python 3.11+ and try again." -ForegroundColor Red
    exit 1
}

# Check requirements file
if (-not (Test-Path "$root\requirements.txt")) {
    Write-Host "[ERROR] requirements.txt not found in project root." -ForegroundColor Red
    exit 1
}

# Install dependencies if uvicorn is not available
if (-not (Get-Command uvicorn -ErrorAction SilentlyContinue)) {
    Write-Host "[INFO] Installing Python dependencies..." -ForegroundColor Yellow
    python -m pip install -r "$root\requirements.txt"
}

# Check API folder
if (-not (Test-Path "$root\api")) {
    Write-Host "[ERROR] api folder not found." -ForegroundColor Red
    exit 1
}

# Check frontend folder
if (-not (Test-Path "$root\frontend")) {
    Write-Host "[ERROR] frontend folder not found." -ForegroundColor Red
    exit 1
}

Write-Host "[1/3] Starting FastAPI backend on http://localhost:8080 ..." -ForegroundColor Green

Start-Process powershell -ArgumentList "-NoExit", "-Command", `
    "Set-Location '$root'; python -m uvicorn api.main:app --reload --port 8080" `
    -WindowStyle Normal

# Wait for backend to be ready
Write-Host "[2/3] Waiting for backend to start..." -ForegroundColor Yellow

$retries = 0
$ready = $false

do {
    Start-Sleep -Seconds 1
    $retries++

    try {
        resp = Invoke-WebRequest `
            -Uri "http://localhost:8080/health" `
            -UseBasicParsing `
            -TimeoutSec 2 `
            -ErrorAction Stop

        $ready = $true
    }
    catch {
        $ready = $false
    }

} while (-not $ready -and $retries -lt 15)

if (-not $ready) {
    Write-Host "[WARN] Backend did not respond in time. Check the backend window." -ForegroundColor Yellow
}
else {
    Write-Host "[OK] Backend is running." -ForegroundColor Green
}

Write-Host "[3/3] Starting frontend server on http://localhost:5173 ..." -ForegroundColor Green

Start-Process powershell -ArgumentList "-NoExit", "-Command", `
    "Set-Location '$root\frontend'; python -m http.server 5173" `
    -WindowStyle Normal

Start-Sleep -Seconds 2

# Open browser
Write-Host ""
Write-Host "Opening frontend in browser: http://localhost:5173" -ForegroundColor Cyan

Start-Process "http://localhost:5173"

Write-Host ""
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "  Backend:  http://localhost:8080" -ForegroundColor White
Write-Host "  Frontend: http://localhost:5173" -ForegroundColor White
Write-Host "  API docs: http://localhost:8080/docs" -ForegroundColor White
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C in the terminal windows to stop servers." -ForegroundColor Gray
Write-Host ""
```
