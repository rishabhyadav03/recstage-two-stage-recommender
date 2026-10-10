# scripts/dev.ps1 - Start backend and frontend together on Windows
$ErrorActionPreference = "Continue"

$RootDir = Resolve-Path "$PSScriptRoot\.."
Set-Location $RootDir

# Locate Python / Uvicorn executable
$Uvicorn = "$RootDir\.venv\Scripts\uvicorn.exe"
if (-not (Test-Path $Uvicorn)) {
    $Uvicorn = "uvicorn"
}

Write-Host ""
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "  RecStage Development Servers" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "  Backend API:  http://127.0.0.1:8000" -ForegroundColor Green
Write-Host "  API Docs:     http://127.0.0.1:8000/docs" -ForegroundColor Green
Write-Host "  Frontend UI:  http://localhost:5173" -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "  Press Ctrl+C to stop both servers." -ForegroundColor Yellow
Write-Host ""

$backendProcess = Start-Process -FilePath $Uvicorn -ArgumentList "api.main:app --port 8000" -WorkingDirectory $RootDir -PassThru

try {
    Set-Location "$RootDir\frontend"
    npm run dev
}
finally {
    Set-Location $RootDir
    Write-Host "`nStopping servers..." -ForegroundColor Yellow
    if ($backendProcess -and -not $backendProcess.HasExited) {
        Stop-Process -Id $backendProcess.Id -Force -ErrorAction SilentlyContinue
    }
    Write-Host "Servers stopped." -ForegroundColor Green
}
