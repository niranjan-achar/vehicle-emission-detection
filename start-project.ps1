$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$backendRoot = Join-Path $projectRoot 'backend'
$frontendRoot = Join-Path $projectRoot 'frontend'
$backendVenvPython = Join-Path $backendRoot '.venv\Scripts\python.exe'

function Get-PythonCommand {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        return 'py -3'
    }

    if (Get-Command python -ErrorAction SilentlyContinue) {
        return 'python'
    }

    throw 'Python was not found. Install Python 3.10+ and re-run this script.'
}

Write-Host 'Preparing project startup...' -ForegroundColor Cyan

if (-not (Test-Path $backendVenvPython)) {
    Write-Host 'Creating backend virtual environment...' -ForegroundColor Yellow
    $pythonCmd = Get-PythonCommand
    Push-Location $backendRoot
    try {
        Invoke-Expression "$pythonCmd -m venv .venv"
    }
    finally {
        Pop-Location
    }
}

if (-not (Test-Path (Join-Path $backendRoot '.env'))) {
    Copy-Item (Join-Path $backendRoot '.env.example') (Join-Path $backendRoot '.env')
    Write-Host 'Created backend .env from .env.example' -ForegroundColor Yellow
}

if (-not (Test-Path (Join-Path $frontendRoot '.env'))) {
    Copy-Item (Join-Path $frontendRoot '.env.example') (Join-Path $frontendRoot '.env')
    Write-Host 'Created frontend .env from .env.example' -ForegroundColor Yellow
}

Write-Host 'Installing backend dependencies (if needed)...' -ForegroundColor Cyan
& $backendVenvPython -m pip install -r (Join-Path $backendRoot 'requirements.txt') | Out-Host

if (-not (Test-Path (Join-Path $frontendRoot 'node_modules'))) {
    Write-Host 'Installing frontend dependencies...' -ForegroundColor Cyan
    Push-Location $frontendRoot
    try {
        npm install | Out-Host
    }
    finally {
        Pop-Location
    }
}

$backendCommand = ".\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000"
$frontendCommand = "npm run dev -- --host 0.0.0.0 --port 5173"

Write-Host 'Starting backend in a new terminal...' -ForegroundColor Green
Start-Process powershell -WorkingDirectory $backendRoot -ArgumentList '-NoExit', '-Command', $backendCommand

Write-Host 'Starting frontend in a new terminal...' -ForegroundColor Green
Start-Process powershell -WorkingDirectory $frontendRoot -ArgumentList '-NoExit', '-Command', $frontendCommand

Write-Host ''
Write-Host 'Project started:' -ForegroundColor Green
Write-Host 'Backend:  http://localhost:8000' -ForegroundColor White
Write-Host 'Frontend: http://localhost:5173' -ForegroundColor White
Write-Host 'API Docs: http://localhost:8000/docs' -ForegroundColor White
