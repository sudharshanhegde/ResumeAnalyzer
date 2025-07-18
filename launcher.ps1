# Resume Analyzer - PowerShell Launcher
# This script provides an easy way to start and manage the Resume Analyzer

param(
    [string]$Action = "menu"
)

$AppPath = "C:\resume Analyzer"
$PythonPath = "$AppPath\nlp_resume_env\Scripts\python.exe"
$AppScript = "$AppPath\app.py"
$AppUrl = "http://localhost:5001"

function Show-Header {
    Write-Host "==========================================" -ForegroundColor Cyan
    Write-Host "   Resume Analyzer - AI-Powered Analysis" -ForegroundColor White
    Write-Host "==========================================" -ForegroundColor Cyan
    Write-Host ""
}

function Test-AppRunning {
    $processes = Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -like "*app.py*" }
    return $processes.Count -gt 0
}

function Start-App {
    Write-Host "Starting Resume Analyzer..." -ForegroundColor Green
    
    Set-Location $AppPath
    
    $processInfo = New-Object System.Diagnostics.ProcessStartInfo
    $processInfo.FileName = $PythonPath
    $processInfo.Arguments = "app.py"
    $processInfo.UseShellExecute = $false
    $processInfo.CreateNoWindow = $false
    $processInfo.WorkingDirectory = $AppPath
    
    $process = [System.Diagnostics.Process]::Start($processInfo)
    
    Write-Host "Waiting for application to start..." -ForegroundColor Yellow
    Start-Sleep -Seconds 3
    
    if (Test-AppRunning) {
        Write-Host "✅ Application started successfully!" -ForegroundColor Green
        Write-Host "📱 Available at: $AppUrl" -ForegroundColor Cyan
        return $true
    } else {
        Write-Host "❌ Failed to start application" -ForegroundColor Red
        return $false
    }
}

function Stop-App {
    Write-Host "Stopping Resume Analyzer..." -ForegroundColor Yellow
    
    $processes = Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -like "*app.py*" }
    
    if ($processes.Count -gt 0) {
        $processes | ForEach-Object { $_.Kill() }
        Write-Host "✅ Application stopped successfully!" -ForegroundColor Green
    } else {
        Write-Host "ℹ️  Application was not running" -ForegroundColor Blue
    }
}

function Open-Browser {
    Write-Host "Opening browser..." -ForegroundColor Green
    Start-Process $AppUrl
}

function Show-Status {
    if (Test-AppRunning) {
        Write-Host "✅ Resume Analyzer is RUNNING" -ForegroundColor Green
        Write-Host "📱 Available at: $AppUrl" -ForegroundColor Cyan
        
        $choice = Read-Host "Would you like to open the browser? (Y/N)"
        if ($choice -eq "Y" -or $choice -eq "y") {
            Open-Browser
        }
    } else {
        Write-Host "❌ Resume Analyzer is NOT running" -ForegroundColor Red
        
        $choice = Read-Host "Would you like to start it now? (Y/N)"
        if ($choice -eq "Y" -or $choice -eq "y") {
            if (Start-App) {
                Open-Browser
            }
        }
    }
}

function Show-Menu {
    Show-Header
    
    Write-Host "Choose an option:" -ForegroundColor White
    Write-Host ""
    Write-Host "[1] 🚀 Quick Start (Start + Open Browser)" -ForegroundColor Green
    Write-Host "[2] ▶️  Start Application Only" -ForegroundColor Blue
    Write-Host "[3] 📊 Check Status" -ForegroundColor Yellow
    Write-Host "[4] ⏹️  Stop Application" -ForegroundColor Red
    Write-Host "[5] 🔍 Run Diagnostics" -ForegroundColor Magenta
    Write-Host "[6] 🌐 Open Browser (if running)" -ForegroundColor Cyan
    Write-Host "[7] ❌ Exit" -ForegroundColor Gray
    Write-Host ""
    
    $choice = Read-Host "Enter your choice (1-7)"
    
    switch ($choice) {
        "1" { 
            if (Start-App) { 
                Open-Browser 
            }
        }
        "2" { Start-App }
        "3" { Show-Status }
        "4" { Stop-App }
        "5" { 
            & $PythonPath "$AppPath\diagnose_nlp.py"
        }
        "6" { 
            if (Test-AppRunning) { 
                Open-Browser 
            } else { 
                Write-Host "❌ Application is not running" -ForegroundColor Red 
            }
        }
        "7" { exit }
        default { 
            Write-Host "Invalid choice. Please try again." -ForegroundColor Red
            Start-Sleep -Seconds 2
            Show-Menu
        }
    }
}

# Main execution
switch ($Action.ToLower()) {
    "start" { Start-App }
    "stop" { Stop-App }
    "status" { Show-Status }
    "quick" { 
        if (Start-App) { 
            Open-Browser 
        }
    }
    "browser" { Open-Browser }
    default { Show-Menu }
}

Write-Host ""
Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
