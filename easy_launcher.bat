@echo off
title Resume Analyzer Launcher

:menu
cls
echo.
echo ==========================================
echo    Resume Analyzer - Easy Launcher
echo ==========================================
echo.
echo Current Status:
netstat -an | findstr :5001 > nul
if %errorlevel% == 0 (
    echo ✅ Application is RUNNING
    echo 🌐 Available at: http://localhost:5001
) else (
    echo ❌ Application is NOT running
)
echo.
echo Choose an option:
echo.
echo [1] 🚀 Quick Start (Recommended)
echo [2] 📊 Open Application (if running)
echo [3] ⏹️  Stop Application
echo [4] 🔍 Check Detailed Status
echo [5] 🛠️  Run Diagnostics
echo [6] 📁 Open Application Folder
echo [7] 📖 View Launch Guide
echo [8] ❌ Exit
echo.
set /p choice=Enter your choice (1-8): 

if "%choice%"=="1" goto quickstart
if "%choice%"=="2" goto openbrowser
if "%choice%"=="3" goto stopapp
if "%choice%"=="4" goto status
if "%choice%"=="5" goto diagnostics
if "%choice%"=="6" goto openfolder
if "%choice%"=="7" goto guide
if "%choice%"=="8" goto exit

echo Invalid choice. Please try again.
pause
goto menu

:quickstart
echo.
echo Starting Resume Analyzer...
start /min cmd /c "cd /d "C:\resume Analyzer" && "C:\resume Analyzer\nlp_resume_env\Scripts\python.exe" app.py"
echo Waiting for application to start...
timeout /t 5 /nobreak > nul
echo Opening browser...
start http://localhost:5001
echo.
echo ✅ Resume Analyzer is now running!
echo Press any key to return to menu...
pause > nul
goto menu

:openbrowser
netstat -an | findstr :5001 > nul
if %errorlevel% == 0 (
    echo Opening browser...
    start http://localhost:5001
) else (
    echo ❌ Application is not running. Please start it first.
)
echo.
pause
goto menu

:stopapp
echo.
echo Stopping Resume Analyzer...
taskkill /f /im python.exe /fi "COMMANDLINE eq *app.py*" > nul 2>&1
echo ✅ Application stopped
echo.
pause
goto menu

:status
echo.
echo ==========================================
echo           Detailed Status Check
echo ==========================================
echo.
netstat -an | findstr :5001 > nul
if %errorlevel% == 0 (
    echo ✅ Application Status: RUNNING
    echo 🌐 Local URL: http://localhost:5001
    echo 🌐 Network URL: http://127.0.0.1:5001
    echo.
    echo Active Python processes:
    tasklist | findstr python.exe
) else (
    echo ❌ Application Status: NOT RUNNING
    echo 🔧 Port 5001 is available
)
echo.
pause
goto menu

:diagnostics
echo.
echo Running diagnostics...
"C:\resume Analyzer\nlp_resume_env\Scripts\python.exe" "C:\resume Analyzer\diagnose_nlp.py"
pause
goto menu

:openfolder
echo.
echo Opening application folder...
explorer "C:\resume Analyzer"
goto menu

:guide
echo.
echo Opening launch guide...
start notepad "C:\resume Analyzer\LAUNCH_GUIDE.md"
goto menu

:exit
echo.
echo Thank you for using Resume Analyzer!
echo.
exit
