@echo off
echo ==========================================
echo    Resume Analyzer - Easy Launcher
echo ==========================================
echo.
echo Choose an option:
echo.
echo [1] Quick Start (Start app + Open browser)
echo [2] Start Application Only
echo [3] Check Status
echo [4] Stop Application
echo [5] Run Diagnostics
echo [6] Exit
echo.
set /p choice=Enter your choice (1-6): 

if "%choice%"=="1" (
    start "Quick Start" "C:\resume Analyzer\quick_start.bat"
) else if "%choice%"=="2" (
    start "Resume Analyzer" "C:\resume Analyzer\start_analyzer.bat"
) else if "%choice%"=="3" (
    call "C:\resume Analyzer\check_status.bat"
) else if "%choice%"=="4" (
    call "C:\resume Analyzer\stop_analyzer.bat"
) else if "%choice%"=="5" (
    "C:\resume Analyzer\nlp_resume_env\Scripts\python.exe" "C:\resume Analyzer\diagnose_nlp.py"
) else if "%choice%"=="6" (
    exit
) else (
    echo Invalid choice. Please try again.
    pause
    goto :eof
)

echo.
echo Press any key to exit...
pause > nul
