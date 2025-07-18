@echo off
echo ==========================================
echo    Resume Analyzer - Status Check
echo ==========================================
echo.

cd /d "C:\resume Analyzer"

echo Checking if application is running...
netstat -an | findstr :5001 > nul
if %errorlevel% == 0 (
    echo ✅ Application is RUNNING on port 5001
    echo.
    echo Opening browser...
    start http://localhost:5001
) else (
    echo ❌ Application is NOT running
    echo.
    echo Would you like to start it now? (Y/N)
    set /p choice=
    if /i "%choice%"=="y" (
        echo Starting application...
        start "Resume Analyzer" "C:\resume Analyzer\quick_start.bat"
    )
)

echo.
pause
