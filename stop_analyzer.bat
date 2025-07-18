@echo off
echo ==========================================
echo    Resume Analyzer - Stop Application
echo ==========================================
echo.

echo Stopping Resume Analyzer...

taskkill /f /im python.exe /fi "WINDOWTITLE eq Resume Analyzer" > nul 2>&1
taskkill /f /im python.exe /fi "COMMANDLINE eq *app.py*" > nul 2>&1

echo.
echo ✅ Resume Analyzer has been stopped
echo.
pause
