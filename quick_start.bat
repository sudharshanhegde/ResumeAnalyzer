@echo off
echo ==========================================
echo    Resume Analyzer - Quick Start
echo ==========================================
echo.
echo This will:
echo 1. Start the Resume Analyzer application
echo 2. Open your web browser automatically
echo 3. Take you directly to the application
echo.
echo Press any key to continue...
pause > nul

cd /d "C:\resume Analyzer"

echo Starting application...
start /b "Resume Analyzer" "C:\resume Analyzer\nlp_resume_env\Scripts\python.exe" app.py

echo Waiting for application to start...
timeout /t 3 /nobreak > nul

echo Opening web browser...
start http://localhost:5001

echo.
echo ==========================================
echo   Resume Analyzer is now running!
echo   
echo   Application URL: http://localhost:5001
echo   
echo   To stop the application, close this window
echo   or press Ctrl+C in the application window
echo ==========================================
echo.
echo Press any key to exit this launcher...
pause > nul
