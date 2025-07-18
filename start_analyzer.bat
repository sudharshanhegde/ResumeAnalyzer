@echo off
echo ==========================================
echo    Resume Analyzer - AI-Powered Analysis
echo ==========================================
echo.
echo Starting the Resume Analyzer...
echo.

cd /d "C:\resume Analyzer"

echo Activating virtual environment...
call "C:\resume Analyzer\nlp_resume_env\Scripts\activate.bat"

echo Starting Flask application...
echo.
echo ==========================================
echo   Application will be available at:
echo   http://localhost:5001
echo ==========================================
echo.
echo Press Ctrl+C to stop the application
echo.

python app.py

pause
