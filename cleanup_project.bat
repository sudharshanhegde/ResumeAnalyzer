@echo off
echo ==========================================
echo    Resume Analyzer - Project Cleanup
echo ==========================================
echo.
echo This script will clean up temporary files
echo and prepare the project for git commit.
echo.
echo What will be cleaned:
echo - Python cache files (__pycache__)
echo - Temporary files (*.tmp, *.log)
echo - Virtual environment (if requested)
echo - Uploaded files (if requested)
echo.
set /p continue=Continue? (Y/N): 
if /i not "%continue%"=="y" goto end

echo.
echo Cleaning Python cache files...
if exist "__pycache__" rmdir /s /q "__pycache__"
if exist "app\__pycache__" rmdir /s /q "app\__pycache__"

echo Cleaning temporary files...
del /q *.tmp *.log *.out 2>nul

echo Cleaning uploaded files...
set /p clean_uploads=Clean uploaded files? (Y/N): 
if /i "%clean_uploads%"=="y" (
    del /q "app\static\uploads\*.pdf" 2>nul
    del /q "app\static\uploads\*.doc" 2>nul
    del /q "app\static\uploads\*.docx" 2>nul
    echo Uploaded files cleaned
)

echo.
echo Checking virtual environment...
set /p clean_venv=Remove virtual environment? (Y/N): 
if /i "%clean_venv%"=="y" (
    echo Removing virtual environment...
    if exist "nlp_resume_env" rmdir /s /q "nlp_resume_env"
    if exist "ui_venv" rmdir /s /q "ui_venv"
    echo Virtual environment removed
    echo.
    echo IMPORTANT: Run setup_clean_nlp.bat to reinstall before using the app
)

echo.
echo ==========================================
echo    Cleanup Complete!
echo ==========================================
echo.
echo Current project structure:
dir /b | findstr /v "nlp_resume_env"
echo.
echo Ready for git commit!
echo.

:end
pause
