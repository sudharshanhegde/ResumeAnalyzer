@echo off
echo Resume Analyzer

IF NOT EXIST venv311 (
    echo Virtual environment not found!
    echo Please run setup.bat first to install dependencies.
    echo.
    pause
    exit
)

echo Activating virtual environment...
call venv311\Scripts\activate

echo.
echo Running the Resume Analyzer application...
python app.py

echo.
echo If the browser doesn't open automatically, go to: http://localhost:5001
pause 