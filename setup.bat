@echo off
echo Setting up Resume Analyzer (First-time installation)...

echo Creating Python virtual environment with Python 3.11...
py -3.11 -m venv venv311
call venv311\Scripts\activate

echo Installing dependencies...
pip install flask==2.0.1
pip install werkzeug==2.0.1
pip install PyPDF2==2.10.5
pip install flask-wtf==0.15.1
pip install python-dotenv==0.19.0
pip install reportlab==3.6.12

echo.
echo Setup complete! You can now run the application using 'run.bat'
echo.
pause 