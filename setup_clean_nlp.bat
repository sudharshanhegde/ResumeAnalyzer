@echo off
echo ====================================
echo  Advanced NLP Resume Analyzer Setup
echo ====================================
echo.

echo Step 1: Cleaning up old virtual environments...
echo.

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python from https://python.org
    pause
    exit /b 1
)

REM Stop any running Python processes (optional warning)
echo WARNING: Please close any running Python applications or VS Code instances
echo Press any key to continue...
pause >nul
echo.

REM Remove old virtual environments
if exist "venv" (
    echo Removing old 'venv' directory...
    rmdir /s /q "venv" 2>nul
    if exist "venv" (
        echo Warning: Could not remove 'venv'. Please close all applications and try again.
        pause
        exit /b 1
    )
)

if exist "venv311" (
    echo Removing old 'venv311' directory...
    rmdir /s /q "venv311" 2>nul
)

if exist "ui_venv" (
    echo Removing old 'ui_venv' directory...
    rmdir /s /q "ui_venv" 2>nul
)

if exist "direct_venv" (
    echo Removing old 'direct_venv' directory...
    rmdir /s /q "direct_venv" 2>nul
)

if exist "simple_venv" (
    echo Removing old 'simple_venv' directory...
    rmdir /s /q "simple_venv" 2>nul
)

if exist "nlp_venv" (
    echo Removing old 'nlp_venv' directory...
    rmdir /s /q "nlp_venv" 2>nul
)

echo Old virtual environments cleaned up!
echo.

echo Step 2: Creating new virtual environment...
echo.

REM Create new virtual environment
python -m venv nlp_resume_env
if %errorlevel% neq 0 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

echo Virtual environment 'nlp_resume_env' created successfully!
echo.

echo Step 3: Activating virtual environment...
echo.

REM Activate virtual environment
call nlp_resume_env\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)

echo Virtual environment activated!
echo.

echo Step 4: Upgrading pip...
echo.

python -m pip install --upgrade pip
if %errorlevel% neq 0 (
    echo ERROR: Failed to upgrade pip
    pause
    exit /b 1
)

echo Step 5: Installing basic dependencies...
echo.

REM Install basic requirements first
pip install flask==2.0.1
pip install werkzeug==2.0.1
pip install PyPDF2==2.10.5
pip install flask-wtf==0.15.1
pip install python-dotenv==0.19.0
pip install reportlab==3.6.12
pip install markdown==3.5.2

if %errorlevel% neq 0 (
    echo ERROR: Failed to install basic dependencies
    pause
    exit /b 1
)

echo Basic dependencies installed!
echo.

echo Step 6: Installing scientific computing libraries...
echo.

pip install numpy>=1.24.0
pip install scipy>=1.10.0
if %errorlevel% neq 0 (
    echo ERROR: Failed to install scientific libraries
    pause
    exit /b 1
)

echo Step 7: Installing machine learning libraries...
echo.

pip install scikit-learn>=1.3.0
if %errorlevel% neq 0 (
    echo ERROR: Failed to install scikit-learn
    pause
    exit /b 1
)

echo Step 8: Installing NLP libraries...
echo.

REM Install NLTK first
pip install nltk>=3.8.1
if %errorlevel% neq 0 (
    echo ERROR: Failed to install NLTK
    pause
    exit /b 1
)

REM Install spaCy
pip install spacy>=3.7.0
if %errorlevel% neq 0 (
    echo ERROR: Failed to install spaCy
    pause
    exit /b 1
)

REM Install text analysis libraries
pip install textstat>=0.7.3
if %errorlevel% neq 0 (
    echo ERROR: Failed to install textstat
    pause
    exit /b 1
)

echo Step 9: Installing PyTorch (CPU version)...
echo.

REM Install PyTorch CPU version (lighter weight)
pip install torch>=2.0.0 torchvision>=0.15.0 torchaudio>=2.0.0 --index-url https://download.pytorch.org/whl/cpu
if %errorlevel% neq 0 (
    echo ERROR: Failed to install PyTorch
    pause
    exit /b 1
)

echo Step 10: Installing Transformers and related libraries...
echo.

REM Install Transformers
pip install transformers>=4.35.0
if %errorlevel% neq 0 (
    echo ERROR: Failed to install Transformers
    pause
    exit /b 1
)

REM Install Sentence Transformers
pip install sentence-transformers>=2.2.2
if %errorlevel% neq 0 (
    echo ERROR: Failed to install Sentence Transformers
    pause
    exit /b 1
)

echo Step 11: Downloading spaCy language model...
echo.

python -m spacy download en_core_web_sm
if %errorlevel% neq 0 (
    echo ERROR: Failed to download spaCy English model
    echo You may need to run this manually later: python -m spacy download en_core_web_sm
    pause
)

echo Step 12: Downloading NLTK data...
echo.

python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True); nltk.download('averaged_perceptron_tagger', quiet=True); nltk.download('maxent_ne_chunker', quiet=True); nltk.download('words', quiet=True); print('NLTK data downloaded successfully!')"
if %errorlevel% neq 0 (
    echo ERROR: Failed to download NLTK data
    pause
)

echo Step 13: Installing additional useful packages...
echo.

pip install wordcloud>=1.9.2
pip install matplotlib>=3.6.0
pip install seaborn>=0.12.0

echo Step 14: Testing installation...
echo.

python -c "import spacy, nltk, sklearn, transformers, sentence_transformers, textstat, numpy; print('All core libraries imported successfully!')"
if %errorlevel% neq 0 (
    echo ERROR: Some libraries failed to import
    pause
    exit /b 1
)

echo.
echo ====================================
echo   INSTALLATION COMPLETED SUCCESSFULLY!
echo ====================================
echo.
echo Virtual environment: nlp_resume_env
echo.
echo Next steps:
echo 1. Test the installation: python test_nlp_setup.py
echo 2. Run the application: python app.py
echo.
echo To activate this environment in the future:
echo   nlp_resume_env\Scripts\activate.bat
echo.
echo Press any key to exit...
pause >nul
