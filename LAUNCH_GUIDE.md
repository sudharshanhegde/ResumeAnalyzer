# Resume Analyzer - Easy Launch Guide

## 🚀 Quick Start Options

### Option 1: Desktop Shortcut (Recommended)

1. Double-click `create_shortcut.ps1` to create a desktop shortcut
2. Double-click the "Resume Analyzer" shortcut on your desktop
3. The application will start automatically and open in your browser

### Option 2: Simple Batch Files

- **`quick_start.bat`** - Starts the app and opens your browser automatically
- **`launcher.bat`** - Interactive menu with all options
- **`start_analyzer.bat`** - Just starts the application
- **`check_status.bat`** - Check if the app is running
- **`stop_analyzer.bat`** - Stop the application

### Option 3: PowerShell Script (Advanced)

- **`launcher.ps1`** - Full-featured PowerShell launcher with menu

## 📋 How to Use

### Easiest Method:

1. Double-click `quick_start.bat`
2. Wait for your browser to open
3. Start analyzing resumes!

### Interactive Method:

1. Double-click `launcher.bat`
2. Choose from the menu options:
   - [1] Quick Start (Recommended)
   - [2] Start Application Only
   - [3] Check Status
   - [4] Stop Application
   - [5] Run Diagnostics
   - [6] Exit

## 🌐 Application URLs

- **Local Access**: http://localhost:5001
- **Network Access**: http://127.0.0.1:5001

## 🔧 Troubleshooting

### If the application doesn't start:

1. Run `diagnose_nlp.py` to check your setup
2. Make sure Python and all dependencies are installed
3. Check that port 5001 is available

### If the browser doesn't open:

1. Manually navigate to http://localhost:5001
2. Check if the application is running with `check_status.bat`

## 📝 Features Available

- **AI-Powered Analysis**: Advanced NLP techniques
- **Skill Detection**: Automatic skill categorization
- **Job Matching**: Semantic similarity analysis
- **Sentiment Analysis**: Resume tone evaluation
- **Improvement Suggestions**: Personalized recommendations

## 🛠️ Technical Details

The application uses:

- **Flask** web framework
- **spaCy** for NLP processing
- **Transformers** for sentiment analysis
- **Sentence Transformers** for semantic similarity
- **NLTK** for text processing

## 📞 Support

If you encounter any issues:

1. Run the diagnostics: `diagnose_nlp.py`
2. Check the application logs
3. Restart the application using the stop/start scripts

---

**Created by**: Resume Analyzer AI System
**Version**: 2.0 (Advanced NLP Edition)
**Date**: July 2025
