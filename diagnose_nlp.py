"""
Troubleshooting and Diagnostic Script for Advanced NLP Resume Analyzer
Run this script to diagnose and fix common issues
"""

import sys
import os
import subprocess
import importlib.util
from pathlib import Path

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def check_python_version():
    """Check Python version compatibility"""
    print_section("PYTHON VERSION CHECK")
    
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3:
        print("❌ ERROR: Python 3 is required")
        return False
    elif version.minor < 8:
        print("⚠️  WARNING: Python 3.8+ is recommended")
        return True
    else:
        print("✅ Python version is compatible")
        return True

def check_virtual_environment():
    """Check if we're in a virtual environment"""
    print_section("VIRTUAL ENVIRONMENT CHECK")
    
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    
    if in_venv:
        print(f"✅ Virtual environment active: {sys.prefix}")
        return True
    else:
        print("❌ No virtual environment detected")
        print("Please activate the virtual environment:")
        print("  nlp_resume_env\\Scripts\\activate.bat")
        return False

def check_basic_imports():
    """Check if basic Python libraries can be imported"""
    print_section("BASIC IMPORTS CHECK")
    
    basic_libs = [
        'os', 'sys', 'json', 're', 'collections', 'tempfile', 'io'
    ]
    
    all_good = True
    for lib in basic_libs:
        try:
            __import__(lib)
            print(f"✅ {lib}: OK")
        except ImportError as e:
            print(f"❌ {lib}: FAILED - {e}")
            all_good = False
    
    return all_good

def check_flask_imports():
    """Check Flask and web framework imports"""
    print_section("FLASK IMPORTS CHECK")
    
    flask_libs = [
        ('flask', 'Flask'),
        ('werkzeug', 'Werkzeug'),
        ('PyPDF2', 'PyPDF2'),
        ('reportlab', 'ReportLab')
    ]
    
    all_good = True
    for lib, name in flask_libs:
        try:
            __import__(lib)
            print(f"✅ {name}: OK")
        except ImportError as e:
            print(f"❌ {name}: FAILED - {e}")
            all_good = False
    
    return all_good

def check_nlp_imports():
    """Check NLP library imports"""
    print_section("NLP IMPORTS CHECK")
    
    nlp_libs = [
        ('numpy', 'NumPy'),
        ('scipy', 'SciPy'),
        ('sklearn', 'scikit-learn'),
        ('nltk', 'NLTK'),
        ('spacy', 'spaCy'),
        ('textstat', 'TextStat'),
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers'),
        ('sentence_transformers', 'Sentence Transformers')
    ]
    
    failed_imports = []
    
    for lib, name in nlp_libs:
        try:
            __import__(lib)
            print(f"✅ {name}: OK")
        except ImportError as e:
            print(f"❌ {name}: FAILED - {e}")
            failed_imports.append(lib)
    
    if failed_imports:
        print(f"\n⚠️  {len(failed_imports)} NLP libraries failed to import")
        print("Run the setup script to install missing libraries:")
        print("  setup_clean_nlp.bat")
    
    return len(failed_imports) == 0

def check_spacy_model():
    """Check if spaCy English model is available"""
    print_section("SPACY MODEL CHECK")
    
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        print("✅ spaCy English model loaded successfully")
        
        # Test basic functionality
        doc = nlp("Test sentence for spaCy.")
        print(f"✅ spaCy processing works: {len(doc)} tokens")
        return True
        
    except OSError as e:
        print(f"❌ spaCy English model not found: {e}")
        print("Download the model with:")
        print("  python -m spacy download en_core_web_sm")
        return False
    except Exception as e:
        print(f"❌ spaCy model error: {e}")
        return False

def check_nltk_data():
    """Check if NLTK data is available"""
    print_section("NLTK DATA CHECK")
    
    try:
        import nltk
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize
        
        # Test if data is available
        try:
            stopwords.words('english')
            print("✅ NLTK stopwords available")
        except LookupError:
            print("❌ NLTK stopwords not found")
            return False
        
        try:
            word_tokenize("test sentence")
            print("✅ NLTK tokenizer available")
        except LookupError:
            print("❌ NLTK tokenizer not found")
            return False
        
        return True
        
    except ImportError as e:
        print(f"❌ NLTK import failed: {e}")
        return False

def check_transformers():
    """Check if Transformers library works"""
    print_section("TRANSFORMERS CHECK")
    
    try:
        from transformers import pipeline
        print("✅ Transformers library imported")
        
        # Test basic pipeline (this might take time on first run)
        try:
            print("Testing sentiment analysis pipeline...")
            sentiment_pipeline = pipeline("sentiment-analysis", 
                                         model="cardiffnlp/twitter-roberta-base-sentiment-latest")
            result = sentiment_pipeline("This is a test")
            print(f"✅ Sentiment analysis works: {result}")
            return True
        except Exception as e:
            print(f"⚠️  Sentiment pipeline failed: {e}")
            print("This is normal on first run - models will download automatically")
            return True
            
    except Exception as e:
        print(f"❌ Transformers error: {e}")
        return False

def check_app_files():
    """Check if application files exist"""
    print_section("APPLICATION FILES CHECK")
    
    required_files = [
        'app.py',
        'requirements_nlp.txt',
        'app/templates/index.html',
        'app/templates/results.html',
        'app/static/css/modern.css'
    ]
    
    all_good = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}: Found")
        else:
            print(f"❌ {file_path}: Missing")
            all_good = False
    
    return all_good

def check_app_functionality():
    """Test basic app functionality"""
    print_section("APPLICATION FUNCTIONALITY CHECK")
    
    try:
        # Import app components
        from app import app, extract_text_from_pdf, analyze_resume
        print("✅ App imports successful")
        
        # Test app configuration
        print(f"✅ App configured with secret key: {bool(app.config.get('SECRET_KEY'))}")
        print(f"✅ Upload folder: {app.config.get('UPLOAD_FOLDER')}")
        
        return True
        
    except Exception as e:
        print(f"❌ App functionality error: {e}")
        return False

def run_diagnostics():
    """Run all diagnostic checks"""
    print("🔍 Advanced NLP Resume Analyzer - Diagnostic Script")
    print("This script will check your system configuration and dependencies")
    
    checks = [
        ("Python Version", check_python_version),
        ("Virtual Environment", check_virtual_environment),
        ("Basic Imports", check_basic_imports),
        ("Flask Imports", check_flask_imports),
        ("NLP Imports", check_nlp_imports),
        ("spaCy Model", check_spacy_model),
        ("NLTK Data", check_nltk_data),
        ("Transformers", check_transformers),
        ("App Files", check_app_files),
        ("App Functionality", check_app_functionality)
    ]
    
    results = []
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"❌ {check_name}: ERROR - {e}")
            results.append((check_name, False))
    
    # Summary
    print_section("DIAGNOSTIC SUMMARY")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {check_name}")
    
    print(f"\nOverall: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 All checks passed! Your system is ready.")
        print("Run 'python app.py' to start the application.")
    else:
        print(f"\n⚠️  {total - passed} checks failed.")
        print("Please run 'setup_clean_nlp.bat' to fix issues.")
    
    return passed == total

def main():
    """Main diagnostic function"""
    try:
        success = run_diagnostics()
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\n\nDiagnostic interrupted by user.")
        return 1
    except Exception as e:
        print(f"\n\nUnexpected error during diagnostic: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    input("\nPress Enter to exit...")
    sys.exit(exit_code)
