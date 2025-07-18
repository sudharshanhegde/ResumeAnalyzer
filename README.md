# Advanced NLP Resume Analyzer

An intelligent resume analysis tool powered by cutting-edge Natural Language Processing (NLP) techniques.

## 🚀 Quick Start (Super Easy!)

### **Just Double-Click and Go!**

1. **Desktop Shortcut**: Double-click "Resume Analyzer" on your desktop
2. **Quick Start**: Double-click `quick_start.bat` 
3. **Interactive Menu**: Double-click `easy_launcher.bat`

**That's it!** The application will start automatically and open in your browser at `http://localhost:5001`

## 🎯 Advanced NLP Features

### Multi-Model Named Entity Recognition (NER)
- **spaCy NER**: Industrial-strength entity extraction for persons, organizations, locations, dates, and more
- **Transformer-based NER**: BERT-based models for high-accuracy entity recognition
- **NLTK NER**: Traditional linguistic approach for comprehensive coverage
- **Enhanced Regex Patterns**: Custom patterns for emails, phones, degrees, GPAs, and URLs

### Advanced Skills Extraction
- **Categorized Skills Database**: Organized by programming languages, web technologies, databases, cloud platforms, AI/ML, and tools
- **Multiple Detection Methods**:
  - Exact string matching with confidence scoring
  - TF-IDF similarity analysis
  - spaCy semantic similarity
  - Context-aware skill identification

### Semantic Analysis
- **Semantic Similarity**: Deep learning-based comparison between resume and job description using sentence transformers
- **Content Alignment**: Measures how well your resume matches the job requirements beyond simple keyword matching

### Text Analytics
- **Readability Analysis**: Multiple readability metrics including Flesch Reading Ease, Gunning Fog Index, and more
- **Sentiment Analysis**: AI-powered sentiment detection using transformer models
- **Text Statistics**: Comprehensive analysis of word count, sentence structure, and content complexity

### Intelligent Suggestions Engine
- **Priority-based Recommendations**: High, medium, and low priority suggestions with impact analysis
- **Contextual Advice**: Suggestions based on semantic similarity, readability scores, and sentiment analysis
- **ATS Optimization**: Specific recommendations for Applicant Tracking System compatibility

## 🛠️ Setup and Installation

### **First Time Setup**

1. **Run the clean installer:**
   ```bash
   setup_clean_nlp.bat
   ```

2. **Create desktop shortcut:**
   ```bash
   create_shortcut.ps1
   ```

3. **Start using the application:**
   - Double-click desktop shortcut, OR
   - Double-click `quick_start.bat`

### **If You Need Help**
- Run `diagnose_nlp.py` for troubleshooting
- Check `EASY_ACCESS_GUIDE.md` for detailed instructions
- Use `easy_launcher.bat` for interactive menu

## 🏗️ Project Structure

```
resume-analyzer/
├── app/
│   ├── static/
│   │   ├── css/
│   │   │   └── modern.css          # Professional UI styling
│   │   ├── js/
│   │   └── uploads/                # Resume upload directory
│   │       └── .gitkeep           # Keep directory in git
│   └── templates/
│       ├── index.html              # Main upload page
│       ├── results.html            # Analysis results page
│       └── editor.html             # Resume editor page
├── nlp_resume_env/                 # Virtual environment (ignored)
├── app.py                          # Main Flask application
├── requirements_nlp.txt            # Complete NLP dependencies
├── requirements.txt                # Basic dependencies
├── diagnose_nlp.py                 # Diagnostic and troubleshooting tool
├── setup_clean_nlp.bat            # Clean installation script
├── quick_start.bat                 # Easy launcher (recommended)
├── easy_launcher.bat               # Interactive menu launcher
├── start_analyzer.bat              # Application starter
├── stop_analyzer.bat               # Application stopper
├── check_status.bat                # Status checker
├── EASY_ACCESS_GUIDE.md            # User guide for easy access
├── LAUNCH_GUIDE.md                 # Detailed launch instructions
├── .gitignore                      # Git ignore file
└── README.md                       # This file
```

## 🚀 Usage

### **Easy Method (Recommended)**

1. **Double-click** `quick_start.bat` or the desktop shortcut
2. **Wait** for the browser to open automatically
3. **Upload** your resume (PDF format)
4. **Paste** a job description (optional but recommended)
5. **Click** "Analyze My Resume"
6. **Get** comprehensive AI-powered analysis!

### **Advanced Method**

1. **Double-click** `easy_launcher.bat`
2. **Choose** from the interactive menu:
   - [1] Quick Start (recommended)
   - [2] Open Application (if already running)
   - [3] Stop Application
   - [4] Check Status
   - [5] Run Diagnostics

### **Manual Method**

```bash
# Activate virtual environment
nlp_resume_env\Scripts\activate.bat

# Run the application
python app.py
```

## 🔧 Troubleshooting

### **Application won't start?**
1. Run `easy_launcher.bat` → Choose [5] Run Diagnostics
2. Check the output for any missing dependencies
3. If needed, run `setup_clean_nlp.bat` for fresh installation

### **Models not downloading?**
1. Check your internet connection
2. Run `diagnose_nlp.py` to verify installation
3. Models are downloaded on first run (may take a few minutes)

### **Port already in use?**
1. Run `easy_launcher.bat` → Choose [3] Stop Application
2. Or restart your computer
3. Try starting again

### **Performance issues?**
1. First run downloads large AI models (be patient)
2. Subsequent runs will be much faster
3. Close other applications to free up memory

### 🚀 **Running the Application**

After successful setup:

```bash
run_nlp.bat
```

Or manually:

```bash
nlp_resume_env\Scripts\activate.bat
python app.py
```

Open your browser and navigate to `http://localhost:5001`

4. Open your browser and navigate to `http://localhost:5001`

### Cleaning Up

If you have multiple virtual environments in your project folder (venv, venv_simple, etc.),
you can safely delete them after closing all python processes:

```bash
# First make sure all Python processes are closed
# Then delete the virtual environment folders manually
# Or use this command if you have permission:
Remove-Item -Recurse -Force venv, venv_simple, ui_venv, direct_venv, simple_venv
```

Only the `venv` folder created by `setup.bat` is needed for the application to run.

## Project Structure

```
resume-analyzer/
├── app/
│   ├── static/
│   │   ├── css/
│   │   │   └── modern.css
│   │   ├── js/
│   │   └── uploads/
│   └── templates/
│       ├── index.html
│       └── results.html
├── app.py
├── requirements.txt
├── run.bat
└── README.md
```

## Technology Stack

- **Backend**: Flask (Python)
- **PDF Processing**: PyPDF2
- **Frontend**: HTML, CSS (with modern design system)
- **Icons**: Boxicons

## How It Works

1. **Upload Your Resume**:

   - Submit your resume in PDF format
   - The system extracts text content using PyPDF2

2. **Analyze Job Description (Optional)**:

   - Paste a job description to compare with your resume
   - The analyzer extracts key skills from both

3. **Get Comprehensive Analysis**:

   - See detected skills in your resume
   - Identify missing skills based on the job description
   - View match percentage with visual indicators
   - Receive detailed improvement suggestions

4. **Edit and Improve Your Resume**:
   - Edit your resume text directly in the browser
   - Follow the personalized suggestions to improve your resume
   - Save changes and download the updated PDF

## Future Enhancements

- Improved keyword extraction algorithms
- Grammar check functionality
- Export analysis report as PDF
- User accounts to save previous analyses
- Advanced visualization of skill matches

## Screenshots

The application features a modern, clean UI with:

- Card-based layout
- Progress indicators
- Color-coded skill tags
- Responsive design for all devices

## License

MIT

## 🧠 NLP Techniques Explained

### 1. Named Entity Recognition (NER)

Multiple approaches ensure comprehensive entity extraction:

- **Statistical Models**: spaCy's CNN-based models for fast, accurate entity recognition
- **Transformer Models**: BERT-based models for state-of-the-art accuracy
- **Rule-based Systems**: NLTK's linguistic patterns for broad coverage
- **Custom Patterns**: Specialized regex for resume-specific entities

### 2. Semantic Similarity

Goes beyond keyword matching:

- **Sentence Transformers**: Creates dense vector representations of text
- **Cosine Similarity**: Measures semantic distance between resume and job description
- **Contextual Understanding**: Identifies related concepts even with different wording

### 3. Text Analytics

Comprehensive text quality assessment:

## 🔬 Technical Details

### **Core NLP Technologies**

1. **spaCy**: Industrial-strength NLP with English language models
2. **Transformers**: Hugging Face library for BERT-based sentiment analysis
3. **Sentence Transformers**: Semantic similarity using sentence-BERT
4. **NLTK**: Traditional NLP toolkit for tokenization and analysis
5. **scikit-learn**: TF-IDF vectorization and machine learning
6. **TextStat**: Comprehensive readability analysis

### **AI Models Used**

- **spaCy en_core_web_sm**: Named entity recognition and POS tagging
- **cardiffnlp/twitter-roberta-base-sentiment**: Sentiment analysis
- **all-MiniLM-L6-v2**: Sentence embeddings for semantic similarity
- **TF-IDF**: Term frequency analysis for keyword matching

### **Analysis Capabilities**

- **Entity Recognition**: Persons, organizations, locations, dates, emails, phones
- **Skill Categorization**: 500+ technical skills across 15+ categories
- **Semantic Similarity**: Deep learning-based content comparison
- **Readability Analysis**: Multiple linguistic complexity metrics
- **Sentiment Analysis**: Professional tone and confidence assessment

## 🎯 Benefits

### **For Job Seekers**
- **Higher Accuracy**: Multiple NLP models ensure comprehensive analysis
- **Semantic Understanding**: Matches concepts, not just keywords
- **Intelligent Feedback**: AI-powered suggestions with priority ranking
- **ATS Optimization**: Ensures compatibility with automated screening systems

### **For Recruiters**
- **Deeper Insights**: Understanding beyond surface-level keyword matching
- **Quality Assessment**: Readability and sentiment analysis of candidate materials
- **Semantic Matching**: Find candidates with related skills and experience

## 📝 Example Analysis Output

### **Match Score**: 87% (Excellent match!)
### **Sentiment**: Positive (89% confidence)
### **Readability**: Grade 12.3 (Professional level)
### **Skills Found**: 23 technical skills across 8 categories
### **Entities**: 15 organizations, 8 locations, 12 dates
### **Suggestions**: 5 high-priority improvements

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆕 Version 2.0 - Advanced NLP Edition

**Latest Features:**
- Complete NLP pipeline with multiple AI models
- Professional modern UI with enhanced analytics
- Easy launcher system with desktop shortcuts
- Comprehensive diagnostic and troubleshooting tools
- Clean project structure with proper `.gitignore`

---

**Created with ❤️ by AI-Powered Resume Analysis System**  
**Version**: 2.0 (Advanced NLP Edition)  
**Last Updated**: July 2025

_Powered by state-of-the-art NLP models from spaCy, Hugging Face Transformers, and NLTK_
