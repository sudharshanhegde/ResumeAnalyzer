# Resume Analyzer

An intelligent ATS (Applicant Tracking System) resume analysis tool that provides accurate skill matching and comprehensive feedback using advanced Natural Language Processing.

## Quick Start

1. Run `setup.bat` to install dependencies
2. Run `run.bat` to start the application
3. Open your browser to `http://localhost:5001`
4. Upload your resume (PDF format) and job description
5. Get detailed analysis and improvement suggestions

## Features

### Core Functionality

- PDF resume text extraction
- Advanced skill detection with context validation
- Job description matching with ATS-style scoring
- Missing skills identification
- Professional improvement suggestions
- Resume editor with PDF export

### Advanced Analysis

- Named Entity Recognition (contact info, dates, organizations)
- Semantic similarity analysis between resume and job description
- Text readability assessment
- Sentiment analysis
- Multiple confidence scoring algorithms
- Category-based skill organization

### Technical Capabilities

- False positive prevention for ambiguous skills (e.g., 'R', 'Go', 'C')
- Context-aware pattern matching
- Multiple NLP model support (spaCy, NLTK, Transformers)
- Comprehensive skill database with 500+ technical skills
- ATS optimization recommendations

## Installation

### Requirements

- Python 3.7+
- Windows OS (for batch files)

### Setup Process

1. Clone or download the repository
2. Run `setup.bat` - this will:
   - Create virtual environment
   - Install required packages
   - Download necessary NLP models
3. Use `run.bat` to start the application

### Manual Installation

```bash
python -m venv venv311
venv311\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python app.py
```

## Project Structure

```
resume-analyzer/
├── app/
│   ├── static/
│   │   ├── css/
│   │   │   ├── modern.css
│   │   │   └── style.css
│   │   ├── js/
│   │   └── uploads/              # Resume storage
│   └── templates/
│       ├── index.html            # Upload interface
│       ├── results.html          # Analysis results
│       └── editor.html           # Resume editor
├── venv311/                      # Virtual environment
├── app.py                        # Main Flask application
├── advanced_skills_extractor.py  # Core ATS analysis engine
├── lightning_extractor.py        # Fast extraction mode
├── requirements.txt              # Python dependencies
├── setup.bat                     # Installation script
├── run.bat                       # Application launcher
└── README.md                     # This file
```

## Usage

### Basic Analysis

1. Upload PDF resume
2. Optionally paste job description
3. Click "Analyze My Resume"
4. Review match percentage and detected skills
5. Follow improvement suggestions

### Advanced Features

- Edit resume text directly in browser
- Export edited resume as PDF
- View detailed skill categories
- Check entity recognition results
- Access readability and sentiment analysis

## Technology Stack

- **Backend**: Flask (Python web framework)
- **NLP Libraries**: spaCy, NLTK, scikit-learn, Transformers
- **PDF Processing**: PyPDF2, ReportLab
- **Frontend**: HTML5, CSS3, JavaScript
- **UI Components**: Boxicons, custom CSS framework

## Analysis Capabilities

### Skill Detection

- Programming languages, frameworks, tools
- Database technologies
- Cloud platforms and DevOps tools
- AI/ML technologies
- Soft skills and certifications

### Pattern Matching

- Context-aware skill validation
- False positive prevention
- Multiple confidence scoring methods
- Industry-specific terminology recognition

### Job Matching

- ATS-style scoring algorithm
- Semantic similarity analysis
- Missing skills identification
- Category-based comparison
- Improvement prioritization

## Configuration

The application uses forced advanced mode for optimal accuracy:

- Advanced NLP analysis (5-15 seconds processing)
- Context validation for ambiguous skills
- Comprehensive entity recognition
- Semantic similarity scoring

## Troubleshooting

### Common Issues

- **Port 5001 in use**: Change port in app.py or stop other applications
- **PDF not reading**: Ensure PDF contains selectable text, not scanned images
- **Missing models**: Run `python -m spacy download en_core_web_sm`
- **Performance issues**: Close other applications, ensure adequate RAM

### Debug Mode

- Application runs in debug mode by default
- Check console output for detailed error messages
- Verify virtual environment activation

## File Management

### Safe to Delete

Based on .gitignore configuration, these files can be safely removed:

- Redundant launcher scripts (keep only run.bat, setup.bat)
- Test and demo files
- Extra documentation files
- Alternative extractor files
- Development virtual environments (ui_venv, etc.)

### Core Files (Keep)

- app.py
- advanced_skills_extractor.py
- lightning_extractor.py
- app/ directory (templates, static files)
- requirements.txt
- setup.bat, run.bat

## License

MIT License - see LICENSE file for details.

## Version Information

Current Version: Advanced ATS Checker with Context Validation
Last Updated: July 2025
Python Compatibility: 3.7+
