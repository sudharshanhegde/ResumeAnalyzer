# Development Guide - Resume Analyzer

## 📁 Project Structure

```
resume-analyzer/
├── app/
│   ├── static/
│   │   ├── css/modern.css          # Professional UI styling
│   │   ├── js/                     # JavaScript files
│   │   └── uploads/                # Resume uploads (ignored in git)
│   └── templates/
│       ├── index.html              # Main upload page
│       ├── results.html            # Analysis results
│       └── editor.html             # Resume editor
├── app.py                          # Main Flask application
├── requirements_nlp.txt            # Complete NLP dependencies
├── diagnose_nlp.py                 # Diagnostic tool
├── setup_clean_nlp.bat            # Clean installation
├── quick_start.bat                 # Easy launcher
├── easy_launcher.bat               # Interactive menu
└── .gitignore                      # Git ignore rules
```

## 🚀 Development Setup

### 1. Clone and Setup

```bash
git clone https://github.com/sudharshanhegde/ResumeAnalyzer.git
cd ResumeAnalyzer
setup_clean_nlp.bat
```

### 2. Development Environment

```bash
# Activate virtual environment
nlp_resume_env\Scripts\activate.bat

# Install development dependencies
pip install -r requirements_nlp.txt

# Run diagnostics
python diagnose_nlp.py
```

### 3. Start Development Server

```bash
# Easy method
quick_start.bat

# Or manual method
python app.py
```

## 🔧 Development Guidelines

### Code Organization

- **app.py**: Main Flask application with NLP functions
- **templates/**: HTML templates with Jinja2 syntax
- **static/css/**: Modern CSS with CSS variables
- **static/js/**: JavaScript for interactivity

### NLP Components

- **spaCy**: Industrial NLP for entity recognition
- **Transformers**: BERT-based sentiment analysis
- **Sentence Transformers**: Semantic similarity
- **NLTK**: Traditional NLP processing
- **scikit-learn**: TF-IDF and ML utilities

### Adding New Features

1. **Skills**: Update skills database in `app.py`
2. **Entities**: Extend entity extraction functions
3. **Analysis**: Add new analysis methods
4. **UI**: Update templates and CSS
5. **Dependencies**: Update `requirements_nlp.txt`

### Testing

```bash
# Run diagnostics
python diagnose_nlp.py

# Test specific components
python -c "import app; print('App imports OK')"
```

## 📦 Deployment

### Local Deployment

```bash
# Production setup
setup_clean_nlp.bat

# Start application
quick_start.bat
```

### Git Workflow

```bash
# Clean project
cleanup_project.bat

# Add and commit
git add .
git commit -m "Your changes"
git push origin main
```

## 🔍 Debugging

### Common Issues

1. **Import Errors**: Run `setup_clean_nlp.bat`
2. **Model Not Found**: Check internet connection
3. **Port In Use**: Run `easy_launcher.bat` → Stop Application
4. **Memory Issues**: Close other applications

### Debug Tools

- **diagnose_nlp.py**: Complete system check
- **easy_launcher.bat**: Interactive troubleshooting
- **Flask Debug Mode**: Set `debug=True` in app.py

## 📋 Maintenance

### Regular Tasks

1. **Update Dependencies**: `pip install -r requirements_nlp.txt --upgrade`
2. **Clean Project**: `cleanup_project.bat`
3. **Test Installation**: `python diagnose_nlp.py`
4. **Update Models**: Download latest spaCy models

### Performance Optimization

- **First Run**: Models download automatically (be patient)
- **Subsequent Runs**: Models cached for speed
- **Memory Usage**: Close unused applications
- **Disk Space**: Clean uploaded files regularly

## 🤝 Contributing

1. **Fork** the repository
2. **Create** feature branch: `git checkout -b feature-name`
3. **Develop** your changes
4. **Test** thoroughly with `diagnose_nlp.py`
5. **Clean** with `cleanup_project.bat`
6. **Submit** pull request

## 📄 Code Style

### Python

- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Add docstrings for functions
- Handle errors gracefully

### HTML/CSS

- Use semantic HTML5 elements
- Maintain responsive design
- Use CSS variables for theming
- Follow BEM methodology

### JavaScript

- Use modern ES6+ features
- Add comments for complex logic
- Handle errors appropriately
- Test across browsers

---

**Happy Coding! 🚀**
