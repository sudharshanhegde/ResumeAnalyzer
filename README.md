# Resume Analyzer

A modern web application that analyzes resumes (PDF) and compares them with job descriptions to identify skill gaps and provide personalized improvement suggestions.

## Features

- Upload and analyze PDF resumes
- Extract text from PDFs using PyPDF2
- Detect skills and keywords from resume text
- Compare resume content with job descriptions 
- Identify skill gaps and calculate match percentage
- Provide personalized resume improvement suggestions
- Edit resume text directly in the browser
- Generate and download updated PDF resumes
- Modern, responsive UI with interactive elements

## Setup

### First-Time Setup
1. Run the setup script (only needed once):
```
setup.bat
```
2. This will create a virtual environment and install all dependencies

### Running the Application
1. After setup is complete, use the run script:
```
run.bat
```
2. The application will start automatically
3. Open your browser and navigate to `http://localhost:5001`

### Alternative Setup (Manual)
1. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```
3. Run the application:
```bash
python app.py
```
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