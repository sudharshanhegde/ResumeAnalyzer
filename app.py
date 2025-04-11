import os
import re
import json
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file
from werkzeug.utils import secure_filename
import markdown
from collections import Counter
import tempfile
import io

# Initialize Flask app
app = Flask(__name__, template_folder='app/templates', static_folder='app/static')
app.config['SECRET_KEY'] = 'resumeanalyzer123'
app.config['UPLOAD_FOLDER'] = 'app/static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['ALLOWED_EXTENSIONS'] = {'md', 'markdown', 'txt'}

# Make sure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_text_from_markdown(md_path):
    """Extract text from Markdown file"""
    try:
        with open(md_path, 'r', encoding='utf-8') as file:
            text = file.read()
            # Convert markdown to plain text for analysis
            html = markdown.markdown(text)
            # Remove HTML tags
            text = re.sub(r'<[^>]+>', '', html)
            return text
    except Exception as e:
        print(f"Error extracting text from Markdown: {e}")
        return None

def basic_entity_extraction(text):
    """Simple entity extraction based on regex patterns"""
    entities = []
    
    # Simple pattern for dates
    date_pattern = r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b\d{4}\b'
    dates = re.findall(date_pattern, text)
    for date in dates[:10]:  # Limit to first 10 dates
        entities.append({"text": date, "label": "DATE"})
    
    # Simple pattern for organizations (uppercase words)
    org_pattern = r'\b[A-Z][A-Z]+\b'
    orgs = re.findall(org_pattern, text)
    # Convert set to list before slicing
    unique_orgs = list(set(orgs))[:10]  # Limit to first 10 unique orgs
    for org in unique_orgs:
        entities.append({"text": org, "label": "ORG"})
    
    return entities

def analyze_resume(resume_text, job_description=None):
    """Analyze resume text without using spaCy"""
    # Common technical skills to look for
    tech_skills = [
        "python", "java", "javascript", "html", "css", "sql", "nosql", "react", 
        "angular", "vue", "node", "express", "django", "flask", "spring", 
        "aws", "azure", "gcp", "docker", "kubernetes", "ci/cd", "git",
        "machine learning", "data science", "nlp", "ai", "artificial intelligence"
    ]
    
    # Look for skills in resume
    resume_lower = resume_text.lower()
    skills = []
    for skill in tech_skills:
        if skill in resume_lower:
            skills.append(skill)
    
    # Extract basic entities
    entities = basic_entity_extraction(resume_text)
    
    # Compare with job description if provided
    match_percentage = 0
    missing_skills = []
    improvement_suggestions = []
    
    if job_description:
        job_description_lower = job_description.lower()
        job_skills = []
        
        # Find skills in job description
        for skill in tech_skills:
            if skill in job_description_lower:
                job_skills.append(skill)
        
        # Calculate match percentage
        if job_skills:
            matches = [skill for skill in skills if skill in job_skills]
            match_percentage = round((len(matches) / len(job_skills)) * 100, 2) if len(job_skills) > 0 else 0
            missing_skills = [skill for skill in job_skills if skill not in skills]
            
            # Generate improvement suggestions
            if missing_skills:
                improvement_suggestions.append({
                    "type": "skills",
                    "message": f"Consider adding these skills to your resume: {', '.join(missing_skills)}"
                })
    
    return {
        "skills": skills,
        "entities": entities,
        "match_percentage": match_percentage,
        "improvement_suggestions": improvement_suggestions
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'resume' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['resume']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Extract text from markdown
        resume_text = extract_text_from_markdown(filepath)
        if not resume_text:
            flash('Error extracting text from file')
            return redirect(request.url)
        
        # Get job description if provided
        job_description = request.form.get('job_description', '')
        
        # Analyze resume
        analysis_results = analyze_resume(resume_text, job_description)
        
        # Read the original markdown content for editing
        with open(filepath, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
        
        return render_template('editor.html', 
                             resume_text=markdown_content,
                             analysis=analysis_results,
                             filename=filename)
    
    flash('Invalid file type. Please upload a Markdown file.')
    return redirect(request.url)

@app.route('/edit-resume/<filename>')
def edit_resume(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        flash('File not found')
        return redirect(url_for('index'))
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            resume_text = f.read()
        return render_template('editor.html', resume_text=resume_text, filename=filename)
    except Exception as e:
        flash(f'Error reading file: {str(e)}')
        return redirect(url_for('index'))

@app.route('/save-resume', methods=['POST'])
def save_resume():
    try:
        data = request.get_json()
        resume_text = data.get('resume_text', '')
        
        if not resume_text:
            return jsonify({'success': False, 'error': 'No resume text provided'})
        
        # Save the markdown content
        filename = 'resume.md'
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(resume_text)
        
        return jsonify({'success': True, 'filename': filename})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/download-resume/<filename>')
def download_resume(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        flash('File not found')
        return redirect(url_for('index'))
    
    return send_file(
        filepath,
        mimetype='text/markdown',
        as_attachment=True,
        download_name=filename
    )

if __name__ == '__main__':
    app.run(debug=True, port=5001) 