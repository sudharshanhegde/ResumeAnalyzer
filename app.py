import os
import re
import json
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file
from werkzeug.utils import secure_filename
import PyPDF2  # Using PyPDF2 instead of PyMuPDF
from collections import Counter
import tempfile
import io
import warnings
warnings.filterwarnings('ignore')

# Advanced NLP imports with fallback handling
NLP_AVAILABLE = True
nlp_models = {
    'spacy': None,
    'sentiment': None,
    'ner': None,
    'sentence_transformer': None
}

try:
    import spacy
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.chunk import ne_chunk
    from nltk.tag import pos_tag
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from transformers import pipeline, AutoTokenizer, AutoModel
    from sentence_transformers import SentenceTransformer
    import textstat
    import numpy as np
    
    print("✅ Advanced NLP libraries loaded successfully!")
    
except ImportError as e:
    print(f"⚠️  Advanced NLP libraries not available: {e}")
    print("📝 Running in basic mode. For advanced features, run: setup_clean_nlp.bat")
    NLP_AVAILABLE = False
    
    # Fallback imports for basic functionality
    try:
        from nltk.tokenize import sent_tokenize
        import nltk
    except ImportError:
        # Define fallback sentence tokenizer
        def sent_tokenize(text):
            sentences = re.split(r'[.!?]+', text)
            return [s.strip() for s in sentences if s.strip()]
        
        class MockNLTK:
            @staticmethod
            def download(*args, **kwargs):
                pass
        
        nltk = MockNLTK()

def load_nlp_models():
    """Lazy load NLP models to improve startup time"""
    global nlp_models
    
    if not NLP_AVAILABLE:
        print("⚠️  NLP models not available. Running in basic mode.")
        return
    
    try:
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
            
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            nltk.download('averaged_perceptron_tagger')
            
        try:
            nltk.data.find('chunkers/maxent_ne_chunker')
        except LookupError:
            nltk.download('maxent_ne_chunker')
            
        try:
            nltk.data.find('corpora/words')
        except LookupError:
            nltk.download('words')
        
        # Load spaCy model
        if nlp_models['spacy'] is None:
            try:
                nlp_models['spacy'] = spacy.load("en_core_web_sm")
                print("✅ spaCy model loaded")
            except OSError:
                print("⚠️  spaCy model not found. Run: python -m spacy download en_core_web_sm")
                nlp_models['spacy'] = None
        
        # Load transformer models (with error handling)
        if nlp_models['sentiment'] is None:
            try:
                nlp_models['sentiment'] = pipeline("sentiment-analysis", 
                                                  model="cardiffnlp/twitter-roberta-base-sentiment-latest")
                print("✅ Sentiment analysis model loaded")
            except Exception as e:
                print(f"⚠️  Sentiment model not available: {e}")
                nlp_models['sentiment'] = None
        
        if nlp_models['ner'] is None:
            try:
                nlp_models['ner'] = pipeline("ner", 
                                           model="dbmdz/bert-large-cased-finetuned-conll03-english",
                                           aggregation_strategy="simple")
                print("✅ NER model loaded")
            except Exception as e:
                print(f"⚠️  NER model not available: {e}")
                nlp_models['ner'] = None
        
        if nlp_models['sentence_transformer'] is None:
            try:
                nlp_models['sentence_transformer'] = SentenceTransformer('all-MiniLM-L6-v2')
                print("✅ Sentence transformer loaded")
            except Exception as e:
                print(f"⚠️  Sentence transformer not available: {e}")
                nlp_models['sentence_transformer'] = None
                
    except Exception as e:
        print(f"⚠️  Error loading NLP models: {e}")
        print("📝 For full NLP features, run: setup_clean_nlp.bat")

# Initialize Flask app
app = Flask(__name__, template_folder='app/templates', static_folder='app/static')
app.config['SECRET_KEY'] = 'resumeanalyzer123'
app.config['UPLOAD_FOLDER'] = 'app/static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

# Make sure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file using PyPDF2"""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                text += reader.pages[page_num].extract_text()
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return None
    return text

def advanced_entity_extraction(text):
    """Advanced entity extraction using multiple NLP techniques"""
    entities = []
    
    # Load models if not already loaded
    load_nlp_models()
    
    # Method 1: spaCy NER (if available)
    if nlp_models['spacy']:
        try:
            doc = nlp_models['spacy'](text)
            for ent in doc.ents:
                if ent.label_ in ['PERSON', 'ORG', 'GPE', 'DATE', 'MONEY', 'PERCENT']:
                    entities.append({
                        "text": ent.text,
                        "label": ent.label_,
                        "confidence": 0.9,  # spaCy doesn't provide confidence scores directly
                        "method": "spacy"
                    })
        except Exception as e:
            print(f"spaCy NER error: {e}")
    
    # Method 2: Transformer-based NER (if available)
    if nlp_models['ner']:
        try:
            ner_results = nlp_models['ner'](text[:512])  # Limit text length for transformer
            for entity in ner_results:
                entities.append({
                    "text": entity['word'],
                    "label": entity['entity_group'],
                    "confidence": entity['score'],
                    "method": "transformer"
                })
        except Exception as e:
            print(f"Transformer NER error: {e}")
    
    # Method 3: NLTK NER (fallback)
    try:
        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)
        chunks = ne_chunk(pos_tags)
        
        for chunk in chunks:
            if hasattr(chunk, 'label'):
                entity_text = ' '.join([token for token, pos in chunk.leaves()])
                entities.append({
                    "text": entity_text,
                    "label": chunk.label(),
                    "confidence": 0.7,
                    "method": "nltk"
                })
    except Exception as e:
        print(f"NLTK NER error: {e}")
    
    # Method 4: Regex patterns (enhanced)
    patterns = {
        'EMAIL': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'PHONE': r'\b(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}\b',
        'DATE': r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b',
        'URL': r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
        'DEGREE': r'\b(?:B\.?S\.?|M\.?S\.?|Ph\.?D\.?|B\.?A\.?|M\.?A\.?|MBA|Bachelor|Master|Doctor)\b',
        'GPA': r'\b(?:GPA|CGPA)[\s:]*(\d+\.?\d*)\b'
    }
    
    for label, pattern in patterns.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            entities.append({
                "text": match,
                "label": label,
                "confidence": 0.8,
                "method": "regex"
            })
    
    # Remove duplicates and sort by confidence
    unique_entities = []
    seen = set()
    for entity in entities:
        key = (entity['text'].lower(), entity['label'])
        if key not in seen:
            seen.add(key)
            unique_entities.append(entity)
    
    return sorted(unique_entities, key=lambda x: x['confidence'], reverse=True)

def extract_skills_advanced(text):
    """Advanced skills extraction using multiple techniques"""
    load_nlp_models()
    
    # Expanded skills database with categories
    skills_database = {
        'programming_languages': [
            'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'c', 'go', 
            'rust', 'kotlin', 'swift', 'php', 'ruby', 'scala', 'r', 'matlab', 'perl'
        ],
        'web_technologies': [
            'html', 'css', 'react', 'angular', 'vue', 'node.js', 'express', 'django', 
            'flask', 'spring', 'asp.net', 'laravel', 'bootstrap', 'jquery', 'webpack'
        ],
        'databases': [
            'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 
            'oracle', 'sqlite', 'cassandra', 'dynamodb', 'neo4j'
        ],
        'cloud_platforms': [
            'aws', 'azure', 'gcp', 'google cloud', 'heroku', 'digitalocean', 
            'kubernetes', 'docker', 'terraform', 'ansible'
        ],
        'ai_ml': [
            'machine learning', 'deep learning', 'neural networks', 'tensorflow', 
            'pytorch', 'scikit-learn', 'pandas', 'numpy', 'nlp', 'computer vision',
            'data science', 'artificial intelligence', 'keras', 'opencv'
        ],
        'tools': [
            'git', 'github', 'gitlab', 'jenkins', 'ci/cd', 'jira', 'confluence', 
            'slack', 'docker', 'vagrant', 'vim', 'vscode', 'intellij'
        ]
    }
    
    # Flatten skills for processing
    all_skills = []
    for category, skills in skills_database.items():
        all_skills.extend([(skill, category) for skill in skills])
    
    # Method 1: Exact matching (case-insensitive)
    text_lower = text.lower()
    found_skills = {}
    
    for skill, category in all_skills:
        if skill.lower() in text_lower:
            if category not in found_skills:
                found_skills[category] = []
            found_skills[category].append({
                'skill': skill,
                'confidence': 0.9,
                'method': 'exact_match'
            })
    
    # Method 2: TF-IDF similarity (if sklearn is available)
    try:
        vectorizer = TfidfVectorizer()
        skill_texts = [skill for skill, _ in all_skills]
        skill_vectors = vectorizer.fit_transform(skill_texts + [text])
        text_vector = skill_vectors[-1]
        skill_vectors = skill_vectors[:-1]
        
        similarities = cosine_similarity(text_vector, skill_vectors).flatten()
        
        for i, (skill, category) in enumerate(all_skills):
            if similarities[i] > 0.1:  # Threshold for similarity
                if category not in found_skills:
                    found_skills[category] = []
                # Avoid duplicates
                if not any(s['skill'] == skill for s in found_skills[category]):
                    found_skills[category].append({
                        'skill': skill,
                        'confidence': float(similarities[i]),
                        'method': 'tfidf'
                    })
    except Exception as e:
        print(f"TF-IDF skills extraction error: {e}")
    
    # Method 3: spaCy similarity (if available)
    if nlp_models['spacy']:
        try:
            doc = nlp_models['spacy'](text)
            for skill, category in all_skills:
                skill_doc = nlp_models['spacy'](skill)
                similarity = doc.similarity(skill_doc)
                if similarity > 0.3:  # Threshold for similarity
                    if category not in found_skills:
                        found_skills[category] = []
                    if not any(s['skill'] == skill for s in found_skills[category]):
                        found_skills[category].append({
                            'skill': skill,
                            'confidence': float(similarity),
                            'method': 'spacy_similarity'
                        })
        except Exception as e:
            print(f"spaCy similarity error: {e}")
    
    return found_skills

def calculate_semantic_similarity(text1, text2):
    """Calculate semantic similarity between two texts"""
    load_nlp_models()
    
    if nlp_models['sentence_transformer']:
        try:
            embeddings = nlp_models['sentence_transformer'].encode([text1, text2])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(similarity)
        except Exception as e:
            print(f"Semantic similarity error: {e}")
    
    # Fallback to TF-IDF similarity
    try:
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([text1, text2])
        similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
        return float(similarity)
    except Exception as e:
        print(f"TF-IDF similarity error: {e}")
        return 0.0

def analyze_text_readability(text):
    """Analyze text readability and complexity"""
    try:
        readability_scores = {
            'flesch_reading_ease': textstat.flesch_reading_ease(text),
            'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
            'automated_readability_index': textstat.automated_readability_index(text),
            'coleman_liau_index': textstat.coleman_liau_index(text),
            'gunning_fog': textstat.gunning_fog(text),
            'smog_index': textstat.smog_index(text),
            'reading_time': textstat.reading_time(text)
        }
        return readability_scores
    except Exception as e:
        print(f"Readability analysis error: {e}")
        return {}

def analyze_sentiment(text):
    """Analyze sentiment of the text"""
    load_nlp_models()
    
    if nlp_models['sentiment']:
        try:
            # Split text into chunks if too long
            max_length = 512
            if len(text) > max_length:
                chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
                sentiments = []
                for chunk in chunks:
                    result = nlp_models['sentiment'](chunk)
                    sentiments.append(result[0])
                
                # Average the sentiment scores
                positive_scores = [s['score'] for s in sentiments if s['label'] == 'POSITIVE']
                negative_scores = [s['score'] for s in sentiments if s['label'] == 'NEGATIVE']
                neutral_scores = [s['score'] for s in sentiments if s['label'] == 'NEUTRAL']
                
                if positive_scores:
                    avg_positive = sum(positive_scores) / len(positive_scores)
                    return {'label': 'POSITIVE', 'score': avg_positive}
                elif negative_scores:
                    avg_negative = sum(negative_scores) / len(negative_scores)
                    return {'label': 'NEGATIVE', 'score': avg_negative}
                else:
                    avg_neutral = sum(neutral_scores) / len(neutral_scores) if neutral_scores else 0.5
                    return {'label': 'NEUTRAL', 'score': avg_neutral}
            else:
                result = nlp_models['sentiment'](text)
                return result[0]
        except Exception as e:
            print(f"Sentiment analysis error: {e}")
    
    return {'label': 'NEUTRAL', 'score': 0.5}

def analyze_resume(resume_text, job_description=None):
    """Enhanced resume analysis using advanced NLP techniques"""
    
    # Initialize results structure
    analysis_results = {
        "skills": {},
        "entities": [],
        "match_percentage": 0,
        "missing_skills": [],
        "improvement_suggestions": [],
        "readability": {},
        "sentiment": {},
        "semantic_similarity": 0,
        "advanced_insights": {}
    }
    
    # Extract skills using advanced techniques
    skills_by_category = extract_skills_advanced(resume_text)
    analysis_results["skills"] = skills_by_category
    
    # Extract entities using multiple NLP techniques
    entities = advanced_entity_extraction(resume_text)
    analysis_results["entities"] = entities
    
    # Analyze text readability
    readability = analyze_text_readability(resume_text)
    analysis_results["readability"] = readability
    
    # Analyze sentiment
    sentiment = analyze_sentiment(resume_text)
    analysis_results["sentiment"] = sentiment
    
    # Advanced insights
    advanced_insights = {
        "total_entities": len(entities),
        "entity_types": list(set([ent["label"] for ent in entities])),
        "total_skills": sum(len(skills) for skills in skills_by_category.values()),
        "skill_categories": list(skills_by_category.keys()),
        "text_length": len(resume_text),
        "word_count": len(resume_text.split()),
        "sentence_count": len(sent_tokenize(resume_text)) if resume_text else 0
    }
    analysis_results["advanced_insights"] = advanced_insights
    
    # Compare with job description if provided
    if job_description:
        # Calculate semantic similarity
        semantic_similarity = calculate_semantic_similarity(resume_text, job_description)
        analysis_results["semantic_similarity"] = semantic_similarity
        
        # Extract skills from job description
        job_skills = extract_skills_advanced(job_description)
        
        # Calculate advanced match percentage
        total_job_skills = sum(len(skills) for skills in job_skills.values())
        total_resume_skills = sum(len(skills) for skills in skills_by_category.values())
        
        if total_job_skills > 0:
            matched_skills = 0
            missing_skills_by_category = {}
            
            for category, job_category_skills in job_skills.items():
                missing_skills_by_category[category] = []
                resume_category_skills = skills_by_category.get(category, [])
                resume_skill_names = [skill['skill'] for skill in resume_category_skills]
                
                for job_skill in job_category_skills:
                    if job_skill['skill'] in resume_skill_names:
                        matched_skills += 1
                    else:
                        missing_skills_by_category[category].append(job_skill['skill'])
            
            # Calculate match percentage
            match_percentage = round((matched_skills / total_job_skills) * 100, 2)
            analysis_results["match_percentage"] = match_percentage
            analysis_results["missing_skills"] = missing_skills_by_category
            
            # Generate advanced improvement suggestions
            improvement_suggestions = generate_advanced_suggestions(
                skills_by_category, job_skills, missing_skills_by_category, 
                entities, readability, sentiment, semantic_similarity
            )
            analysis_results["improvement_suggestions"] = improvement_suggestions
    
    return analysis_results

def generate_advanced_suggestions(resume_skills, job_skills, missing_skills, 
                                entities, readability, sentiment, semantic_similarity):
    """Generate intelligent improvement suggestions based on advanced analysis"""
    suggestions = []
    
    # Skills-based suggestions
    if missing_skills:
        for category, skills in missing_skills.items():
            if skills:
                suggestions.append({
                    "type": "skills",
                    "priority": "high",
                    "title": f"Add Missing {category.replace('_', ' ').title()} Skills",
                    "description": f"Consider adding these {category.replace('_', ' ')} skills if you have experience:",
                    "suggestion_list": skills[:5],  # Limit to top 5
                    "impact": "Improves ATS matching and recruiter interest"
                })
    
    # Semantic similarity suggestions
    if semantic_similarity < 0.3:
        suggestions.append({
            "type": "content",
            "priority": "high",
            "title": "Improve Content Alignment",
            "description": "Your resume content doesn't closely match the job description.",
            "suggestion_list": [
                "Use more keywords from the job description",
                "Align your experience descriptions with job requirements",
                "Include industry-specific terminology"
            ],
            "impact": f"Current similarity: {semantic_similarity:.2%}"
        })
    
    # Readability suggestions
    if readability:
        flesch_score = readability.get('flesch_reading_ease', 0)
        if flesch_score < 60:  # Difficult to read
            suggestions.append({
                "type": "readability",
                "priority": "medium",
                "title": "Improve Readability",
                "description": "Your resume may be difficult to read quickly.",
                "suggestion_list": [
                    "Use shorter sentences",
                    "Simplify complex terminology",
                    "Use bullet points for easy scanning",
                    "Break up long paragraphs"
                ],
                "impact": f"Current readability score: {flesch_score:.1f}/100"
            })
    
    # Sentiment analysis suggestions
    if sentiment.get('label') == 'NEGATIVE':
        suggestions.append({
            "type": "tone",
            "priority": "medium", 
            "title": "Improve Tone and Language",
            "description": "Consider using more positive language in your resume.",
            "suggestion_list": [
                "Focus on achievements rather than responsibilities",
                "Use action verbs (achieved, led, improved, created)",
                "Highlight positive outcomes and impacts",
                "Avoid negative or passive language"
            ],
            "impact": "Creates a more compelling first impression"
        })
    
    # Entity-based suggestions
    email_found = any(ent['label'] == 'EMAIL' for ent in entities)
    phone_found = any(ent['label'] == 'PHONE' for ent in entities)
    
    if not email_found or not phone_found:
        missing_contact = []
        if not email_found:
            missing_contact.append("professional email address")
        if not phone_found:
            missing_contact.append("phone number")
            
        suggestions.append({
            "type": "contact",
            "priority": "high",
            "title": "Add Missing Contact Information",
            "description": "Essential contact information is missing.",
            "suggestion_list": missing_contact,
            "impact": "Ensures recruiters can reach you easily"
        })
    
    # Quantification suggestions
    suggestions.append({
        "type": "quantify",
        "priority": "high",
        "title": "Quantify Your Achievements",
        "description": "Add measurable impact to make your experience stand out:",
        "suggestion_list": [
            "Include percentages, numbers, dollar amounts, or time periods",
            "Example: 'Reduced processing time by 30% by implementing automated workflows'",
            "Example: 'Led a team of 5 developers to deliver project 2 weeks ahead of schedule'",
            "Example: 'Increased user engagement by 25% through UI/UX improvements'"
        ],
        "impact": "Makes achievements more credible and memorable"
    })
    
    # ATS optimization suggestions
    suggestions.append({
        "type": "ats",
        "priority": "medium",
        "title": "Optimize for ATS Systems",
        "description": "Make your resume ATS-friendly:",
        "suggestion_list": [
            "Use standard section headings (Experience, Education, Skills)",
            "Include keywords from job descriptions naturally",
            "Use standard fonts (Arial, Calibri, Times New Roman)",
            "Save in PDF format with text (not image) content",
            "Avoid tables, graphics, or unusual formatting"
        ],
        "impact": "Ensures your resume passes initial automated screening"
    })
    
    return suggestions

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'resume' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['resume']
    job_description = request.form.get('job_description', '')
    
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Extract text from PDF
        resume_text = extract_text_from_pdf(filepath)
        if not resume_text:
            flash('Error extracting text from PDF')
            return redirect(request.url)
        
        # Analyze resume
        analysis_results = analyze_resume(resume_text, job_description)
        
        return render_template('results.html', 
                              results=analysis_results, 
                              resume_text=resume_text, 
                              job_description=job_description,
                              filename=filename)
    
    flash('Invalid file type. Please upload a PDF.')
    return redirect(request.url)

@app.route('/edit-resume/<filename>')
def edit_resume(filename):
    """Render the resume editor page"""
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        flash('Resume file not found')
        return redirect(url_for('index'))
    
    # Extract text from PDF
    resume_text = extract_text_from_pdf(filepath)
    if not resume_text:
        flash('Error extracting text from PDF')
        return redirect(url_for('index'))
    
    return render_template('editor.html', 
                          filename=filename,
                          resume_text=resume_text)

@app.route('/save-resume', methods=['POST'])
def save_resume():
    """Save the edited resume text and convert back to PDF"""
    data = request.get_json()
    if not data or 'text' not in data or 'filename' not in data:
        return jsonify({'success': False, 'message': 'Invalid data'})
    
    edited_text = data['text']
    original_filename = data['filename']
    
    # Create a simple PDF with the edited text
    try:
        # Generate a temporary file path for the new PDF
        temp_file = os.path.join(app.config['UPLOAD_FOLDER'], f"edited_{original_filename}")
        
        # Creating a simple PDF from the text using reportlab
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import inch
        
        doc = SimpleDocTemplate(temp_file, pagesize=letter)
        styles = getSampleStyleSheet()
        
        # Split text into paragraphs and create PDF content
        paragraphs = []
        for line in edited_text.split('\n'):
            if line.strip():
                paragraphs.append(Paragraph(line, styles['Normal']))
            else:
                paragraphs.append(Paragraph("<br/>", styles['Normal']))
        
        doc.build(paragraphs)
        
        return jsonify({
            'success': True, 
            'message': 'Resume saved successfully',
            'filename': f"edited_{original_filename}"
        })
    
    except Exception as e:
        print(f"Error saving PDF: {e}")
        return jsonify({'success': False, 'message': f'Error saving PDF: {e}'})

@app.route('/download-resume/<filename>')
def download_resume(filename):
    """Download the edited resume PDF"""
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        flash('Resume file not found')
        return redirect(url_for('index'))
    
    return send_file(filepath, as_attachment=True, download_name=filename)

if __name__ == '__main__':
    app.run(debug=True, port=5001) 