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
    """Skills extraction with lightning-fast primary mode"""
    try:
        # PRIMARY: Use lightning-fast extractor (4ms processing time)
        from lightning_extractor import extract_skills_lightning_fast
        print("⚡ Using lightning-fast extraction mode (4ms)")
        return extract_skills_lightning_fast(text)
    except ImportError:
        print("⚠️  Lightning extractor not available, trying ultra-fast extractor")
        try:
            from ultra_fast_extractor import extract_skills_ultra_fast
            return extract_skills_ultra_fast(text)
        except ImportError:
            return extract_skills_fast_fallback(text)
    except Exception as e:
        print(f"⚠️  Lightning extraction failed: {e}, falling back")
        return extract_skills_fast_fallback(text)

def extract_skills_premium(text):
    """Premium skills extraction (advanced NLP) - slower but highest quality"""
    try:
        # For users who want maximum quality and don't mind 5-15 second wait
        from advanced_skills_extractor import extract_advanced_skills
        print("🎯 Using premium extraction mode (5-15s, highest quality)")
        return extract_advanced_skills(text)
    except ImportError:
        print("⚠️  Premium extractor not available, using ultra-fast mode")
        return extract_skills_advanced(text)
    except Exception as e:
        print(f"⚠️  Premium extraction failed: {e}, falling back to ultra-fast mode")
        return extract_skills_advanced(text)

def extract_skills_fast_fallback(text):
    """Fast skills extraction fallback"""
    try:
        from fast_skills_extractor import extract_skills_quickly
        return extract_skills_quickly(text)
    except ImportError:
        print("⚠️  Fast extractor not available, using basic fallback")
        return extract_skills_basic_fallback(text)

def extract_skills_basic_fallback(text):
    """Basic skills extraction for when all else fails"""
    # Ultra-basic skills database for ultimate fallback
    basic_skills = {
        'programming_languages': ['python', 'java', 'javascript', 'c++', 'c#', 'php', 'ruby', 'go', 'typescript'],
        'web_technologies': ['html', 'css', 'react', 'angular', 'vue', 'django', 'flask', 'spring', 'bootstrap'],
        'databases': ['sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'oracle', 'sqlite'],
        'cloud_platforms': ['aws', 'azure', 'gcp', 'kubernetes', 'docker', 'jenkins', 'terraform'],
        'ai_ml': ['machine learning', 'tensorflow', 'pytorch', 'pandas', 'numpy', 'scikit-learn', 'nlp'],
        'tools': ['git', 'github', 'jira', 'docker', 'linux', 'unix']
    }
    
    text_lower = text.lower()
    found_skills = {}
    
    for category, skills in basic_skills.items():
        for skill in skills:
            if skill.lower() in text_lower:
                if category not in found_skills:
                    found_skills[category] = []
                found_skills[category].append({
                    'skill': skill,
                    'confidence': 0.7,
                    'method': 'basic_fallback',
                    'context': '',
                    'variations_found': [skill]
                })
    
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

def analyze_resume(resume_text, job_description=None, analysis_mode='lightning'):
    """Enhanced resume analysis with user-selectable processing modes"""
    
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
        "advanced_insights": {},
        "job_match_analysis": {},
        "processing_info": {
            "analysis_mode": analysis_mode
        }
    }
    
    import time
    start_time = time.time()
    
    # Extract skills using the selected method
    print(f"🔍 Extracting skills using {analysis_mode} mode...")
    
    # FORCE ADVANCED MODE FOR DEBUGGING
    print("🎯 FORCING ADVANCED MODE FOR PROPER ATS FUNCTIONALITY")
    analysis_mode = 'advanced'
    
    if analysis_mode == 'advanced':
        # Use advanced NLP analysis (5-15 seconds)
        skills_raw = extract_skills_premium(resume_text)
    else:
        # Use lightning fast analysis (2-5ms) - default
        skills_raw = extract_skills_advanced(resume_text)
    
    skills_extraction_time = time.time() - start_time
    
    # Normalize skills format for compatibility
    if isinstance(skills_raw, dict) and 'skills_by_category' in skills_raw:
        # Lightning format - extract the skills_by_category
        skills_by_category = skills_raw['skills_by_category']
        total_skills_found = skills_raw.get('total_skills_found', 0)
    else:
        # Traditional format
        skills_by_category = skills_raw
        total_skills_found = sum(len(skills) for skills in skills_by_category.values() if isinstance(skills, list))
    
    analysis_results["skills"] = skills_by_category
    
    # Extract entities using multiple NLP techniques
    entities_start = time.time()
    entities = advanced_entity_extraction(resume_text)
    entities_time = time.time() - entities_start
    
    analysis_results["entities"] = entities
    
    # Analyze text readability (if textstat available)
    readability_start = time.time()
    readability = analyze_text_readability(resume_text)
    readability_time = time.time() - readability_start
    
    analysis_results["readability"] = readability
    
    # Analyze sentiment
    sentiment_start = time.time()
    sentiment = analyze_sentiment(resume_text)
    sentiment_time = time.time() - sentiment_start
    
    analysis_results["sentiment"] = sentiment
    
    # Advanced insights
    # Use the already calculated total_skills_found
    advanced_insights = {
        "total_entities": len(entities),
        "entity_types": list(set([ent["label"] for ent in entities])),
        "total_skills": total_skills_found,
        "skill_categories": list(skills_by_category.keys()),
        "text_length": len(resume_text),
        "word_count": len(resume_text.split()),
        "sentence_count": len(sent_tokenize(resume_text)) if resume_text else 0
    }
    analysis_results["advanced_insights"] = advanced_insights
    
    # Compare with job description if provided using mode-appropriate matching
    job_match_start = time.time()
    if job_description:
        try:
            if analysis_mode == 'advanced':
                # Use advanced NLP job matching (5-15 seconds, semantic analysis)
                print("🎯 Using advanced NLP job matching with semantic analysis")
                try:
                    from advanced_skills_extractor import extract_advanced_skills
                    
                    # Extract skills from job description using advanced analysis
                    job_skills = extract_advanced_skills(job_description)
                    
                    # Calculate semantic similarity and advanced matching
                    job_match_results = calculate_advanced_job_match_enhanced(
                        skills_by_category, job_description, job_skills
                    )
                    
                except ImportError:
                    print("⚠️ Advanced extractor not available, using lightning extractor for job skills")
                    # Use lightning extractor for job skills
                    from lightning_extractor import extract_skills_lightning_fast
                    job_skills = extract_skills_lightning_fast(job_description)
                    
                    # Use lightning job matching instead
                    from lightning_extractor import calculate_job_match_lightning_fast
                    job_match_results = calculate_job_match_lightning_fast(skills_by_category, job_description)
                
                # Map results to expected template structure  
                analysis_results["job_match_analysis"] = job_match_results
                analysis_results["match_percentage"] = job_match_results.get("overall_score", job_match_results.get("match_percentage", 0))
                analysis_results["semantic_similarity"] = job_match_results.get("semantic_similarity", 0) / 100
                
                # Extract and map all required fields for template
                missing_skills = job_match_results.get("missing_skills", {})
                found_skills = job_match_results.get("matched_skills", job_match_results.get("found_skills", {}))
                
                # Convert defaultdict objects to regular dictionaries for proper template rendering
                def convert_defaultdict_to_dict(obj):
                    """Convert defaultdict objects to regular dictionaries recursively"""
                    if hasattr(obj, 'default_factory'):  # It's a defaultdict
                        return dict(obj)
                    elif isinstance(obj, dict):
                        return {k: convert_defaultdict_to_dict(v) for k, v in obj.items()}
                    else:
                        return obj
                
                # Convert both missing_skills and found_skills
                missing_skills = convert_defaultdict_to_dict(missing_skills)
                found_skills = convert_defaultdict_to_dict(found_skills)
                
                # Ensure found_skills is always a dictionary
                if not isinstance(found_skills, dict):
                    print(f"⚠️ Warning: found_skills is {type(found_skills)}, converting to dict")
                    found_skills = {}
                
                # Ensure missing_skills is always a dictionary
                if not isinstance(missing_skills, dict):
                    print(f"⚠️ Warning: missing_skills is {type(missing_skills)}, converting to dict")
                    missing_skills = {}
                
                analysis_results["missing_skills"] = missing_skills
                analysis_results["found_skills"] = found_skills
                
                # Calculate totals for advanced mode
                if "total_job_skills" in job_match_results:
                    analysis_results["total_job_skills"] = job_match_results["total_job_skills"]
                    analysis_results["matched_skills"] = job_match_results.get("matched_skills", 0)
                else:
                    # Fallback calculation for advanced mode
                    total_job_skills = sum(len(skills) for skills in job_skills.get('skills_by_category', {}).values())
                    matched_count = 0
                    for category_found in found_skills.values():
                        matched_count += len(category_found) if isinstance(category_found, list) else 0
                    
                    analysis_results["total_job_skills"] = total_job_skills
                    analysis_results["matched_skills"] = matched_count
                
                # Generate advanced improvement suggestions
                improvement_suggestions = generate_advanced_suggestions_v2(
                    skills_by_category, missing_skills, entities, readability, 
                    sentiment, job_match_results
                )
                analysis_results["improvement_suggestions"] = improvement_suggestions
                
            else:
                # Use lightning-fast job matching (2-5ms, pattern matching)
                from lightning_extractor import calculate_job_match_lightning_fast
                print("⚡ Using lightning-fast job matching (4ms)")
                job_match_results = calculate_job_match_lightning_fast(skills_by_category, job_description)
                
                # Map results to expected template structure
                analysis_results["job_match_analysis"] = job_match_results
                analysis_results["match_percentage"] = job_match_results.get("overall_score", 0)
                analysis_results["semantic_similarity"] = job_match_results.get("semantic_similarity", 0) / 100
                
                # Extract and map all required fields for template
                missing_skills = job_match_results.get("missing_skills", {})
                found_skills = job_match_results.get("found_skills", {})
                
                # Convert defaultdict objects to regular dictionaries for proper template rendering
                def convert_defaultdict_to_dict(obj):
                    """Convert defaultdict objects to regular dictionaries recursively"""
                    if hasattr(obj, 'default_factory'):  # It's a defaultdict
                        return dict(obj)
                    elif isinstance(obj, dict):
                        return {k: convert_defaultdict_to_dict(v) for k, v in obj.items()}
                    else:
                        return obj
                
                # Convert both missing_skills and found_skills
                missing_skills = convert_defaultdict_to_dict(missing_skills)
                found_skills = convert_defaultdict_to_dict(found_skills)
                
                # Ensure data types are correct
                if not isinstance(missing_skills, dict):
                    print(f"⚠️ Warning: missing_skills is {type(missing_skills)}, converting to dict")
                    missing_skills = {}
                    
                if not isinstance(found_skills, dict):
                    print(f"⚠️ Warning: found_skills is {type(found_skills)}, converting to dict")
                    found_skills = {}
                
                analysis_results["missing_skills"] = missing_skills
                analysis_results["found_skills"] = found_skills
                analysis_results["total_job_skills"] = job_match_results.get("total_job_skills", 0)
                analysis_results["matched_skills"] = job_match_results.get("matched_skills", 0)
                
                # Generate basic improvement suggestions (fast)
                improvement_suggestions = generate_lightning_suggestions(
                    skills_by_category, missing_skills, entities, job_match_results
                )
                analysis_results["improvement_suggestions"] = improvement_suggestions
            
        except ImportError:
            # Fallback chain
            print(f"⚠️ Preferred {analysis_mode} job matching not available, using fallback")
            try:
                from lightning_extractor import calculate_job_match_lightning_fast
                job_match_results = calculate_job_match_lightning_fast(skills_by_category, job_description)
                
                # Map results properly for template
                analysis_results["job_match_analysis"] = job_match_results
                analysis_results["match_percentage"] = job_match_results.get("overall_score", 0)
                missing_skills = job_match_results.get("missing_skills", {})
                found_skills = job_match_results.get("found_skills", {})
                
                # Convert defaultdict objects to regular dictionaries for proper template rendering
                def convert_defaultdict_to_dict(obj):
                    """Convert defaultdict objects to regular dictionaries recursively"""
                    if hasattr(obj, 'default_factory'):  # It's a defaultdict
                        return dict(obj)
                    elif isinstance(obj, dict):
                        return {k: convert_defaultdict_to_dict(v) for k, v in obj.items()}
                    else:
                        return obj
                
                # Convert both missing_skills and found_skills
                missing_skills = convert_defaultdict_to_dict(missing_skills)
                found_skills = convert_defaultdict_to_dict(found_skills)
                
                # Ensure data types are correct
                if not isinstance(missing_skills, dict):
                    missing_skills = {}
                if not isinstance(found_skills, dict):
                    found_skills = {}
                
                analysis_results["missing_skills"] = missing_skills
                analysis_results["found_skills"] = found_skills
                analysis_results["total_job_skills"] = job_match_results.get("total_job_skills", 0)
                analysis_results["matched_skills"] = job_match_results.get("matched_skills", 0)
                
            except ImportError:
                # Ultimate fallback
                basic_match_results = calculate_basic_job_match(skills_by_category, job_description)
                analysis_results.update(basic_match_results)
                
                # Convert defaultdict objects to regular dictionaries for proper template rendering
                def convert_defaultdict_to_dict(obj):
                    """Convert defaultdict objects to regular dictionaries recursively"""
                    if hasattr(obj, 'default_factory'):  # It's a defaultdict
                        return dict(obj)
                    elif isinstance(obj, dict):
                        return {k: convert_defaultdict_to_dict(v) for k, v in obj.items()}
                    else:
                        return obj
                
                # Ensure fallback also has proper structure
                if "missing_skills" not in analysis_results:
                    analysis_results["missing_skills"] = {}
                if "found_skills" not in analysis_results:
                    analysis_results["found_skills"] = basic_match_results.get("found_skills", {})
                if "total_job_skills" not in analysis_results:
                    analysis_results["total_job_skills"] = basic_match_results.get("total_job_skills", 0)
                if "matched_skills" not in analysis_results:
                    analysis_results["matched_skills"] = basic_match_results.get("matched_skills", 0)
                
                # Convert defaultdict objects
                analysis_results["missing_skills"] = convert_defaultdict_to_dict(analysis_results.get("missing_skills", {}))
                analysis_results["found_skills"] = convert_defaultdict_to_dict(analysis_results.get("found_skills", {}))
                    
                # Double check data types for fallback
                if not isinstance(analysis_results.get("found_skills"), dict):
                    analysis_results["found_skills"] = {}
                if not isinstance(analysis_results.get("missing_skills"), dict):
                    analysis_results["missing_skills"] = {}
                
        except Exception as e:
            print(f"⚠️ Job matching failed: {e}, using basic matching")
            basic_match_results = calculate_basic_job_match(skills_by_category, job_description)
            analysis_results.update(basic_match_results)
    
    job_match_time = time.time() - job_match_start
    total_time = time.time() - start_time
    
    # Add processing information
    analysis_results["processing_info"] = {
        "total_time": round(total_time, 3),
        "skills_extraction_time": round(skills_extraction_time, 3),
        "entities_time": round(entities_time, 3),
        "readability_time": round(readability_time, 3),
        "sentiment_time": round(sentiment_time, 3),
        "job_match_time": round(job_match_time, 3),
        "analysis_mode": analysis_mode,
        "mode": analysis_mode  # Keep for backward compatibility
    }
    
    print(f"✅ Analysis completed in {total_time:.2f} seconds")
    
    # Debug output to check data structure
    print(f"🔍 Debug - Final analysis results keys: {list(analysis_results.keys())}")
    print(f"🔍 Debug - Match percentage: {analysis_results.get('match_percentage', 'MISSING')}")
    print(f"🔍 Debug - Total job skills: {analysis_results.get('total_job_skills', 'MISSING')}")
    print(f"🔍 Debug - Matched skills: {analysis_results.get('matched_skills', 'MISSING')}")
    print(f"🔍 Debug - Missing skills keys: {list(analysis_results.get('missing_skills', {}).keys())}")
    
    # Safe debug for found_skills
    found_skills = analysis_results.get('found_skills', {})
    if isinstance(found_skills, dict):
        print(f"🔍 Debug - Found skills keys: {list(found_skills.keys())}")
    else:
        print(f"🔍 Debug - Found skills type issue: {type(found_skills)} = {found_skills}")
        # Fix the found_skills if it's not a dict
        analysis_results['found_skills'] = {}
    
    # Debug skills extraction
    print(f"🔍 Debug - Resume skills categories: {list(analysis_results.get('skills', {}).get('skills_by_category', {}).keys())}")
    for category, skills in analysis_results.get('skills', {}).get('skills_by_category', {}).items():
        skill_names = [skill['skill'] for skill in skills[:3]]  # First 3 skills per category
        print(f"🔍 Debug - Resume {category}: {skill_names}")
    
    return analysis_results

def calculate_advanced_job_match_enhanced(skills_by_category, job_description, job_skills):
    """Enhanced job matching using advanced NLP analysis with ATS checker"""
    try:
        # Use the new advanced ATS checker
        from advanced_skills_extractor import calculate_ats_score
        
        # Format skills for ATS checker
        formatted_skills = {}
        for category, skills in skills_by_category.items():
            formatted_skills[category] = []
            for skill in skills:
                if isinstance(skill, dict):
                    formatted_skills[category].append(skill)
                else:
                    # Convert string to dict format
                    formatted_skills[category].append({
                        'skill': str(skill),
                        'confidence': 0.8
                    })
        
        # Calculate ATS score
        ats_results = calculate_ats_score(formatted_skills, job_description)
        
        # Format results for compatibility with existing template
        results = {
            'overall_score': ats_results.get('overall_score', 0),
            'semantic_similarity': ats_results.get('semantic_similarity', 0),
            'category_scores': ats_results.get('category_scores', {}),
            'matched_skills': ats_results.get('matched_skills', {}),
            'missing_skills': ats_results.get('missing_skills', {}),
            'found_skills': ats_results.get('matched_skills', {}),  # Alias for template compatibility
            'extra_skills': ats_results.get('extra_skills', {}),
            'total_job_skills': sum(len(skills) for skills in ats_results.get('missing_skills', {}).values()) + 
                               sum(len(skills) for skills in ats_results.get('matched_skills', {}).values()),
            'matched_skills_count': sum(len(skills) for skills in ats_results.get('matched_skills', {}).values()),
            'advanced_analysis': True
        }
        
        print(f"✅ Advanced ATS analysis complete - Score: {results['overall_score']}%")
        return results
        
    except Exception as e:
        print(f"⚠️ Error in advanced job matching: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback to basic matching
        return {
            'overall_score': 0,
            'semantic_similarity': 0,
            'category_scores': {},
            'matched_skills': {},
            'missing_skills': {},
            'found_skills': {},
            'extra_skills': {},
            'total_job_skills': 0,
            'matched_skills_count': 0,
            'error': str(e)
        }


def calculate_basic_job_match(skills_by_category, job_description):
    """Basic job matching for fallback"""
    try:
        from lightning_extractor import calculate_job_match_lightning_fast
        return calculate_job_match_lightning_fast(skills_by_category, job_description)
    except ImportError:
        # Ultimate fallback with simple keyword matching
        job_words = set(job_description.lower().split())
        resume_words = set()
        
        for category, skills in skills_by_category.items():
            for skill in skills:
                if isinstance(skill, dict):
                    resume_words.update(skill['skill'].lower().split())
        
        matched_words = job_words.intersection(resume_words)
        match_percentage = (len(matched_words) / max(len(job_words), 1)) * 100
        
        return {
            'match_percentage': round(match_percentage, 1),
            'overall_score': round(match_percentage, 1),
            'missing_skills': {},
            'found_skills': {},
            'total_job_skills': len(job_words),
            'matched_skills': len(matched_words),
            'semantic_similarity': 50,  # Default moderate similarity
        }


def generate_lightning_suggestions(resume_skills, missing_skills, entities, job_match_results):
    """Generate fast improvement suggestions for lightning mode"""
    suggestions = []
    
    # Quick skills-based suggestions
    if missing_skills:
        high_priority_skills = []
        for category, skills_list in missing_skills.items():
            if skills_list:
                # Take top 3 skills per category for speed
                category_skills = skills_list[:3] if isinstance(skills_list, list) else [skills_list]
                high_priority_skills.extend(category_skills)
        
        if high_priority_skills:
            suggestions.append({
                "type": "skills",
                "priority": "high",
                "title": "Add Missing Skills",
                "description": "Focus on these key skills mentioned in the job description:",
                "suggestion_list": high_priority_skills[:5],  # Top 5 for speed
                "impact": "Improves job match score"
            })
    
    # Quick match score feedback
    overall_score = job_match_results.get("overall_score", 0)
    if overall_score < 60:
        suggestions.append({
            "type": "match",
            "priority": "high", 
            "title": "Improve Job Match",
            "description": f"Current match: {overall_score}%. Target: 70%+",
            "suggestion_list": [
                "Add more relevant keywords from job description",
                "Quantify achievements with numbers",
                "Use action verbs (developed, implemented, led)",
                "Include relevant project examples"
            ],
            "impact": "Better ATS ranking and recruiter interest"
        })
    
    # Quick contact check
    email_found = any(ent.get('label') == 'EMAIL' for ent in entities)
    if not email_found:
        suggestions.append({
            "type": "contact",
            "priority": "high",
            "title": "Add Contact Information", 
            "description": "Ensure your email is clearly visible",
            "suggestion_list": ["Add professional email address at the top"],
            "impact": "Essential for recruiter contact"
        })
    
    return suggestions

def generate_advanced_suggestions_v2(resume_skills, missing_skills, entities, 
                                   readability, sentiment, job_match_results):
    """Generate intelligent improvement suggestions based on advanced analysis"""
    suggestions = []
    
    # Skills-based suggestions from advanced analysis
    if missing_skills:
        for category, skills_list in missing_skills.items():
            if skills_list:
                # Sort by importance if available
                if isinstance(skills_list[0], dict):
                    skills_list = sorted(skills_list, key=lambda x: x.get('importance', 0), reverse=True)
                    skill_names = [skill['skill'] for skill in skills_list[:5]]
                else:
                    skill_names = skills_list[:5]
                
                suggestions.append({
                    "type": "skills",
                    "priority": "high",
                    "title": f"Add Missing {category.replace('_', ' ').title()} Skills",
                    "description": f"Consider adding these {category.replace('_', ' ')} skills if you have experience:",
                    "suggestion_list": skill_names,
                    "impact": "Improves ATS matching and recruiter interest"
                })
    
    # Advanced job match suggestions
    overall_score = job_match_results.get("overall_score", 0)
    semantic_similarity = job_match_results.get("semantic_similarity", 0)
    
    if overall_score < 60:
        suggestions.append({
            "type": "overall_match",
            "priority": "high",
            "title": "Improve Overall Job Match",
            "description": f"Your resume matches {overall_score}% of job requirements.",
            "suggestion_list": [
                "Focus on developing the most important missing skills",
                "Use more specific technical terminology",
                "Quantify your achievements with numbers and metrics",
                "Align your experience descriptions with job requirements"
            ],
            "impact": f"Target: Increase match score to 70%+"
        })
    
    if semantic_similarity < 50:
        suggestions.append({
            "type": "semantic_alignment",
            "priority": "high",
            "title": "Improve Language Alignment",
            "description": f"Your resume language similarity is {semantic_similarity}%.",
            "suggestion_list": [
                "Use keywords and phrases from the job description",
                "Mirror the job posting's technical language",
                "Include industry-specific terminology",
                "Describe your experience using similar context"
            ],
            "impact": "Better ATS parsing and recruiter recognition"
        })
    
    # Category-specific insights
    category_scores = job_match_results.get("category_scores", {})
    weak_categories = [cat for cat, score in category_scores.items() if score < 40]
    
    if weak_categories:
        suggestions.append({
            "type": "category_focus",
            "priority": "medium",
            "title": "Strengthen Weak Skill Areas",
            "description": "Focus development on these skill categories:",
            "suggestion_list": [cat.replace('_', ' ').title() for cat in weak_categories],
            "impact": "Balanced skill profile for better job matching"
        })
    
    # Continue with existing suggestions for readability, sentiment, etc.
    if readability:
        flesch_score = readability.get('flesch_reading_ease', 0)
        if flesch_score < 60:
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
    
    return suggestions

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
    analysis_mode = request.form.get('analysis_mode', 'lightning')  # Default to lightning
    
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
        
        # Analyze resume with chosen mode
        analysis_results = analyze_resume(resume_text, job_description, analysis_mode)
        
        return render_template('results.html', 
                              results=analysis_results, 
                              resume_text=resume_text, 
                              job_description=job_description,
                              filename=filename,
                              analysis_mode=analysis_mode)
    
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