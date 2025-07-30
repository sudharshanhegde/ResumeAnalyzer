"""
Advanced ATS (Applicant Tracking System) Checker
Uses NLTK, spaCy, and advanced NLP for accurate resume-job matching
"""

import re
import spacy
import nltk
from collections import defaultdict, Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import json

class AdvancedATSChecker:
    def __init__(self):
        """Initialize advanced NLP models and skill databases"""
        print("🚀 Initializing Advanced ATS Checker...")
        
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
            print("✅ spaCy model loaded")
        except OSError:
            print("⚠️ spaCy model not found, install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Load sentence transformer for semantic similarity
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Sentence transformer loaded")
        except Exception as e:
            print(f"⚠️ Sentence transformer error: {e}")
            self.sentence_model = None
        
        # Comprehensive skill database with categories
        self.skill_database = {
            'programming_languages': {
                'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'php', 'ruby', 'go', 'rust',
                'scala', 'swift', 'kotlin', 'dart', 'r', 'matlab', 'perl', 'c', 'assembly', 'vb.net',
                'cobol', 'fortran', 'pascal', 'objective-c', 'shell', 'bash', 'powershell'
            },
            'web_technologies': {
                'html', 'css', 'react', 'angular', 'vue', 'svelte', 'jquery', 'bootstrap', 'tailwind',
                'sass', 'scss', 'less', 'webpack', 'gulp', 'grunt', 'babel', 'ajax', 'json', 'xml',
                'rest', 'api', 'graphql', 'websockets', 'pwa', 'spa', 'ssr', 'jamstack', 'nextjs', 'nuxtjs', 'gatsby'
            },
            'frameworks': {
                'django', 'flask', 'fastapi', 'spring', 'express', 'nestjs', 'laravel', 'symfony',
                'rails', 'asp.net', 'blazor', 'unity', 'xamarin', 'flutter', 'react native', 'ionic', 'cordova'
            },
            'databases': {
                'mysql', 'postgresql', 'mongodb', 'redis', 'sqlite', 'oracle', 'sql server', 'mariadb',
                'cassandra', 'elasticsearch', 'dynamodb', 'neo4j', 'influxdb', 'couchdb', 'firebase', 'supabase', 'sql'
            },
            'cloud_platforms': {
                'aws', 'azure', 'gcp', 'google cloud', 'heroku', 'digitalocean', 'linode', 'vultr',
                'cloudflare', 'vercel', 'netlify', 'railway', 'amazon web services', 'microsoft azure'
            },
            'devops_tools': {
                'docker', 'kubernetes', 'jenkins', 'gitlab ci', 'github actions', 'terraform', 'ansible',
                'puppet', 'chef', 'vagrant', 'nginx', 'apache', 'linux', 'ubuntu', 'centos', 'k8s', 'ci/cd'
            },
            'data_science': {
                'pandas', 'numpy', 'matplotlib', 'seaborn', 'plotly', 'scikit-learn', 'tensorflow',
                'pytorch', 'keras', 'opencv', 'nltk', 'spacy', 'jupyter', 'anaconda', 'hadoop', 'spark',
                'kafka', 'airflow', 'machine learning', 'ml', 'ai', 'artificial intelligence', 'deep learning'
            },
            'tools': {
                'git', 'github', 'gitlab', 'bitbucket', 'jira', 'confluence', 'slack', 'discord', 'teams',
                'zoom', 'figma', 'sketch', 'photoshop', 'illustrator', 'canva', 'notion', 'trello', 'asana'
            },
            'soft_skills': {
                'leadership', 'communication', 'teamwork', 'problem solving', 'critical thinking',
                'project management', 'agile', 'scrum', 'time management', 'collaboration', 'analytical',
                'creative', 'adaptable', 'detail oriented', 'organized', 'multitasking'
            }
        }
        
        # Create reverse mapping for entity recognition
        self.skill_to_category = {}
        for category, skills in self.skill_database.items():
            for skill in skills:
                self.skill_to_category[skill.lower()] = category
        
        print("✅ Advanced ATS Checker initialized")

    def extract_entities_advanced(self, text):
        """Extract entities using spaCy NER"""
        entities = {
            'PERSON': [],
            'ORG': [],
            'GPE': [],
            'WORK_OF_ART': [],
            'PRODUCT': [],
            'EVENT': [],
            'DATE': [],
            'MONEY': [],
            'PERCENT': []
        }
        
        if not self.nlp:
            return entities
        
        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ in entities:
                entities[ent.label_].append({
                    'text': ent.text,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'confidence': 0.9  # spaCy doesn't provide confidence, using high default
                })
        
        return entities

    def extract_skills_with_context(self, text):
        """Extract skills with context using advanced NLP"""
        skills_by_category = defaultdict(list)
        text_lower = text.lower()
        
        if self.nlp:
            doc = self.nlp(text)
            sentences = [sent.text for sent in doc.sents]
        else:
            sentences = nltk.sent_tokenize(text)
        
        # Define ambiguous/short skills that need context validation
        ambiguous_skills = {
            'r': ['programming', 'language', 'statistical', 'analytics', 'data', 'rstudio', 'cran'],
            'c': ['programming', 'language', 'c++', 'coding', 'development', 'compiler'],
            'go': ['golang', 'programming', 'language', 'google', 'backend', 'development'],
            'ml': ['machine learning', 'artificial intelligence', 'ai', 'model', 'algorithm'],
            'ai': ['artificial intelligence', 'machine learning', 'ml', 'neural', 'deep learning'],
            'd': ['programming', 'language', 'dlang', 'systems']
        }
        
        # Extract skills with context
        for category, skills in self.skill_database.items():
            for skill in skills:
                skill_lower = skill.lower()
                
                # Check if this is an ambiguous skill that needs context validation
                is_ambiguous = skill_lower in ambiguous_skills or len(skill_lower) <= 2
                
                # Use more precise patterns for short/ambiguous skills
                if is_ambiguous:
                    # For ambiguous skills, use exact word boundary and validate context
                    pattern = rf'\b{re.escape(skill_lower)}\b'
                    matches = list(re.finditer(pattern, text_lower))
                    
                    valid_contexts = []
                    for match in matches:
                        start_pos = match.start()
                        end_pos = match.end()
                        
                        # Get surrounding context (50 chars before and after)
                        context_start = max(0, start_pos - 50)
                        context_end = min(len(text_lower), end_pos + 50)
                        surrounding_context = text_lower[context_start:context_end]
                        
                        # Find the sentence containing this match
                        match_sentence = None
                        for sentence in sentences:
                            sentence_lower = sentence.lower()
                            sentence_start = text_lower.find(sentence_lower)
                            sentence_end = sentence_start + len(sentence_lower)
                            if sentence_start <= start_pos < sentence_end:
                                match_sentence = sentence.strip()
                                break
                        
                        # Validate context for ambiguous skills
                        if self._is_valid_skill_context(skill_lower, surrounding_context, match_sentence, ambiguous_skills):
                            valid_contexts.append(match_sentence if match_sentence else surrounding_context)
                    
                    if valid_contexts:
                        confidence = self._calculate_skill_confidence(skill, valid_contexts, text_lower)
                        # Higher threshold for ambiguous skills
                        if confidence >= 0.6:
                            skills_by_category[category].append({
                                'skill': skill.title(),
                                'confidence': confidence,
                                'context': valid_contexts[:2],
                                'frequency': len(valid_contexts)
                            })
                else:
                    # For clear, unambiguous skills, use the original approach
                    patterns = [
                        rf'\b{re.escape(skill_lower)}\b',
                        rf'\b{re.escape(skill_lower.replace(" ", ""))}\b',
                        rf'\b{re.escape(skill_lower.replace(".", ""))}\b'
                    ]
                    
                    for pattern in patterns:
                        matches = list(re.finditer(pattern, text_lower))
                        if matches:
                            contexts = []
                            for sentence in sentences:
                                if skill_lower in sentence.lower():
                                    contexts.append(sentence.strip())
                            
                            if contexts:
                                confidence = self._calculate_skill_confidence(skill, contexts, text_lower)
                                skills_by_category[category].append({
                                    'skill': skill.title(),
                                    'confidence': confidence,
                                    'context': contexts[:2],
                                    'frequency': len(matches)
                                })
                            break
        
        return dict(skills_by_category)

    def _is_valid_skill_context(self, skill_lower, surrounding_context, sentence_context, ambiguous_skills):
        """Validate if the skill mention is in a proper technical context"""
        if skill_lower not in ambiguous_skills:
            return True
        
        # Check both surrounding context and sentence context
        contexts_to_check = [surrounding_context]
        if sentence_context:
            contexts_to_check.append(sentence_context.lower())
        
        required_keywords = ambiguous_skills[skill_lower]
        
        for context in contexts_to_check:
            if context:
                # Check if any of the required keywords are present
                if any(keyword in context for keyword in required_keywords):
                    return True
                
                # Additional context checks for specific skills
                if skill_lower == 'r':
                    # Check for R programming specific patterns
                    if any(pattern in context for pattern in ['r studio', 'r programming', 'r language', 'cran', 'tidyverse', 'ggplot']):
                        return True
                elif skill_lower == 'c':
                    # Check for C programming specific patterns
                    if any(pattern in context for pattern in ['c programming', 'c language', 'c++', 'gcc', 'coding in c']):
                        return True
                elif skill_lower == 'go':
                    # Check for Go programming specific patterns
                    if any(pattern in context for pattern in ['golang', 'go programming', 'go language', 'google go']):
                        return True
        
        return False

    def _calculate_skill_confidence(self, skill, contexts, full_text):
        """Calculate confidence score for a skill based on context"""
        base_confidence = 0.7
        
        # Frequency bonus
        frequency = full_text.count(skill.lower())
        frequency_bonus = min(frequency * 0.1, 0.2)
        
        # Context quality bonus
        context_bonus = 0
        for context in contexts:
            context_lower = context.lower()
            # Check for experience indicators
            if any(word in context_lower for word in ['experience', 'years', 'expert', 'proficient', 'skilled']):
                context_bonus += 0.1
            # Check for project indicators
            if any(word in context_lower for word in ['project', 'developed', 'built', 'created', 'implemented']):
                context_bonus += 0.05
        
        return min(base_confidence + frequency_bonus + context_bonus, 0.95)

    def calculate_semantic_similarity(self, text1, text2):
        """Calculate semantic similarity using sentence transformers"""
        if not self.sentence_model:
            # Fallback to simple TF-IDF
            vectorizer = TfidfVectorizer().fit([text1, text2])
            vectors = vectorizer.transform([text1, text2])
            return cosine_similarity(vectors[0], vectors[1])[0][0]
        
        try:
            embeddings = self.sentence_model.encode([text1, text2])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(similarity)
        except Exception as e:
            print(f"⚠️ Semantic similarity error: {e}")
            return 0.5

    def analyze_job_requirements(self, job_description):
        """Analyze job description to extract requirements"""
        # Extract skills from job description
        job_skills = self.extract_skills_with_context(job_description)
        
        # Extract entities (companies, technologies, etc.)
        job_entities = self.extract_entities_advanced(job_description)
        
        # Identify required vs preferred skills
        required_skills = defaultdict(list)
        preferred_skills = defaultdict(list)
        
        job_lower = job_description.lower()
        for category, skills in job_skills.items():
            for skill_obj in skills:
                skill_name = skill_obj['skill'].lower()
                
                # Check if skill appears in required context
                required_indicators = ['required', 'must have', 'essential', 'mandatory', 'needed']
                preferred_indicators = ['preferred', 'nice to have', 'bonus', 'plus', 'advantage']
                
                is_required = any(indicator in job_lower and skill_name in job_lower for indicator in required_indicators)
                is_preferred = any(indicator in job_lower and skill_name in job_lower for indicator in preferred_indicators)
                
                if is_required:
                    required_skills[category].append(skill_obj)
                elif is_preferred:
                    preferred_skills[category].append(skill_obj)
                else:
                    # Default to required if not specified
                    required_skills[category].append(skill_obj)
        
        return {
            'skills_by_category': job_skills,
            'required_skills': dict(required_skills),
            'preferred_skills': dict(preferred_skills),
            'entities': job_entities,
            'total_skills': sum(len(skills) for skills in job_skills.values())
        }

    def compare_resume_with_job(self, resume_skills, job_analysis):
        """Compare resume skills with job requirements using advanced matching"""
        results = {
            'overall_score': 0,
            'category_scores': {},
            'matched_skills': defaultdict(list),
            'missing_skills': defaultdict(list),
            'extra_skills': defaultdict(list),
            'semantic_similarity': 0,
            'detailed_analysis': {}
        }
        
        job_skills = job_analysis['skills_by_category']
        required_skills = job_analysis['required_skills']
        
        total_required_skills = 0
        total_matched_skills = 0
        
        # Compare each category
        for category in set(list(resume_skills.keys()) + list(job_skills.keys())):
            resume_category_skills = resume_skills.get(category, [])
            job_category_skills = job_skills.get(category, [])
            required_category_skills = required_skills.get(category, [])
            
            # Create skill name sets for comparison
            resume_skill_names = {skill['skill'].lower() for skill in resume_category_skills}
            job_skill_names = {skill['skill'].lower() for skill in job_category_skills}
            required_skill_names = {skill['skill'].lower() for skill in required_category_skills}
            
            # Find matches using fuzzy matching
            category_matched = []
            category_missing = []
            
            for job_skill_obj in job_category_skills:
                job_skill_name = job_skill_obj['skill'].lower()
                matched = False
                
                # Direct match
                if job_skill_name in resume_skill_names:
                    category_matched.append(job_skill_obj['skill'])
                    matched = True
                else:
                    # Fuzzy matching
                    for resume_skill_name in resume_skill_names:
                        if self._skills_match(job_skill_name, resume_skill_name):
                            category_matched.append(job_skill_obj['skill'])
                            matched = True
                            break
                
                if not matched:
                    category_missing.append(job_skill_obj['skill'])
            
            # Calculate category score
            category_total = len(job_category_skills)
            category_matched_count = len(category_matched)
            
            if category_total > 0:
                category_score = (category_matched_count / category_total) * 100
                results['category_scores'][category] = round(category_score, 1)
                
                # Weight required skills more heavily
                if category in required_skills:
                    total_required_skills += len(required_category_skills)
                    matched_required = len([skill for skill in category_matched 
                                         if skill.lower() in required_skill_names])
                    total_matched_skills += matched_required
                else:
                    total_required_skills += len(job_category_skills)
                    total_matched_skills += category_matched_count
            
            results['matched_skills'][category] = category_matched
            results['missing_skills'][category] = category_missing
            
            # Find extra skills (in resume but not in job)
            extra_skills = []
            for resume_skill_obj in resume_category_skills:
                resume_skill_name = resume_skill_obj['skill'].lower()
                if resume_skill_name not in job_skill_names:
                    extra_skills.append(resume_skill_obj['skill'])
            results['extra_skills'][category] = extra_skills
        
        # Calculate overall score
        if total_required_skills > 0:
            results['overall_score'] = round((total_matched_skills / total_required_skills) * 100, 1)
        
        # Add the matched skills count for template rendering
        results['matched_skills_count'] = total_matched_skills
        results['total_job_skills'] = total_required_skills
        
        # Convert defaultdict objects to regular dictionaries for proper template rendering
        results['matched_skills'] = dict(results['matched_skills'])
        results['missing_skills'] = dict(results['missing_skills'])
        results['extra_skills'] = dict(results['extra_skills'])
        
        return results

    def _skills_match(self, skill1, skill2):
        """Advanced skill matching with synonyms and variations"""
        skill1 = skill1.lower().strip()
        skill2 = skill2.lower().strip()
        
        # Direct match
        if skill1 == skill2:
            return True
        
        # Common variations
        variations = {
            'javascript': ['js', 'java script'],
            'typescript': ['ts', 'type script'],
            'node.js': ['nodejs', 'node'],
            'react.js': ['react', 'reactjs'],
            'vue.js': ['vue', 'vuejs'],
            'angular.js': ['angular', 'angularjs'],
            'c++': ['cpp', 'c plus plus'],
            'c#': ['csharp', 'c sharp'],
            'machine learning': ['ml'],
            'artificial intelligence': ['ai'],
            'sql': ['mysql', 'postgresql', 'sqlite'],
            'docker': ['containerization'],
            'kubernetes': ['k8s'],
            'amazon web services': ['aws'],
            'google cloud platform': ['gcp', 'google cloud']
        }
        
        # Check variations
        for main_skill, vars in variations.items():
            if (skill1 == main_skill and skill2 in vars) or (skill2 == main_skill and skill1 in vars):
                return True
            if skill1 in vars and skill2 in vars:
                return True
        
        # Partial matching (one contains the other)
        if skill1 in skill2 or skill2 in skill1:
            return True
        
        return False

# Global instance
ats_checker = AdvancedATSChecker()

def extract_advanced_skills(text):
    """Extract skills using advanced NLP methods"""
    skills_by_category = ats_checker.extract_skills_with_context(text)
    entities = ats_checker.extract_entities_advanced(text)
    
    total_skills = sum(len(skills) for skills in skills_by_category.values())
    
    return {
        'skills_by_category': skills_by_category,
        'entities': entities,
        'total_skills': total_skills,
        'advanced_insights': {
            'entity_count': sum(len(ents) for ents in entities.values()),
            'categories_found': len(skills_by_category),
            'avg_confidence': np.mean([skill['confidence'] for skills in skills_by_category.values() for skill in skills]) if total_skills > 0 else 0
        }
    }

def calculate_semantic_similarity(text1, text2):
    """Calculate semantic similarity between two texts"""
    return ats_checker.calculate_semantic_similarity(text1, text2)

def analyze_job_description(job_description):
    """Analyze job description to extract requirements"""
    return ats_checker.analyze_job_requirements(job_description)

def calculate_ats_score(resume_skills, job_description):
    """Calculate ATS score comparing resume with job description"""
    # Analyze job requirements
    job_analysis = analyze_job_description(job_description)
    
    # Compare resume with job
    comparison_results = ats_checker.compare_resume_with_job(resume_skills, job_analysis)
    
    # Calculate semantic similarity
    resume_text = ' '.join([skill['skill'] for skills in resume_skills.values() for skill in skills])
    semantic_sim = calculate_semantic_similarity(resume_text, job_description)
    comparison_results['semantic_similarity'] = round(semantic_sim * 100, 1)
    
    return comparison_results

print("✅ Advanced ATS Checker module loaded")
