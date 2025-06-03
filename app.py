import streamlit as st

# --- UI Configuration (MUST BE FIRST STREAMLIT COMMAND) ---
st.set_page_config(layout="wide", page_title="AI Resume Analyzer")

import pandas as pd
import PyPDF2
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import spacy
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import io

# --- NLTK and spaCy Setup ---
@st.cache_resource
def load_nlp_resources():
    """Downloads NLTK data and loads spaCy model."""
    # NLTK downloads - Includes punkt_tab
    packages = [
        ("corpora/wordnet", "wordnet"),
        ("corpora/stopwords", "stopwords"),
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab") # Ensure this is downloaded
    ]
    for path, pkg_id in packages:
        try:
            nltk.data.find(path)
        except LookupError: # Use LookupError for missing NLTK data
            print(f"NLTK data package not found: {pkg_id}. Downloading...")
            nltk.download(pkg_id)
            print(f"NLTK data package {pkg_id} downloaded.")

    # spaCy model download/load
    NLP_MODEL = "en_core_web_lg" # Using a larger model for better vectors
    try:
        nlp = spacy.load(NLP_MODEL)
    except OSError:
        print(f"Downloading spaCy model: {NLP_MODEL}")
        # Use print for feedback during initial load
        print(f"Downloading required NLP model ({NLP_MODEL})... This may take a few minutes.")
        spacy.cli.download(NLP_MODEL)
        nlp = spacy.load(NLP_MODEL)
        print(f"NLP model ({NLP_MODEL}) downloaded and loaded successfully.")
    return nlp, set(stopwords.words("english")), WordNetLemmatizer()

# Load resources globally *after* set_page_config
nlp, stop_words, lem = load_nlp_resources()

# --- Backend Functions ---

def extract_text_from_pdf(uploaded_file):
    """Extracts text from an uploaded PDF file."""
    text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.getvalue()))
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    except PyPDF2.errors.PdfReadError:
        st.error(f"Error reading PDF: {uploaded_file.name}. File might be corrupted or password-protected.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while processing {uploaded_file.name}: {e}")
        return None
    if not text:
        st.warning(f"Could not extract text from {uploaded_file.name}. The PDF might contain only images or have unusual formatting.")
        return None
    return text

# Updated preprocess to accept stop_words and lem
def preprocess(text, stop_words_arg, lem_arg):
    """Cleans and preprocesses text using NLTK."""
    if not text:
        return ""
    text = re.sub(r'\W+', ' ', text.lower())
    tokens = word_tokenize(text)
    # Use passed arguments
    processed_tokens = [lem_arg.lemmatize(t) for t in tokens if t not in stop_words_arg and len(t) > 1]
    return ' '.join(processed_tokens)

# Updated similarity to accept nlp object
@st.cache_data
def calculate_similarity(_nlp, jd_text, resume_text):
    """Calculates cosine similarity between JD and resume using spaCy vectors."""
    if not jd_text or not resume_text:
        return 0.0
    jd_doc = _nlp(jd_text)
    resume_doc = _nlp(resume_text)
    if not jd_doc.has_vector or not resume_doc.has_vector or not jd_doc.vector_norm or not resume_doc.vector_norm:
        return 0.0 # Return 0 if vectors can't be generated
    similarity = cosine_similarity([jd_doc.vector], [resume_doc.vector])[0][0]
    return max(0.0, min(1.0, similarity)) * 100

# Updated skill extraction to accept nlp object
@st.cache_data
def extract_skills(_nlp, text):
    """Extracts potential skills (noun chunks) from text using spaCy."""
    if not text:
        return set()
    doc = _nlp(text)
    # Use the globally loaded stop_words here for filtering noun chunks
    skills = {chunk.text.lower() for chunk in doc.noun_chunks if len(chunk.text.split()) <= 3}
    skills = {s for s in skills if len(s) > 2 and not all(token in stop_words for token in s.split())}
    return skills

# Updated skill analysis to accept nlp object
@st.cache_data
def analyze_skills(_nlp, jd_processed, resume_processed):
    """Performs skill gap analysis."""
    jd_skills = extract_skills(_nlp, jd_processed)
    resume_skills = extract_skills(_nlp, resume_processed)
    if not jd_skills:
        return [], [], "Could not extract skills from Job Description."
    strengths = list(jd_skills.intersection(resume_skills))
    missing_skills = list(jd_skills.difference(resume_skills))
    feedback = f"Candidate shows strength in {len(strengths)} key areas. "
    if missing_skills:
        feedback += f"Potential gaps identified in {len(missing_skills)} areas like: {', '.join(missing_skills[:3])}..."
    else:
        feedback += "Covers all key skill areas identified."
    MAX_SKILLS_DISPLAY = 10
    return strengths[:MAX_SKILLS_DISPLAY], missing_skills[:MAX_SKILLS_DISPLAY], feedback

# --- Helper Function for Dashboard ---
@st.cache_data
def convert_df_to_csv(df):
    """Converts DataFrame to CSV string for download."""
    return df.to_csv(index=False).encode('utf-8')

# --- Main UI Layout (after resource loading) ---
st.title("📄 AI-Powered Resume Analyzer")
st.markdown("""
Upload resumes (PDF format) and provide a job description to get AI-driven match scores and skill analysis.
""")

# --- Global Variables & Initialization ---
MAX_JD_LENGTH = 5000
if 'results' not in st.session_state:
    st.session_state['results'] = None

# --- UI Sections ---
col1, col2 = st.columns(2)

with col1:
    st.header("Upload Resumes")
    uploaded_files = st.file_uploader(
        "Select one or more PDF resumes",
        type="pdf",
        accept_multiple_files=True,
        help="Upload resumes in PDF format. You can select multiple files at once."
    )
    if uploaded_files:
        st.success(f"{len(uploaded_files)} resume(s) selected.")
    else:
        st.info("Please upload at least one resume.")

with col2:
    st.header("Job Description")
    jd_placeholder = "Paste the job description here...\nExample: Seeking a Python developer with experience in Django, REST APIs, and PostgreSQL..."
    job_description_raw = st.text_area(
        "Paste the Job Description below",
        height=300,
        placeholder=jd_placeholder,
        max_chars=MAX_JD_LENGTH,
        key="job_description_input",
        help=f"Enter the full job description (max {MAX_JD_LENGTH} characters)."
    )
    st.caption(f"{len(job_description_raw)} / {MAX_JD_LENGTH} characters")

# --- Analysis Trigger ---
st.divider()

if st.button("Analyze Resumes", type="primary", use_container_width=True, disabled=(not uploaded_files or not job_description_raw)):
    st.session_state['results'] = None # Clear previous results
    results_list = []
    with st.spinner("Analyzing resumes... This may take a few moments."):
        # --- Explicitly use loaded resources ---
        # Use the globally loaded nlp, stop_words, lem
        current_nlp, current_stop_words, current_lem = nlp, stop_words, lem
        # ----------------------------------------

        # 1. Preprocess Job Description
        st.write("Preprocessing Job Description...")
        # Pass the explicit variables
        jd_processed = preprocess(job_description_raw, current_stop_words, current_lem)
        if not jd_processed:
            st.error("Could not process the job description. Please ensure it contains valid text.")
            st.stop()
        st.write("Job Description Preprocessed.")

        # 2. Process Resumes and Calculate Scores/Skills
        valid_resumes_count = 0
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Processing: {uploaded_file.name} ({i+1}/{len(uploaded_files)})...")
            resume_text_raw = extract_text_from_pdf(uploaded_file)
            if resume_text_raw:
                # Pass the explicit variables
                resume_text_processed = preprocess(resume_text_raw, current_stop_words, current_lem)
                if resume_text_processed:
                    # Pass current_nlp to similarity and skill functions
                    match_score = calculate_similarity(current_nlp, jd_processed, resume_text_processed)
                    strengths, missing, feedback = analyze_skills(current_nlp, jd_processed, resume_text_processed)
                    results_list.append({
                        "Resume": uploaded_file.name,
                        "Match Score": match_score,
                        "Key Strengths": ", ".join(strengths) if strengths else "-",
                        "Missing Skills": ", ".join(missing) if missing else "-",
                        "Feedback": feedback
                    })
                    valid_resumes_count += 1
                else:
                     st.warning(f"Could not preprocess text from {uploaded_file.name}. Skipping.")
            progress_bar.progress((i + 1) / len(uploaded_files))

        status_text.text(f"Analysis Complete! Processed {valid_resumes_count} valid resumes.")

        if valid_resumes_count > 0:
            st.success(f"Successfully analyzed {valid_resumes_count} out of {len(uploaded_files)} resumes.")
            st.session_state['results'] = pd.DataFrame(results_list)
        else:
            st.error("No valid resumes could be processed. Please check the uploaded files.")
            st.session_state['results'] = None

# --- Results Dashboard ---
st.divider()
st.header("Analysis Results")

if st.session_state.get('results') is not None and not st.session_state['results'].empty:
    results_df = st.session_state['results']

    # Prepare dataframe for display
    display_df = results_df.copy()
    display_df['Match Score'] = pd.to_numeric(display_df['Match Score'], errors='coerce').fillna(0)

    st.dataframe(
        display_df,
        column_config={
            "Resume": st.column_config.TextColumn(
                "Resume",
                help="The name of the uploaded resume file.",
                width="medium"
            ),
            "Match Score": st.column_config.ProgressColumn(
                "Match Score (%)",
                help="Semantic similarity score between the resume and job description (0-100). Higher is better.",
                format="%.1f",
                min_value=0,
                max_value=100,
            ),
            "Key Strengths": st.column_config.TextColumn(
                "Key Strengths",
                help="Skills/keywords found in both the job description and the resume.",
                width="large"
            ),
            "Missing Skills": st.column_config.TextColumn(
                "Missing Skills",
                help="Skills/keywords found in the job description but not identified in the resume.",
                width="large"
            ),
            "Feedback": st.column_config.TextColumn(
                "Overall Feedback",
                help="A brief summary of the candidate's fit based on skill analysis.",
                width="large"
            )
        },
        use_container_width=True,
        hide_index=True,
    )

    # Download Button
    csv_data = convert_df_to_csv(results_df)
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv_data,
        file_name='resume_analysis_results.csv',
        mime='text/csv',
        use_container_width=True
    )

elif st.session_state.get('results') is not None and st.session_state['results'].empty:
    st.warning("Analysis completed, but no valid results were generated. Check processing messages above.")
else:
    st.info("Upload resumes and a job description, then click 'Analyze Resumes' to see results here.")

# --- Footer/Explanation ---
st.divider()
with st.expander("How Scoring & Analysis Works"):
    # Use the actual model name directly in the text
    st.markdown(f"""
    1.  **Text Extraction:** Text is extracted from uploaded PDF resumes using PyPDF2.
    2.  **Preprocessing:** Both resume text and the job description are cleaned (lowercase, remove special characters & numbers, remove common English stop words, lemmatization using NLTK).
    3.  **Similarity Calculation:** Advanced NLP models (en_core_web_lg via spaCy) convert the cleaned text into numerical representations (vectors). The cosine similarity between the job description vector and each resume vector is calculated.
    4.  **Score:** The similarity score is converted to a percentage (0-100%). Higher scores indicate a better semantic match.
    5.  **Skill Analysis:** Potential skills (noun phrases) are extracted from the job description and resume using spaCy. The system identifies skills present in both ('Key Strengths') and skills present in the JD but missing from the resume ('Missing Skills').
    """)

