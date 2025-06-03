Here's a clean, professional, and well-formatted `README.md` for your **AI-Powered Resume Screening Tool**:

---

# 🧠 AI-Powered Resume Screening Tool

A **Streamlit web app** that intelligently screens multiple PDF resumes against a job description using advanced **Natural Language Processing (NLP)** techniques. This tool provides **match scores**, identifies **key strengths and missing skills**, and delivers **actionable feedback** for each candidate.

---

## 🚀 Features

* 📁 **Batch Resume Upload** – Analyze multiple PDF resumes at once.
* 📝 **Job Description Input** – Paste or upload a job description to tailor the analysis.
* 🤖 **AI-Driven Scoring** – Uses **spaCy (`en_core_web_lg`)** and **cosine similarity** for semantic matching.
* 🧩 **Skill Gap Analysis** – Compares extracted skills from resumes and job descriptions.
* 📊 **Downloadable Results** – Export the analysis to a CSV file.
* 🖥️ **Interactive Dashboard** – Built with Streamlit for a clean and intuitive interface.

---

## 🛠️ Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/resume-screening-tool.git
cd resume-screening-tool
```

### 2. Set Up a Virtual Environment

```bash
python -m venv venv
# Activate the environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

### 4. Run the Application

```bash
streamlit run app.py
```

Visit: [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📋 Usage Guide

1. **Upload Resumes** – Click "Browse files" to upload one or more PDF resumes.
2. **Enter Job Description** – Paste the job description or upload a `.txt` file (sample provided).
3. **Click "Analyze Resumes"** – The app will begin analyzing.
4. **View Results** – See:

   * Match Score
   * Highlighted Strengths
   * Missing Skills
   * Feedback Summary
5. **Download CSV** – Export the results for further review or reporting.

---

## 📦 Requirements

* Python 3.7+
* See `requirements.txt` for the full list of Python packages.

---

## 🧪 Sample Data

Included in the repository for testing purposes:

* `resume_alice_strong.pdf`
* `resume_bob_average.pdf`
* `resume_charlie_weak.pdf`
* `sample_job_description.txt`

---

## ⚙️ How It Works

1. **Text Extraction** – Uses `PyPDF2` to extract text from PDF resumes.
2. **Preprocessing** – Cleans, tokenizes, lemmatizes, and removes stopwords (via `nltk`).
3. **Semantic Similarity** – Applies spaCy vectors and cosine similarity to compute resume-job match scores.
4. **Skill Extraction** – Extracts and compares relevant skills between resumes and job descriptions.
5. **Feedback Generation** – Summarizes fit, strengths, and skill gaps.

---

## 📄 License

This project is for **educational and demonstration purposes** only.

---

## 🧰 Built With

* [Streamlit](https://streamlit.io/)
* [spaCy](https://spacy.io/)
* [nltk](https://www.nltk.org/)
* [PyPDF2](https://pypi.org/project/PyPDF2/)

---

