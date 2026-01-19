# Resume Parser and Job Classifier
An end-to-end NLP-based machine learning application that parses resumes in PDF format, extracts relevant information, and recommends the top 3 most suitable job roles using probabilistic classification.
This project simulates how Applicant Tracking Systems (ATS) and job-matching platforms analyze resumes in real-world hiring pipelines.

---

🚀 Features
- 📥 Upload resumes in PDF format
- 🧠 Extract and clean unstructured resume text using spaCy
- 📊 Classify resumes into job roles using NLP & Machine Learning
- 🏆 Recommend Top-3 matching job roles with confidence scores
- 🧰 Extract skills using keyword-based matching
- 📈 Display resume statistics (word count, skill count)
- 🌐 Interactive web interface built with Streamlit

---

### 🧠 Project Architecture

**Resume (PDF)**  
↓  
**PyPDF2 (Text Extraction)**  
↓  
**spaCy NLP Pipeline (Cleaning & Lemmatization)**  
↓  
**TF-IDF Vectorization**  
↓  
**Logistic Regression Classifier**  
↓  
**Top-3 Job Role Predictions + Skills**  
↓  
**Streamlit Web Application**

---

## 🛠️ Tech Stack

| Category       | Tools                  |
|----------------|-----------------------|
| Language       | Python                |
| NLP (Natural Language Processing)            | spaCy                 |
| ML             | scikit-learn          |
| PDF Parsing    | PyPDF2                |
| Vectorization  | TF-IDF (Term Frequency-Inverse Document Frequency)               |
| Model          | Logistic Regression   |
| Deployment     | Streamlit             |

---
## 📂 Project Structure

resume-parser-job-classifier/  
│  
├── app.py                  # 🌐 Streamlit application  
├── model.py                # 🧠 Model training script  
├── parser.py               # 🧹 Resume text extraction & cleaning  
├── features.py             # 🛠️ Skill extraction & resume stats  
├── resume_classifier.pkl   # 🤖 Trained ML model  
├── data/  
│   └── resumes/  
│       └── resume_data.csv # 📊 Kaggle dataset  
├── requirements.txt  
└── README.md  

---

## 📊 Dataset

**Source:** Kaggle Resume Dataset  
Contains structured resume information such as:  

- Skills  
- Career objectives  
- Education  
- Experience  
- Job roles (labels)  

**Training Labels**  

- `X (Features)`: Combined resume text fields  
- `y (Target)`: Job role / position name  

---

## 🧪 Model Details

- **Text Representation:** TF-IDF (unigrams + bigrams)  
- **Classifier:** Logistic Regression (multi-class)  
- **Prediction Strategy:** Top-3 roles using probability distribution (`predict_proba`)  

**Why Top-3?**  
- Resumes often span multiple roles  
- Improves realism and interpretability  

---

## 📸 Sample Output

-🏆 Top-3 Matching Job Roles
  - Data Engineer — Confidence: 0.07
  - Data Scientist — Confidence: 0.06
  - Big Data Analyst — Confidence: 0.06
    
- 🛠️ Extracted Skills
  - Programming: Python, Java, SQL
  - Data: Machine Learning, Spark
  - Tools: AWS, Git

---
🎯 Why This Project?
- 🏗️ Demonstrates end-to-end ML system design
- 🧹 Handles real-world noisy text data
- 🧠 Emphasizes interpretability over blind accuracy
- 🌐 Mimics real hiring recommendation systems

---
🔮 Future Enhancements
- 🔍 Explainable AI (highlight keywords influencing predictions)
- 🤖 BERT / Sentence Transformer embeddings
- ⚖️ Class imbalance handling
- 📊 Model evaluation dashboard
- ☁️ Cloud deployment

---

## 💻 How to Run the Project

1️⃣ **Clone the repository**

2️⃣ **Install dependencies**
   - pip install -r requirements.txt
   - python -m spacy download en_core_web_sm

3️⃣ **Run the Streamlit app**
   - streamlit run app.py

---
