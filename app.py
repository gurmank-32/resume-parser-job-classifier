# streamlit app
import streamlit as st
import joblib
import numpy as np

from parser import extract_text, clean_text
from features import extract_skills, resume_stats

# ==============================
# Load trained model
# ==============================
model = joblib.load("resume_classifier.pkl")

# ==============================
# Streamlit UI
# ==============================
st.set_page_config(page_title="Resume Parser & Job Role Classifier")
st.title("📄 Resume Parser & Job Role Classifier")

st.write(
    "Upload a resume PDF to predict the most suitable job role "
    "and extract key skills."
)

# ==============================
# Upload PDF
# ==============================
uploaded_file = st.file_uploader(
    "Upload Resume (PDF)",
    type=["pdf"]
)

if uploaded_file is not None:

    # ==============================
    # Extract text from PDF
    # ==============================
    raw_text = extract_text(uploaded_file)

    if raw_text.strip() == "":
        st.error("Could not extract text from the PDF.")
    else:
        # ==============================
        # Clean text (SAME as training)
        # ==============================
        cleaned_text = clean_text(raw_text)

        # ==============================
        # Predict job role (Top-1)
        # ==============================
        prediction = model.predict([cleaned_text])[0]

        # ==============================
        # Predict probabilities (Top-3)
        # ==============================
        probs = model.predict_proba([cleaned_text])[0]
        classes = model.classes_

        top_n = 3
        top_indices = np.argsort(probs)[-top_n:][::-1]

        # ==============================
        # Extract skills & stats
        # ==============================
        skills = extract_skills(raw_text)
        stats = resume_stats(raw_text)

        # ==============================
        # Display Results
        # ==============================
        st.subheader("🏆 Top 3 Matching Job Roles")

        for rank, idx in enumerate(top_indices, start=1):
            st.write(
                f"**{rank}. {classes[idx]}** — Confidence: {probs[idx]:.2f}"
            )

        st.subheader("🧠 Extracted Skills")
        for category, skill_list in skills.items():
            if skill_list:
                st.write(f"**{category.capitalize()}**: {', '.join(skill_list)}")

        st.subheader("📊 Resume Statistics")
        st.write(f"**Word Count:** {stats['word_count']}")
        st.write(f"**Total Skills Found:** {stats['skill_count']}")

        # Optional: show raw text
        with st.expander("🔍 View Extracted Resume Text"):
            st.text(raw_text)
