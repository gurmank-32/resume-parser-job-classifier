# Streamlit app 
import streamlit as st
import joblib
from parser import extract_text
from features import clean_text, extract_skills

# Load trained model
model = joblib.load("resume_classifier.pkl")

st.title("Resume Parser & Job Role Classifier")

uploaded_file = st.file_uploader("Upload Resume (PDF)", type="pdf")

if uploaded_file:
    text = extract_text(uploaded_file)
    cleaned_text = clean_text(text)
    role = model.predict([cleaned_text])[0]
    skills = extract_skills(text)

    st.subheader("Predicted Job Role")
    st.success(role)

    st.subheader("Extracted Skills")
    st.write(skills)

