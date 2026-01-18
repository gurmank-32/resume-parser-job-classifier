# Streamlit app 
uploaded_file = st.file_uploader("Upload your resume (PDF)", type="pdf")

if uploaded_file is not None:
    # Extract text from the uploaded PDF
    text = extract_text(uploaded_file)  # ← THIS LINE USES THE FUNCTION
    cleaned_text = clean_text(text)
    X_vector = vectorizer.transform([cleaned_text])
    predicted_role = model.predict(X_vector)[0]
