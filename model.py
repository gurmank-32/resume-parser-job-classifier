#train and predict
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Import text cleaning (NOT PDF extraction)
from parser import clean_text

# ==============================
# 1. Load Kaggle Resume Dataset
# ==============================
df = pd.read_csv("data/resumes/resume_data.csv")
df.rename(columns={'﻿job_position_name': 'job_position_name'}, inplace=True)
df['job_position_name'] = df['job_position_name'].astype(str).str.strip()
df['job_position_name'] = df['job_position_name'].replace(['', 'nan', 'None'], pd.NA)
print(df['job_position_name'].notna().sum())
print(df['job_position_name'].value_counts().head(10))

# IMPORTANT: Verify column names
print("Dataset columns:", df.columns)
TEXT_COLUMNS = [
    "career_objective",
    "skills",
    "responsibilities",
    "related_skils_in_job",
    "positions",
    "degree_names",
    "major_field_of_studies",
    "professional_company_names",
    "certification_skills"
]

# Fill missing values
df[TEXT_COLUMNS] = df[TEXT_COLUMNS].fillna("")

# Combine into ONE text feature
df["combined_text"] = df[TEXT_COLUMNS].astype(str).agg(" ".join, axis=1)


# Typical Kaggle columns
X = df["combined_text"]     # Resume TEXT
y = df["job_position_name"]       # Job Role LABEL

# ==============================
# 2. Clean Resume Text
# ==============================
X = X.apply(clean_text)

# ==============================
# 3. Train-Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==============================
# 4. Build ML Pipeline
# ==============================
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            max_features=3000,
            ngram_range=(1, 2)
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        )
    )
])

# ==============================
# 5. Train Model
# ==============================
model.fit(X_train, y_train)

# ==============================
# 6. Evaluate Model
# ==============================
y_pred = model.predict(X_test)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ==============================
# 7. Save Trained Model
# ==============================
joblib.dump(model, "resume_classifier.pkl")

print("\nModel saved as resume_classifier.pkl")
