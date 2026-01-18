#train and predict
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from parser import extract_text
from features import clean_text

# Load labels
labels_df = pd.read_csv("data/labels.csv")

# Extract text from resumes
texts = []
roles = []
for idx, row in labels_df.iterrows():
    text = extract_text(f"data/resumes/{row['file']}")
    texts.append(clean_text(text))
    roles.append(row['role'])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(texts, roles, test_size=0.2, random_state=42)

# Pipeline: TF-IDF + Logistic Regression
model = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=3000)),
    ("clf", LogisticRegression(max_iter=1000))
])

# Train model
model.fit(X_train, y_train)

# Evaluate
from sklearn.metrics import classification_report
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
import joblib
joblib.dump(model, "resume_classifier.pkl")
