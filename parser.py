# parser.py
import spacy
import re

nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    """
    Cleans resume text:
    - Lowercase
    - Remove special characters & numbers
    - Lemmatization is synonym to one understanding
    - Remove stopwords
    """
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)  # remove numbers & special chars
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)
