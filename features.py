# TF (Term Frequency) and IDF (Inverse Document Frequency)
from sklearn.feature_extraction.text import TfidfVectorizer

SKILLS = {
    "programming": ["python", "java", "c++", "sql", "r"],
    "data": ["machine learning", "deep learning", "statistics", "pandas", "numpy"],
    "tools": ["tableau", "power bi", "excel", "aws", "git"],
    "product": ["roadmap", "stakeholders", "user stories", "a/b testing"]
}

def extract_skills(text):
    """
    Extract skills using keyword matching.
    Returns a dictionary of skill categories.
    """
    extracted = {}
    for category, keywords in SKILLS.items():
        extracted[category] = [kw for kw in keywords if kw in text]
    return extracted


def build_tfidf(corpus, max_features=3000):
    """
    Convert cleaned text corpus into TF-IDF features.
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2)
    )
    X = vectorizer.fit_transform(corpus)
    return X, vectorizer

def resume_stats(text):
    """
    Extract basic resume statistics.
    """
    return {
        "word_count": len(text.split()),
        "skill_count": sum(len(v) for v in extract_skills(text).values())
    }
