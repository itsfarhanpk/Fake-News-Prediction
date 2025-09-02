import os
import re
import nltk
import streamlit as st
import pandas as pd
from io import StringIO
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# ──────────────────────────────────────────────────────────────────────────────
# 0. Ensure NLTK data
# ──────────────────────────────────────────────────────────────────────────────
try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords")

# ──────────────────────────────────────────────────────────────────────────────
# 1. Text preprocessing
# ──────────────────────────────────────────────────────────────────────────────
def stem_text(text: str) -> str:
    porter = PorterStemmer()
    tokens = re.sub(r"[^a-zA-Z]", " ", text).lower().split()
    filtered = [porter.stem(tok) for tok in tokens if tok not in stopwords.words("english")]
    return " ".join(filtered)

# ──────────────────────────────────────────────────────────────────────────────
# 2. Load & vectorize dataset (cached)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_vectorize(csv_path: str):
    df = pd.read_csv(csv_path).fillna("")
    df["content"] = (df["author"] + " " + df["title"]).apply(stem_text)
    X_texts = df["content"].tolist()
    y = df["label"].values

    vect = TfidfVectorizer(max_features=5000)
    X = vect.fit_transform(X_texts)
    return X, y, vect

# ──────────────────────────────────────────────────────────────────────────────
# 3. Train model once as a resource
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def build_model():
    # use the already-cached data
    X, y, _vect = load_and_vectorize(DATA_PATH)
    X_tr, _, y_tr, _ = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=2
    )
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_tr, y_tr)
    return clf

# ──────────────────────────────────────────────────────────────────────────────
# 4. Streamlit app
# ──────────────────────────────────────────────────────────────────────────────
st.title("📰 Fake vs Real News Classifier")

DATA_PATH = r"train.csv"
X, y, vectorizer = load_and_vectorize(DATA_PATH)
model = build_model()

news_text = st.text_area("Enter the news text to classify:")

if st.button("🔍 Predict"):
    if not news_text.strip():
        st.warning("Please enter some text before predicting.")
    else:
        processed = stem_text(news_text)
        vec = vectorizer.transform([processed])
        pred = model.predict(vec)[0]
        if pred == 0:
            st.success("✅ This news is **Real**.")
        else:
            st.error("⚠️ This news is **Fake**.")
