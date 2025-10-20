"""
Text preprocessing utilities for the Review of Scientific Papers project.
"""

import re
import pandas as pd
import spacy
import nltk
from nltk.corpus import stopwords

# Download NLTK stopwords (first run only)
nltk.download("stopwords", quiet=True)

# Load spaCy small English model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # Download automatically if missing
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

EN_STOPWORDS = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    """Lowercase, remove numbers/punctuation, and strip spaces."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_and_lemmatize(text: str) -> str:
    """Tokenize and lemmatize text using spaCy; remove stopwords."""
    doc = nlp(text)
    lemmas = [
        token.lemma_
        for token in doc
        if token.is_alpha and token.lemma_ not in EN_STOPWORDS
    ]
    return " ".join(lemmas)


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply text cleaning and lemmatization to the citation_context column."""
    df = df.copy()
    df["clean_text"] = df["citation_context"].apply(clean_text)
    df["clean_text"] = df["clean_text"].apply(tokenize_and_lemmatize)
    return df


if __name__ == "__main__":
    # Simple test when running standalone
    sample = "This method SIGNIFICANTLY improves performance, unlike previous work (Smith 2020)."
    print("Original:", sample)
    print("Cleaned:", tokenize_and_lemmatize(clean_text(sample)))
