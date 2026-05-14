"""
preprocessor.py
---------------
NLP text preprocessing and feature engineering for sentiment classification.
Applies tokenization, cleaning, stopword removal, stemming, and TF-IDF vectorization.
"""

import re
import string
import logging
import argparse
import pickle
from pathlib import Path

import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

STOP_WORDS = set(stopwords.words('english'))
STEMMER = PorterStemmer()
LEMMATIZER = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """
    Clean raw review text by removing noise.

    Steps:
    1. Lowercase
    2. Remove URLs, HTML tags, special characters
    3. Remove punctuation and extra whitespace
    """
    if not isinstance(text, str):
        return ''

    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def tokenize_and_filter(text: str, use_stemming: bool = False) -> list[str]:
    """
    Tokenize text and remove stopwords.
    Optionally apply stemming or lemmatization.

    Args:
        text: Cleaned input text
        use_stemming: If True, apply stemming; else apply lemmatization

    Returns:
        List of processed tokens
    """
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]

    if use_stemming:
        tokens = [STEMMER.stem(t) for t in tokens]
    else:
        tokens = [LEMMATIZER.lemmatize(t) for t in tokens]

    return tokens


def label_sentiment(rating) -> str:
    """
    Convert numeric star ratings to sentiment labels.

    Args:
        rating: Numeric rating (1-5) or string

    Returns:
        'positive', 'negative', or 'neutral'
    """
    try:
        r = float(str(rating).replace('stars', '').replace('star', '').strip())
        if r >= 4.0:
            return 'positive'
        elif r <= 2.0:
            return 'negative'
        else:
            return 'neutral'
    except (ValueError, TypeError):
        return 'neutral'


def preprocess_dataframe(df: pd.DataFrame, text_col: str = 'text', rating_col: str = 'rating') -> pd.DataFrame:
    """
    Full preprocessing pipeline for a reviews DataFrame.

    Args:
        df: Input DataFrame with review text and ratings
        text_col: Column name containing review text
        rating_col: Column name containing star ratings

    Returns:
        Processed DataFrame with cleaned text, tokens, and sentiment labels
    """
    logger.info(f"Preprocessing {len(df)} reviews...")

    df = df.dropna(subset=[text_col]).copy()
    df['cleaned_text'] = df[text_col].apply(clean_text)
    df = df[df['cleaned_text'].str.len() > 10].copy()

    df['tokens'] = df['cleaned_text'].apply(tokenize_and_filter)
    df['processed_text'] = df['tokens'].apply(lambda t: ' '.join(t))

    df['token_count'] = df['tokens'].apply(len)
    df['char_count'] = df['cleaned_text'].apply(len)

    if rating_col in df.columns:
        df['sentiment'] = df[rating_col].apply(label_sentiment)
    else:
        df['sentiment'] = 'unknown'

    logger.info(f"Sentiment distribution:\n{df['sentiment'].value_counts()}")
    logger.info(f"Preprocessing complete. {len(df)} valid reviews retained.")

    return df


def build_tfidf_features(df: pd.DataFrame, max_features: int = 10000, test_size: float = 0.2):
    """
    Build TF-IDF feature matrix and split into train/test sets.

    Args:
        df: Preprocessed DataFrame
        max_features: Max vocabulary size for TF-IDF
        test_size: Fraction of data for test split

    Returns:
        X_train, X_test, y_train, y_test, vectorizer
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )

    df_labeled = df[df['sentiment'] != 'unknown'].copy()

    X = vectorizer.fit_transform(df_labeled['processed_text'])
    y = df_labeled['sentiment']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    logger.info(f"TF-IDF features: {X.shape[1]} terms")
    logger.info(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    return X_train, X_test, y_train, y_test, vectorizer


def save_vectorizer(vectorizer: TfidfVectorizer, path: str = 'models/vectorizer.pkl'):
    """Serialize and save the fitted TF-IDF vectorizer."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(vectorizer, f)
    logger.info(f"Vectorizer saved to {path}")


def load_vectorizer(path: str = 'models/vectorizer.pkl') -> TfidfVectorizer:
    """Load a previously saved TF-IDF vectorizer."""
    with open(path, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess review data')
    parser.add_argument('--input', required=True, help='Input CSV path')
    parser.add_argument('--output', required=True, help='Output CSV path')
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    processed = preprocess_dataframe(df)
    processed.to_csv(args.output, index=False)
    print(f"Saved {len(processed)} processed reviews to {args.output}")
