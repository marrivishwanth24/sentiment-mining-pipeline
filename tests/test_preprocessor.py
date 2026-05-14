"""
tests/test_preprocessor.py
Tests for the NLP preprocessing pipeline.
"""

import pytest
import pandas as pd
from pipeline.preprocessor import clean_text, tokenize_and_filter, label_sentiment, preprocess_dataframe


def test_clean_text_removes_urls():
    text = "Great product! Check http://example.com for more info"
    result = clean_text(text)
    assert 'http' not in result
    assert 'example' not in result


def test_clean_text_lowercases():
    result = clean_text("GREAT Product")
    assert result == result.lower()


def test_clean_text_removes_special_chars():
    result = clean_text("Hello!!! This is great #amazing @user")
    assert '!' not in result
    assert '#' not in result


def test_clean_text_handles_none():
    result = clean_text(None)
    assert result == ''


def test_tokenize_removes_stopwords():
    tokens = tokenize_and_filter("this is a great product")
    assert 'this' not in tokens
    assert 'is' not in tokens
    assert 'great' in tokens or 'product' in tokens


def test_label_sentiment_positive():
    assert label_sentiment(5) == 'positive'
    assert label_sentiment(4) == 'positive'
    assert label_sentiment('5 stars') == 'positive'


def test_label_sentiment_negative():
    assert label_sentiment(1) == 'negative'
    assert label_sentiment(2) == 'negative'


def test_label_sentiment_neutral():
    assert label_sentiment(3) == 'neutral'


def test_label_sentiment_invalid():
    result = label_sentiment('not a number')
    assert result == 'neutral'


def test_preprocess_dataframe_basic():
    df = pd.DataFrame({
        'text': ['This product is amazing!', 'Terrible quality, very bad', 'Its okay I guess'],
        'rating': [5, 1, 3]
    })
    result = preprocess_dataframe(df)

    assert 'cleaned_text' in result.columns
    assert 'tokens' in result.columns
    assert 'processed_text' in result.columns
    assert 'sentiment' in result.columns
    assert len(result) <= len(df)


def test_preprocess_dataframe_drops_empty():
    df = pd.DataFrame({
        'text': ['Valid review text here', None, '', '   '],
        'rating': [5, 3, 2, 1]
    })
    result = preprocess_dataframe(df)
    assert len(result) == 1
