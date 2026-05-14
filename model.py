"""
model.py
--------
Machine learning model training, evaluation, and inference for sentiment classification.
Supports Logistic Regression and SVM with cross-validation and detailed metrics reporting.
"""

import logging
import pickle
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from pipeline.preprocessor import preprocess_dataframe, build_tfidf_features, save_vectorizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SentimentClassifier:
    """
    Sentiment classification model wrapping Scikit-learn estimators.
    Supports training, evaluation, cross-validation, and inference.
    """

    MODELS = {
        'logistic_regression': LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        'svm': LinearSVC(
            C=0.5,
            class_weight='balanced',
            max_iter=2000,
            random_state=42
        )
    }

    def __init__(self, model_type: str = 'logistic_regression'):
        if model_type not in self.MODELS:
            raise ValueError(f"model_type must be one of {list(self.MODELS.keys())}")

        self.model_type = model_type
        self.model = self.MODELS[model_type]
        self.is_trained = False
        self.classes_ = None

    def train(self, X_train, y_train) -> 'SentimentClassifier':
        """Train the classifier on training data."""
        logger.info(f"Training {self.model_type} on {X_train.shape[0]} samples...")
        self.model.fit(X_train, y_train)
        self.classes_ = self.model.classes_
        self.is_trained = True
        logger.info("Training complete.")
        return self

    def evaluate(self, X_test, y_test) -> dict:
        """
        Evaluate model performance on test data.

        Returns:
            Dict containing accuracy, precision, recall, F1, and full report
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before evaluation")

        y_pred = self.model.predict(X_test)

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'report': classification_report(y_test, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }

        logger.info("\n" + "="*50)
        logger.info(f"MODEL: {self.model_type.upper()}")
        logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall:    {metrics['recall']:.4f}")
        logger.info(f"F1-Score:  {metrics['f1']:.4f}")
        logger.info("\nClassification Report:\n" + metrics['report'])

        return metrics

    def cross_validate(self, X, y, cv: int = 5) -> dict:
        """
        Run k-fold cross-validation and return mean scores.

        Args:
            X: Feature matrix
            y: Target labels
            cv: Number of folds

        Returns:
            Dict of mean and std for each metric
        """
        logger.info(f"Running {cv}-fold cross-validation...")

        results = {}
        for metric in ['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted']:
            scores = cross_val_score(self.model, X, y, cv=cv, scoring=metric, n_jobs=-1)
            results[metric] = {'mean': scores.mean(), 'std': scores.std()}
            logger.info(f"{metric}: {scores.mean():.4f} ± {scores.std():.4f}")

        return results

    def predict(self, texts: list[str], vectorizer) -> list[str]:
        """
        Predict sentiment for new text samples.

        Args:
            texts: List of raw review strings
            vectorizer: Fitted TF-IDF vectorizer

        Returns:
            List of predicted sentiment labels
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")

        from pipeline.preprocessor import clean_text, tokenize_and_filter
        processed = [' '.join(tokenize_and_filter(clean_text(t))) for t in texts]
        X = vectorizer.transform(processed)
        return self.model.predict(X).tolist()

    def predict_proba(self, texts: list[str], vectorizer) -> list[dict]:
        """
        Get class probability estimates (Logistic Regression only).

        Returns:
            List of dicts mapping class label → probability
        """
        if self.model_type != 'logistic_regression':
            raise NotImplementedError("predict_proba only available for logistic_regression")

        from pipeline.preprocessor import clean_text, tokenize_and_filter
        processed = [' '.join(tokenize_and_filter(clean_text(t))) for t in texts]
        X = vectorizer.transform(processed)
        probs = self.model.predict_proba(X)

        return [
            {cls: float(prob) for cls, prob in zip(self.classes_, row)}
            for row in probs
        ]

    def save(self, path: str = 'models/sentiment_model.pkl'):
        """Serialize and save the trained model."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str = 'models/sentiment_model.pkl') -> 'SentimentClassifier':
        """Load a previously saved model."""
        with open(path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model loaded from {path}")
        return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and evaluate sentiment model')
    parser.add_argument('--input', required=True, help='Preprocessed CSV path')
    parser.add_argument('--model', default='logistic_regression', help='Model type')
    parser.add_argument('--cv', type=int, default=5, help='Cross-validation folds')
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    processed = preprocess_dataframe(df)
    X_train, X_test, y_train, y_test, vectorizer = build_tfidf_features(processed)

    classifier = SentimentClassifier(model_type=args.model)
    classifier.train(X_train, y_train)
    metrics = classifier.evaluate(X_test, y_test)

    classifier.save()
    save_vectorizer(vectorizer)

    print(f"\nFinal Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1-Score: {metrics['f1']:.4f}")
