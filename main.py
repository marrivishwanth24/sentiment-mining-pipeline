"""
main.py — Sentiment Analysis Pipeline
--------------------------------------
Trains on a real CSV dataset (Amazon Fine Food Reviews format).
Expected CSV columns: 'Score' (1-5 stars) and 'Text' (review text).

Place your CSV at: data/reviews.csv
Then run: uvicorn main:app --reload
"""

import os
import re
import pickle
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
logger = logging.getLogger(__name__)

# ── Stopwords ──────────────────────────────────────────────────────────────────
STOPWORDS = {
    "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
    "yourself","yourselves","he","him","his","himself","she","her","hers",
    "herself","it","its","itself","they","them","their","theirs","themselves",
    "what","which","who","whom","this","that","these","those","am","is","are",
    "was","were","be","been","being","have","has","had","having","do","does",
    "did","doing","a","an","the","and","but","if","or","because","as","until",
    "while","of","at","by","for","with","about","against","between","into",
    "through","during","before","after","above","below","to","from","up","down",
    "in","out","on","off","over","under","again","further","then","once","here",
    "there","when","where","why","how","all","both","each","few","more","most",
    "other","some","such","no","nor","not","only","own","same","so","than",
    "too","very","s","t","can","will","just","don","should","now","d","ll",
    "m","o","re","ve","y","ain","aren","couldn","didn","doesn","hadn","hasn",
    "haven","isn","ma","mightn","mustn","needn","shan","shouldn","wasn",
    "weren","won","wouldn",
}

# ── Text preprocessing ─────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize_and_filter(text: str) -> list[str]:
    tokens = text.split()
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]

def preprocess(text: str) -> str:
    return " ".join(tokenize_and_filter(clean_text(text)))

def label_from_score(score) -> str:
    """Convert star rating 1-5 to sentiment label."""
    try:
        s = float(score)
        if s >= 4:   return "positive"
        elif s <= 2: return "negative"
        else:        return "neutral"
    except (ValueError, TypeError):
        return None

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_PATH  = Path("data/reviews.csv")
MODEL_PATH = Path("models/sentiment_model.pkl")
VEC_PATH   = Path("models/vectorizer.pkl")
META_PATH  = Path("models/metrics.pkl")

# ── Global state ───────────────────────────────────────────────────────────────
model      = None
vectorizer = None
metrics    = {}

# ── Real training from CSV ─────────────────────────────────────────────────────
def train_from_csv(csv_path: Path, sample_size: int = 50_000):
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, classification_report
    )

    logger.info(f"Loading dataset from {csv_path} ...")
    df = pd.read_csv(csv_path)
    logger.info(f"Raw rows: {len(df)}")

    # Normalise column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Find text column
    text_col = next(
        (c for c in ["text", "review", "review_text", "reviewtext", "comment", "content"]
         if c in df.columns), None
    )
    if text_col is None:
        raise ValueError(
            f"Could not find a text column. Columns found: {list(df.columns)}\n"
            "Rename your review text column to 'Text'."
        )

    # Find label column
    if "sentiment" in df.columns:
        df["label"] = df["sentiment"].str.lower().str.strip()
        df = df[df["label"].isin(["positive", "negative", "neutral"])]
    else:
        score_col = next(
            (c for c in ["score", "label", "rating", "stars", "overall"]
             if c in df.columns), None
        )
        if score_col is None:
            raise ValueError(
                f"Could not find a score column. Columns found: {list(df.columns)}\n"
                "Rename your rating column to 'Score'."
            )
        df["label"] = df[score_col].apply(label_from_score)
        df = df.dropna(subset=["label"])

    df = df.dropna(subset=[text_col])
    df = df[df[text_col].astype(str).str.strip() != ""]

    # Balance classes
    min_count = min(df["label"].value_counts().min(), sample_size // 3)
    balanced = pd.concat([
        df[df["label"] == cls].sample(
            n=min(len(df[df["label"] == cls]), min_count), random_state=42
        )
        for cls in ["positive", "negative", "neutral"]
    ]).sample(frac=1, random_state=42).reset_index(drop=True)

    logger.info(f"Balanced dataset: {len(balanced)} — {dict(balanced['label'].value_counts())}")

    # Preprocess text
    logger.info("Preprocessing text ...")
    balanced["processed"] = balanced[text_col].astype(str).apply(preprocess)
    balanced = balanced[balanced["processed"].str.strip() != ""]

    X_raw = balanced["processed"].tolist()
    y     = balanced["label"].tolist()

    # TF-IDF
    logger.info("Fitting TF-IDF vectorizer ...")
    vec = TfidfVectorizer(
        max_features=15_000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )
    X = vec.fit_transform(X_raw)

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train model
    logger.info("Training Logistic Regression ...")
    clf = LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        multi_class="multinomial",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    prec   = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec    = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1     = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    report = classification_report(y_test, y_pred, zero_division=0)

    # Cross-validation
    logger.info("Running 5-fold cross-validation ...")
    cv_scores = cross_val_score(
        clf, X_train, y_train, cv=5, scoring="f1_weighted", n_jobs=-1
    )

    real_metrics = {
        "accuracy":           round(acc  * 100, 2),
        "precision":          round(prec * 100, 2),
        "recall":             round(rec  * 100, 2),
        "f1":                 round(f1   * 100, 2),
        "cv_f1_mean":         round(cv_scores.mean() * 100, 2),
        "cv_f1_std":          round(cv_scores.std()  * 100, 2),
        "train_size":         X_train.shape[0],
        "test_size":          X_test.shape[0],
        "total_samples":      len(balanced),
        "vocabulary_size":    len(vec.vocabulary_),
        "report":             report,
        "class_distribution": dict(balanced["label"].value_counts()),
        "dataset":            str(csv_path),
    }

    logger.info(f"\n{'='*50}")
    logger.info(f"Accuracy:  {real_metrics['accuracy']}%")
    logger.info(f"Precision: {real_metrics['precision']}%")
    logger.info(f"Recall:    {real_metrics['recall']}%")
    logger.info(f"F1-Score:  {real_metrics['f1']}%")
    logger.info(f"CV F1:     {real_metrics['cv_f1_mean']}% ± {real_metrics['cv_f1_std']}%")
    logger.info(f"\n{report}")

    # Save
    Path("models").mkdir(exist_ok=True)
    with open(MODEL_PATH, "wb") as f: pickle.dump(clf, f)
    with open(VEC_PATH,   "wb") as f: pickle.dump(vec, f)
    with open(META_PATH,  "wb") as f: pickle.dump(real_metrics, f)

    logger.info("Model, vectorizer and metrics saved.")
    return clf, vec, real_metrics


# ── Fallback training (no dataset) ────────────────────────────────────────────
def fallback_train():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    logger.warning("No dataset found — using seed data. Add data/reviews.csv for real metrics.")

    samples = [
        ("absolutely love amazing quality highly recommend outstanding", "positive"),
        ("great works perfectly fast delivery excellent five stars", "positive"),
        ("fantastic exceeded expectations would buy again wonderful", "positive"),
        ("perfect quick shipping very satisfied love well made durable", "positive"),
        ("excellent quality pleased recommend everyone awesome product", "positive"),
        ("terrible broke after days complete waste money worst ever", "negative"),
        ("poor quality avoid disappointed horrible stopped working useless", "negative"),
        ("defective arrived broken customer service awful garbage", "negative"),
        ("does not work misleading description terrible build quality", "negative"),
        ("worst bought waste time return nightmare disappointing broken", "negative"),
        ("product okay nothing special does job average quality", "neutral"),
        ("decent meets basic needs nothing extraordinary mediocre fine", "neutral"),
        ("average works nothing special acceptable okay serves purpose", "neutral"),
    ]

    texts  = [s[0] for s in samples]
    labels = [s[1] for s in samples]

    vec = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
    X   = vec.fit_transform(texts)
    clf = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=42)
    clf.fit(X, labels)

    fallback_metrics = {
        "accuracy":  "N/A",
        "precision": "N/A",
        "recall":    "N/A",
        "f1":        "N/A",
        "note":      "Running in fallback mode. Place reviews CSV at data/reviews.csv and restart.",
    }

    Path("models").mkdir(exist_ok=True)
    with open(MODEL_PATH, "wb") as f: pickle.dump(clf, f)
    with open(VEC_PATH,   "wb") as f: pickle.dump(vec, f)
    with open(META_PATH,  "wb") as f: pickle.dump(fallback_metrics, f)

    return clf, vec, fallback_metrics


# ── FastAPI ────────────────────────────────────────────────────────────────────
app = FastAPI(title="Sentiment Analysis API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.on_event("startup")
def startup():
    global model, vectorizer, metrics
    if MODEL_PATH.exists() and VEC_PATH.exists():
        logger.info("Loading saved model ...")
        with open(MODEL_PATH, "rb") as f: model      = pickle.load(f)
        with open(VEC_PATH,   "rb") as f: vectorizer = pickle.load(f)
        if META_PATH.exists():
            with open(META_PATH, "rb") as f: metrics = pickle.load(f)
        logger.info("Model loaded successfully")
    elif DATA_PATH.exists():
        model, vectorizer, metrics = train_from_csv(DATA_PATH)
    else:
        model, vectorizer, metrics = fallback_train()


# ── Schemas ────────────────────────────────────────────────────────────────────
class ReviewRequest(BaseModel):
    text: str


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.get("/health")
def health():
    return {
        "status":       "healthy",
        "model_loaded": model is not None,
        "dataset_used": DATA_PATH.exists(),
    }


@app.get("/metrics")
def get_metrics():
    """Return real training metrics — accuracy, F1, cross-validation scores."""
    return metrics


@app.post("/analyze")
def analyze(req: ReviewRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Please enter some text")

    processed  = preprocess(req.text)
    tokens     = tokenize_and_filter(clean_text(req.text))
    X          = vectorizer.transform([processed])
    prediction = model.predict(X)[0]
    proba      = model.predict_proba(X)[0]
    confidence = round(float(max(proba)) * 100, 1)

    scores = {
        cls: round(float(p) * 100, 1)
        for cls, p in zip(model.classes_, proba)
    }

    return {
        "sentiment":   prediction,
        "confidence":  confidence,
        "scores":      scores,
        "word_count":  len(tokens),
        "top_words":   tokens[:10],
    }


@app.post("/batch")
def batch(reviews: list[str]):
    """Analyze up to 50 reviews at once and return a summary breakdown."""
    if not reviews:
        raise HTTPException(status_code=400, detail="Provide at least one review")

    reviews = reviews[:50]
    results = []

    for text in reviews:
        processed  = preprocess(text)
        X          = vectorizer.transform([processed])
        pred       = model.predict(X)[0]
        proba      = model.predict_proba(X)[0]
        confidence = round(float(max(proba)) * 100, 1)
        results.append({
            "text":       text[:120],
            "sentiment":  pred,
            "confidence": confidence,
            "scores": {
                cls: round(float(p) * 100, 1)
                for cls, p in zip(model.classes_, proba)
            },
        })

    counts = {"positive": 0, "negative": 0, "neutral": 0}
    for r in results:
        counts[r["sentiment"]] += 1

    total = len(results)
    return {
        "results":   results,
        "summary":   counts,
        "total":     total,
        "breakdown": {k: f"{round(v/total*100,1)}%" for k, v in counts.items()},
    }


@app.post("/retrain")
def retrain():
    """Retrain on data/reviews.csv — call after dropping in a new dataset."""
    global model, vectorizer, metrics
    if not DATA_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="No dataset at data/reviews.csv — add your CSV first"
        )
    model, vectorizer, metrics = train_from_csv(DATA_PATH)
    return {"status": "retrained", "metrics": metrics}
