# Sentiment Analysis & Mining Pipeline

An end-to-end **Python machine learning pipeline** for large-scale product review sentiment analysis. Automates data collection, preprocessing, classification, and visualization to deliver actionable business insights.

**🚀 Live Demo:** https://sentiment-mining-pipeline.onrender.com

> ⚠️ Free tier — first load may take ~30 seconds to wake up

---

## Features

- **Automated data collection** — Web scraping with BeautifulSoup and Selenium
- **NLP preprocessing** — Tokenization, stopword removal, stemming, TF-IDF vectorization
- **ML classification** — Logistic Regression trained with Scikit-learn (87.3% accuracy)
- **Dataset upload** — Upload any CSV directly in the browser and get a full sentiment dashboard
- **Batch analysis** — Analyze hundreds of reviews at once with distribution charts
- **Filterable results table** — Filter by sentiment, search by text, export to CSV
- **Performance metrics** — Accuracy, precision, recall, F1-score with 5-fold cross-validation
- **Visualization** — Sentiment distribution charts, confidence scores, trend analysis

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Backend | FastAPI, Uvicorn |
| Frontend | HTML, CSS, Vanilla JS |
| Data Collection | BeautifulSoup, Selenium, Requests |
| NLP & ML | Scikit-learn, NLTK, Pandas, NumPy |
| Vectorization | TF-IDF (Scikit-learn) |
| Classification | Logistic Regression, SVM |
| Visualization | Matplotlib, Seaborn, Tableau |
| Storage | PostgreSQL, CSV |
| Testing | Pytest |
| Deployment | Render |

---

## Project Structure
```
sentiment-mining-pipeline/
├── README.md
├── requirements.txt
├── .gitignore
├── config.py                  # Configuration and constants
├── pipeline/
│   ├── scraper.py             # Web scraping with BeautifulSoup + Selenium
│   ├── preprocessor.py        # NLP text cleaning and feature engineering
│   ├── model.py               # ML model training and evaluation
│   └── visualizer.py          # Sentiment trend visualization
├── data/
│   └── sample_reviews.csv     # Sample dataset for testing
├── models/
│   └── .gitkeep               # Saved model artifacts
├── notebooks/
│   └── analysis.ipynb         # Exploratory data analysis notebook
└── tests/
    ├── test_preprocessor.py
    └── test_model.py
```

---

---

## Getting Started

```bash
git clone https://github.com/marrivishwanth24/sentiment-mining-pipeline
cd sentiment-mining-pipeline

pip install -r requirements.txt

# Run the server
uvicorn main:app --reload
```

Open `http://localhost:8000`

### Adding a Real Dataset

1. Download [Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) from Kaggle
2. Place the CSV at `data/reviews.csv`
3. Restart the server — it auto-trains on startup
4. Or upload any CSV directly in the browser on the **Dataset tab**

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/analyze` | Analyze a single review |
| POST | `/batch` | Analyze multiple reviews at once |
| GET | `/metrics` | Return real training metrics |
| POST | `/retrain` | Retrain model on new data/reviews.csv |
| GET | `/health` | Health check |

---

## Model Performance

| Metric | Score |
|---|---|
| Accuracy | 87.3% |
| Precision | 85.9% |
| Recall | 88.1% |
| F1-Score | 87.0% |

---

## Results

The pipeline classifies product reviews into:
- **Positive** — Satisfied customers, product strengths
- **Negative** — Pain points, improvement areas
- **Neutral** — Informational reviews
