# Sentiment Analysis & Mining Pipeline

An end-to-end **Python machine learning pipeline** for large-scale product review sentiment analysis. Automates data collection, preprocessing, classification, and visualization to deliver actionable business insights.

---

## Features

- **Automated data collection** — Web scraping with BeautifulSoup and Selenium
- **NLP preprocessing** — Tokenization, stopword removal, stemming, TF-IDF vectorization
- **ML classification** — Logistic Regression model trained with Scikit-learn
- **Performance metrics** — Accuracy, precision, recall, F1-score evaluation
- **Visualization** — Sentiment trend dashboards built with Matplotlib and Tableau
- **Scalable pipeline** — Processes thousands of reviews end-to-end automatically

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Data Collection | BeautifulSoup, Selenium, Requests |
| NLP & ML | Scikit-learn, NLTK, Pandas, NumPy |
| Vectorization | TF-IDF (Scikit-learn) |
| Classification | Logistic Regression, SVM |
| Visualization | Matplotlib, Seaborn, Tableau |
| Storage | PostgreSQL, CSV |
| Testing | Pytest |

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

## Getting Started

```bash
git clone https://github.com/marrivishwanth24/sentiment-mining-pipeline
cd sentiment-mining-pipeline

pip install -r requirements.txt
python -m nltk.downloader stopwords wordnet

python pipeline/scraper.py --url "https://example.com/reviews" --pages 10
python pipeline/preprocessor.py --input data/reviews.csv --output data/processed.csv
python pipeline/model.py --input data/processed.csv
python pipeline/visualizer.py --input data/processed.csv
```

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

The pipeline processes product reviews and classifies them into:
- **Positive** — Satisfied customers, product strengths
- **Negative** — Pain points, improvement areas
- **Neutral** — Informational reviews

Sentiment trends are visualized over time to surface actionable insights for product strategy.
