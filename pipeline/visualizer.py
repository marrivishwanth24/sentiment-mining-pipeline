"""
visualizer.py
-------------
Sentiment trend visualization and analytics dashboard generation.
Produces charts for sentiment distribution, trends over time, and word frequency.
"""

import logging
import argparse
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

COLORS = {
    'positive': '#4CAF50',
    'negative': '#F44336',
    'neutral': '#9E9E9E'
}

plt.style.use('seaborn-v0_8-whitegrid')


def plot_sentiment_distribution(df: pd.DataFrame, output_path: str = 'output/sentiment_distribution.png'):
    """Bar chart showing overall sentiment breakdown."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    counts = df['sentiment'].value_counts()
    colors = [COLORS.get(s, '#607D8B') for s in counts.index]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(counts.index, counts.values, color=colors, width=0.5, edgecolor='white', linewidth=1.5)

    for bar, count in zip(bars, counts.values):
        pct = count / len(df) * 100
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                f'{count:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_title('Sentiment Distribution', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Sentiment', fontsize=12)
    ax.set_ylabel('Review Count', fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved sentiment distribution chart → {output_path}")


def plot_sentiment_over_time(df: pd.DataFrame, date_col: str = 'date',
                              output_path: str = 'output/sentiment_trends.png'):
    """Line chart showing sentiment trends over time."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if date_col not in df.columns:
        logger.warning(f"Column '{date_col}' not found — skipping time series chart")
        return

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])
    df['month'] = df[date_col].dt.to_period('M')

    monthly = df.groupby(['month', 'sentiment']).size().unstack(fill_value=0)
    monthly.index = monthly.index.to_timestamp()

    fig, ax = plt.subplots(figsize=(12, 6))

    for sentiment in ['positive', 'negative', 'neutral']:
        if sentiment in monthly.columns:
            ax.plot(monthly.index, monthly[sentiment],
                    color=COLORS[sentiment], linewidth=2.5,
                    marker='o', markersize=4, label=sentiment.capitalize())
            ax.fill_between(monthly.index, monthly[sentiment],
                            alpha=0.08, color=COLORS[sentiment])

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45, ha='right')

    ax.set_title('Sentiment Trends Over Time', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Review Count', fontsize=12)
    ax.legend(fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved sentiment trend chart → {output_path}")


def plot_top_words(df: pd.DataFrame, sentiment: str = 'negative',
                   top_n: int = 20, output_path: str = None):
    """Horizontal bar chart of top words for a given sentiment."""
    if output_path is None:
        output_path = f'output/top_words_{sentiment}.png'
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    subset = df[df['sentiment'] == sentiment]['processed_text'].dropna()

    all_words = []
    for text in subset:
        all_words.extend(str(text).split())

    word_counts = Counter(all_words).most_common(top_n)
    words, counts = zip(*word_counts) if word_counts else ([], [])

    fig, ax = plt.subplots(figsize=(9, 6))
    y_pos = range(len(words))

    bars = ax.barh(y_pos, counts, color=COLORS.get(sentiment, '#607D8B'),
                   alpha=0.8, edgecolor='white', linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(words, fontsize=11)
    ax.invert_yaxis()

    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                str(count), va='center', fontsize=10)

    ax.set_title(f'Top {top_n} Words in {sentiment.capitalize()} Reviews',
                 fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Frequency', fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved word frequency chart → {output_path}")


def generate_summary_report(df: pd.DataFrame, metrics: dict = None) -> str:
    """
    Generate a text summary report of sentiment analysis results.
    Suitable for delivering insights to non-technical stakeholders.
    """
    total = len(df)
    counts = df['sentiment'].value_counts()

    lines = [
        "=" * 60,
        "SENTIMENT ANALYSIS SUMMARY REPORT",
        "=" * 60,
        f"\nTotal Reviews Analyzed: {total:,}",
        "\nSentiment Breakdown:",
    ]

    for sentiment in ['positive', 'negative', 'neutral']:
        count = counts.get(sentiment, 0)
        pct = count / total * 100 if total > 0 else 0
        lines.append(f"  {sentiment.capitalize():10} {count:6,}  ({pct:.1f}%)")

    if metrics:
        lines += [
            "\nModel Performance:",
            f"  Accuracy:  {metrics.get('accuracy', 0):.2%}",
            f"  F1-Score:  {metrics.get('f1', 0):.2%}",
            f"  Precision: {metrics.get('precision', 0):.2%}",
            f"  Recall:    {metrics.get('recall', 0):.2%}",
        ]

    lines.append("=" * 60)
    report = "\n".join(lines)
    print(report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize sentiment analysis results')
    parser.add_argument('--input', required=True, help='Processed CSV path')
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    plot_sentiment_distribution(df)
    plot_sentiment_over_time(df)
    for s in ['positive', 'negative']:
        plot_top_words(df, sentiment=s)
    generate_summary_report(df)
