"""
scraper.py
----------
Web scraping module for collecting product reviews at scale.
Uses BeautifulSoup for static pages and Selenium for JavaScript-rendered content.
"""

import time
import csv
import logging
import argparse
from datetime import datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ReviewScraper:
    """
    Scrapes product reviews from e-commerce pages.
    Supports both static (BeautifulSoup) and dynamic (Selenium) scraping.
    """

    def __init__(self, headless: bool = True, delay: float = 1.5):
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; ResearchBot/1.0)'
        })
        self.driver = None
        self.headless = headless

    def _init_driver(self):
        """Initialize Selenium Chrome driver for dynamic content."""
        options = Options()
        if self.headless:
            options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-blink-features=AutomationControlled')
        self.driver = webdriver.Chrome(options=options)
        logger.info("Selenium driver initialized")

    def scrape_static(self, url: str, selectors: dict) -> list[dict]:
        """
        Scrape reviews from a static HTML page using BeautifulSoup.

        Args:
            url: Target URL to scrape
            selectors: CSS selectors for review elements
                       e.g. {'container': '.review', 'text': '.review-body', 'rating': '.star-rating'}

        Returns:
            List of review dicts with text, rating, date fields
        """
        reviews = []
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            containers = soup.select(selectors.get('container', '.review'))
            logger.info(f"Found {len(containers)} reviews on {url}")

            for container in containers:
                review = {}

                text_el = container.select_one(selectors.get('text', '.review-text'))
                review['text'] = text_el.get_text(strip=True) if text_el else ''

                rating_el = container.select_one(selectors.get('rating', '.rating'))
                review['rating'] = rating_el.get_text(strip=True) if rating_el else ''

                date_el = container.select_one(selectors.get('date', '.review-date'))
                review['date'] = date_el.get_text(strip=True) if date_el else ''

                review['url'] = url
                review['scraped_at'] = datetime.now().isoformat()

                if review['text']:
                    reviews.append(review)

        except requests.RequestException as e:
            logger.error(f"Failed to scrape {url}: {e}")

        time.sleep(self.delay)
        return reviews

    def scrape_dynamic(self, url: str, next_button_selector: str, max_pages: int = 5) -> list[dict]:
        """
        Scrape reviews from a JavaScript-rendered page using Selenium.
        Handles pagination by clicking through pages.

        Args:
            url: Target URL
            next_button_selector: CSS selector for the "next page" button
            max_pages: Maximum number of pages to scrape

        Returns:
            List of review dicts
        """
        if not self.driver:
            self._init_driver()

        reviews = []
        self.driver.get(url)
        time.sleep(2)

        for page in range(max_pages):
            logger.info(f"Scraping page {page + 1}")

            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            page_reviews = self._parse_reviews(soup)
            reviews.extend(page_reviews)

            try:
                next_btn = WebDriverWait(self.driver, 5).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, next_button_selector))
                )
                next_btn.click()
                time.sleep(self.delay)
            except Exception:
                logger.info(f"No more pages after {page + 1}")
                break

        return reviews

    def _parse_reviews(self, soup: BeautifulSoup) -> list[dict]:
        """Parse review elements from a BeautifulSoup object."""
        reviews = []
        for el in soup.select('.review, [data-hook="review"]'):
            text = el.get_text(strip=True)
            if text and len(text) > 20:
                reviews.append({
                    'text': text[:2000],
                    'scraped_at': datetime.now().isoformat()
                })
        return reviews

    def save_to_csv(self, reviews: list[dict], output_path: str):
        """Save scraped reviews to CSV file."""
        if not reviews:
            logger.warning("No reviews to save")
            return

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fieldnames = reviews[0].keys()

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(reviews)

        logger.info(f"Saved {len(reviews)} reviews to {output_path}")

    def close(self):
        """Clean up Selenium driver."""
        if self.driver:
            self.driver.quit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Scrape product reviews')
    parser.add_argument('--url', required=True, help='Target URL to scrape')
    parser.add_argument('--pages', type=int, default=5, help='Max pages to scrape')
    parser.add_argument('--output', default='data/reviews.csv', help='Output CSV path')
    args = parser.parse_args()

    scraper = ReviewScraper(headless=True)
    try:
        reviews = scraper.scrape_static(args.url, {})
        scraper.save_to_csv(reviews, args.output)
        print(f"Scraped {len(reviews)} reviews → {args.output}")
    finally:
        scraper.close()
