"""
BMW PressClub Article Scraper Package

This package provides tools for scraping articles from BMW PressClub
and preparing them for LLM fine-tuning.

Usage:
    from scraper import BMWArticleScraper, ArticleCollection

    # Scrape articles
    scraper = BMWArticleScraper()
    articles = scraper.scrape(fetch_details=True)
    scraper.close()

    # Save to files
    collection = ArticleCollection(articles=articles)
    collection.save_json("output.json")
    collection.save_alpaca_jsonl("training.jsonl")

    # Load from file
    collection = ArticleCollection.from_json_file("output.json")
    alpaca_data = collection.to_alpaca_json(tasks=['summarization'])

Command-line usage:
    python -m chatbmw.scraper --help
    python -m chatbmw.scraper --detail -o training.jsonl

Configuration:
    Edit config.yaml in project root to customize settings.
"""

from .config import config
from .schemas import Article, ArticleCollection
from .client import HttpClient
from .parser import HtmlParser
from .cleaner import DataCleaner
from .scraper import BMWArticleScraper
from .cli import main

__version__ = "1.0.0"

__all__ = [
    # Config
    "config",
    # Main classes
    "BMWArticleScraper",
    "Article",
    "ArticleCollection",
    # Components
    "HttpClient",
    "HtmlParser",
    "DataCleaner",
    # CLI
    "main",
]
