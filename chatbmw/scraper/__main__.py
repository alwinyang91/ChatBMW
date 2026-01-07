"""
Entry point for running the scraper as a module.

Usage:
    python -m chatbmw.scraper
    python -m chatbmw.scraper --help
    python -m chatbmw.scraper --detail --format jsonl
"""

from .cli import main

if __name__ == "__main__":
    main()
