"""
Main BMW Article Scraper Class

This module contains the main scraper class that orchestrates all components.
"""

import time
from datetime import datetime
from pathlib import Path

from .config import config
from .schemas import Article, ArticleCollection
from .client import HttpClient
from .parser import HtmlParser
from .cleaner import DataCleaner


class BMWArticleScraper:
    """
    Main scraper for BMW PressClub articles.

    Orchestrates all components (HTTP client, parser, cleaner, exporter)
    to scrape and process articles from the BMW PressClub website.

    Example:
        scraper = BMWArticleScraper()
        articles = scraper.scrape(fetch_details=True, limit=50)
        scraper.export(articles, "output.json", format="json")
        scraper.close()
    """

    def __init__(self):
        """
        Initialize the scraper with configuration from config.yaml.
        """
        self.client = HttpClient()
        self.cleaner = DataCleaner()

    def scrape(
        self,
        fetch_details: bool | None = None,
        limit: int | None = None,
    ) -> list[Article]:
        """
        Scrape articles from BMW PressClub.

        Args:
            fetch_details: Whether to fetch full article content (default: from config)
            limit: Maximum number of articles to return (default: from config)

        Returns:
            list[Article]: List of scraped articles
        """
        # Use config values if not explicitly provided
        if fetch_details is None:
            fetch_details = config.scraper.fetch_details
        if limit is None:
            limit = config.scraper.limit
        print(f"🚗 Starting BMW PressClub scraper...")
        print(f"   Target: {config.urls.article_listing}")
        if limit:
            print(f"   Target count: {limit} articles")
        
        # Collect with buffer to account for filtering (collect 20% more or at least 5 extra)
        collection_limit = limit
        if limit:
            buffer = max(int(limit * 0.2), 5)  # 20% or 5, whichever is larger
            collection_limit = limit + buffer
            print(f"   Collecting {collection_limit} articles (buffer for filtering)")
        
        # Fetch and parse article listing with pagination
        articles = self._scrape_listing_paginated(limit=collection_limit)
        
        print(f"📰 Collected {len(articles)} articles from listing")
        
        # Fetch full content if requested
        if fetch_details:
            articles = self._fetch_article_details(articles)
        
        # Clean articles
        print("🧹 Cleaning data...")
        articles = [self.cleaner.clean_article(a) for a in articles]
        
        # Filter with tracking
        articles, filtered_valid = self.cleaner.filter_valid_articles(articles, track_filtered=True)
        articles, filtered_dupes = self.cleaner.deduplicate(articles, track_filtered=True)
        
        # Log filtered articles
        all_filtered = filtered_valid + filtered_dupes
        if all_filtered:
            print(f"   ⚠️  Filtered out {len(all_filtered)} invalid/duplicate articles:")
            for filtered in all_filtered:
                title_preview = filtered["title"][:60] + "..." if len(filtered["title"]) > 60 else filtered["title"]
                print(f"      - {title_preview}")
                print(f"        Reason: {filtered['reason']}")
            
            # Write to log file
            log_path = self._log_filtered_articles(all_filtered)
            print(f"   📝 Filtered articles logged to: {log_path}")
        
        # Apply limit after filtering to ensure we get exactly the requested number
        if limit and len(articles) > limit:
            articles = articles[:limit]
        
        print(f"✅ Final dataset: {len(articles)} articles")
        
        return articles

    def _scrape_listing_paginated(self, limit: int | None = None) -> list[Article]:
        """
        Scrape article listings with pagination support.

        BMW PressClub supports `?page=N` parameter for pagination.

        Args:
            limit: Maximum number of articles to collect

        Returns:
            list[Article]: List of articles from all pages
        """
        all_articles: list[Article] = []
        seen_urls: set[str] = set()  # Track URLs to avoid duplicates
        page = 1
        max_pages = 100  # Safety limit
        articles_per_page = 20  # Approximate
        
        # Calculate how many pages we need
        if limit:
            estimated_pages = (limit // articles_per_page) + 2  # +2 for buffer
            max_pages = min(estimated_pages, max_pages)
        
        print(f"📥 Fetching article listings (up to {max_pages} pages)...")
        
        while page <= max_pages:
            try:
                # Build URL with page parameter
                if page == 1:
                    url = config.urls.article_listing
                else:
                    url = f"{config.urls.article_listing}?page={page}"
                
                print(f"   Page {page}: {url}")
                
                html = self.client.get(url)
                parser = HtmlParser(html)
                blocks = parser.find_article_blocks()
                
                if not blocks:
                    print(f"   No more articles found on page {page}")
                    break
                
                # Parse articles from this page
                new_count = 0
                for block in blocks:
                    article = parser.parse_article_block(block)
                    if article and article.url and article.url not in seen_urls:
                        all_articles.append(article)
                        seen_urls.add(article.url)
                        new_count += 1
                
                print(f"   Found {new_count} new articles (total: {len(all_articles)})")
                
                # Check if we have enough
                if limit and len(all_articles) >= limit:
                    print(f"   Reached target of {limit} articles")
                    all_articles = all_articles[:limit]
                    break
                
                # Check if this page had no new articles (end of content)
                if new_count == 0:
                    print(f"   No new articles on page {page}, stopping")
                    break
                
                page += 1
                
                # Rate limiting between pages
                if page <= max_pages:
                    time.sleep(config.scraper.delay)
                
            except Exception as e:
                print(f"   ⚠️ Error on page {page}: {e}")
                break
        
        return all_articles

    def _scrape_listing(self) -> list[Article]:
        """
        Scrape article listings from the main page (single page only).

        Returns:
            list[Article]: List of articles from listing
        """
        print(f"📥 Fetching article listing...")
        
        try:
            html = self.client.get(config.urls.article_listing)
            print(f"   Received {len(html):,} bytes")
            
            parser = HtmlParser(html)
            blocks = parser.find_article_blocks()
            print(f"   Found {len(blocks)} article blocks")
            
            articles = []
            for block in blocks:
                article = parser.parse_article_block(block)
                if article:
                    articles.append(article)
            
            return articles
            
        except Exception as e:
            print(f"❌ Error fetching listing: {e}")
            return []

    def _fetch_article_details(self, articles: list[Article]) -> list[Article]:
        """
        Fetch full content for each article.

        Args:
            articles: List of articles to fetch details for

        Returns:
            list[Article]: Articles with content filled in
        """
        print(f"\n📄 Fetching details for {len(articles)} articles...")
        
        for i, article in enumerate(articles, 1):
            if not article.url:
                continue
            
            # Print progress
            title_preview = article.title[:40] + "..." if len(article.title) > 40 else article.title
            print(f"   [{i}/{len(articles)}] {title_preview}")
            
            try:
                html = self.client.get(article.url)
                parser = HtmlParser(html)
                article.content = parser.parse_article_content()
                
            except Exception as e:
                print(f"      ⚠️ Error: {e}")
            
            # Rate limiting
            time.sleep(config.scraper.delay)
        
        return articles

    def export(
        self,
        articles: list[Article],
        filepath: str,
        format: str | None = None,
    ) -> None:
        """
        Export articles to file.

        Args:
            articles: List of articles to export
            filepath: Output file path
            format: 'json' or 'jsonl' (default: auto-detect from extension)
        """
        collection = ArticleCollection(articles=articles)
        
        # Auto-detect format from extension if not specified
        if format is None:
            format = "jsonl" if filepath.endswith(".jsonl") else "json"
        
        if format == "json":
            collection.save_json(filepath)
        else:
            collection.save_alpaca_jsonl(filepath)

    def _log_filtered_articles(self, filtered_articles: list[dict]) -> str:
        """
        Write filtered articles to a log file.

        Args:
            filtered_articles: List of filtered article information

        Returns:
            str: Path to the log file
        """
        # Create output directory if it doesn't exist
        output_dir = Path(config.output.directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate log filename with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_filename = f"filtered_articles_{timestamp}.log"
        log_path = output_dir / log_filename
        
        # Write to log file
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("BMW PressClub Scraper - Filtered Articles Log\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total filtered: {len(filtered_articles)}\n")
            f.write("=" * 80 + "\n\n")
            
            for i, filtered in enumerate(filtered_articles, 1):
                f.write(f"[{i}] {filtered['title']}\n")
                f.write(f"     URL: {filtered.get('url', 'N/A')}\n")
                f.write(f"     Reason: {filtered['reason']}\n")
                if 'content_length' in filtered:
                    f.write(f"     Content Length: {filtered['content_length']} chars\n")
                f.write("\n")
        
        return str(log_path)

    def close(self):
        """
        Close the scraper and release resources.
        """
        self.client.close()
