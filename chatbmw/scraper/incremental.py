"""
Incremental Update Module for BMW PressClub Scraper

This module handles incremental updates to avoid re-scraping existing articles.

Logic:
1. Load existing dataset
2. Fetch latest article listing (fast, no details)
3. Find new articles not in existing dataset
4. Fetch details only for new articles
5. Merge and keep latest N articles
"""

import json
import time
from pathlib import Path
from datetime import datetime

from .config import config
from .schemas import Article
from .client import HttpClient
from .parser import HtmlParser
from .cleaner import DataCleaner


def parse_date(date_str: str) -> datetime:
    """
    Parse date string to datetime for sorting.
    """
    # Try DD.MM.YYYY format
    try:
        return datetime.strptime(date_str, "%d.%m.%Y")
    except ValueError:
        pass
    
    # Try other formats
    formats = [
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%B %d, %Y",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    # Return a very old date if parsing fails
    return datetime(1900, 1, 1)


class IncrementalUpdater:
    """
    Handles incremental updates to BMW news dataset.

    Example:
        updater = IncrementalUpdater()
        articles = updater.update(
            existing_file="data/bmw_training_latest.json",
            target_count=1000
        )
    """

    def __init__(self):
        self.client = HttpClient()
        self.cleaner = DataCleaner()

    def load_existing_data(self, filepath: str) -> tuple[list[Article], set[str]]:
        """
        Load existing dataset and extract URLs.

        Returns:
            tuple: (list of articles, set of URLs)
        """
        path = Path(filepath)
        if not path.exists():
            print(f"📂 No existing data found at {filepath}")
            return [], set()

        print(f"📂 Loading existing data from {filepath}")
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            articles = []
            urls = set()

            # Handle both formats: {"articles": [...]} and [...]
            article_list = data.get("articles", data) if isinstance(data, dict) else data

            print(f"   Processing {len(article_list)} articles...", end="", flush=True)
            for item in article_list:
                article = Article(
                    title=item.get("title", ""),
                    date=item.get("date", ""),
                    type=item.get("type", item.get("article_type", "")),
                    summary=item.get("summary", ""),
                    tags=item.get("tags", []),
                    url=item.get("url", ""),
                    content=item.get("content", ""),
                )
                articles.append(article)
                if article.url:
                    urls.add(article.url)

            print(f" ✓")
            print(f"   Loaded {len(articles)} existing articles ({len(urls)} unique URLs)")
            return articles, urls
        except Exception as e:
            print(f" ✗ Error loading file: {e}")
            return [], set()

    def fetch_latest_listing(self, target_count: int, existing_urls: set[str] | None = None, need_expansion: bool = False) -> list[Article]:
        """
        Fetch latest article listing (without details) to find new articles.
        
        Args:
            target_count: Target number of articles to fetch
            existing_urls: Set of existing URLs to skip
            need_expansion: If True, also collect old articles to expand dataset
        """
        existing_urls = existing_urls or set()
        all_articles: list[Article] = []
        all_listed_articles: list[Article] = []  # All articles from listing (including old ones)
        seen_urls: set[str] = set()
        page = 1
        articles_per_page = 20
        max_pages = (target_count // articles_per_page) + 5
        consecutive_old_pages = 0  # Track pages with only old articles

        print(f"📥 Fetching latest article listings...")
        print(f"   Target: {target_count} articles (estimated {max_pages} pages)")
        if need_expansion:
            print(f"   📈 Expansion mode: will collect old articles if needed")

        while page <= max_pages:
            try:
                if page == 1:
                    url = config.urls.article_listing
                else:
                    url = f"{config.urls.article_listing}?page={page}"

                print(f"   📄 Fetching page {page}/{max_pages}...", end="", flush=True)
                html = self.client.get(url)
                parser = HtmlParser(html)
                blocks = parser.find_article_blocks()

                if not blocks:
                    print(f" (no articles found)")
                    break

                new_count = 0
                old_count = 0
                for block in blocks:
                    article = parser.parse_article_block(block)
                    if article and article.url and article.url not in seen_urls:
                        seen_urls.add(article.url)
                        # Check if this URL already exists in existing data
                        if article.url in existing_urls:
                            old_count += 1
                            # If we need expansion, also collect old articles
                            if need_expansion:
                                all_listed_articles.append(article)
                        else:
                            all_articles.append(article)
                            all_listed_articles.append(article)
                            new_count += 1

                print(f" ✓ {new_count} new, {old_count} existing (total: {len(all_articles)}/{target_count})")

                # If we need expansion, continue collecting even old articles
                if need_expansion:
                    # Continue until we have enough articles in total
                    if len(all_listed_articles) >= target_count:
                        print(f"   ✓ Reached target of {target_count} articles (including old ones)")
                        break
                else:
                    # Original logic: stop if we've seen many old articles in a row
                    if old_count > 0 and new_count == 0:
                        consecutive_old_pages += 1
                        if consecutive_old_pages >= 3:
                            print(f"   ⚠️ Found {consecutive_old_pages} consecutive pages with only old articles")
                            print(f"   Stopping early (likely reached end of new content)")
                            break
                    else:
                        consecutive_old_pages = 0

                if new_count == 0 and old_count == 0:
                    print(f"   No articles found on page {page}, stopping")
                    break

                # Check if we have enough new articles (if not in expansion mode)
                if not need_expansion and len(all_articles) >= target_count:
                    print(f"   ✓ Reached target of {target_count} new articles")
                    break

                page += 1
                time.sleep(config.scraper.delay)

            except Exception as e:
                print(f" ✗ Error: {e}")
                print(f"   ⚠️ Stopping at page {page}")
                break

        if need_expansion:
            print(f"   ✅ Collected {len(all_listed_articles)} articles from listing (including old ones)")
            return all_listed_articles[:target_count]
        else:
            print(f"   ✅ Collected {len(all_articles)} articles from listing")
            return all_articles[:target_count]

    def fetch_article_details(self, articles: list[Article]) -> list[Article]:
        """
        Fetch full content for articles.
        """
        print(f"\n📄 Fetching details for {len(articles)} new articles...")

        for i, article in enumerate(articles, 1):
            if not article.url:
                continue

            title_preview = article.title[:40] + "..." if len(article.title) > 40 else article.title
            print(f"   [{i}/{len(articles)}] {title_preview}")

            try:
                html = self.client.get(article.url)
                parser = HtmlParser(html)
                article.content = parser.parse_article_content()
            except Exception as e:
                print(f"      ⚠️ Error: {e}")

            time.sleep(config.scraper.delay)

        return articles

    def update(
        self,
        existing_file: str,
        target_count: int = 1000,
        fetch_details: bool = True,
    ) -> list[Article]:
        """
        Perform incremental update.

        Args:
            existing_file: Path to existing dataset
            target_count: Target number of articles
            fetch_details: Whether to fetch full content for new articles

        Returns:
            list[Article]: Updated list of articles
        """
        print("🔄 Starting incremental update...")
        print(f"   Target: {target_count} articles")

        # Step 1: Load existing data
        existing_articles, existing_urls = self.load_existing_data(existing_file)

        # Check if we need to expand the dataset
        need_expansion = len(existing_articles) < target_count
        if need_expansion:
            print(f"   📈 Need to expand from {len(existing_articles)} to {target_count} articles")

        # Step 2: Fetch latest listing
        print()
        latest_articles = self.fetch_latest_listing(target_count, existing_urls, need_expansion=need_expansion)

        # Step 3: Find new articles and handle expansion
        new_articles = [a for a in latest_articles if a.url not in existing_urls]
        print(f"\n🆕 Found {len(new_articles)} new articles")

        # If no new articles but we need expansion, use articles from listing
        if not new_articles and need_expansion:
            print("   No new articles found, but need to expand dataset...")
            # When need_expansion=True, latest_articles includes all articles from listing
            # (both new and old). We need to merge them properly.
            
            # Create a map of existing articles by URL for quick lookup
            existing_by_url = {a.url: a for a in existing_articles if a.url}
            
            # Collect articles from listing
            # For articles that exist in our dataset, use the existing one (has full content)
            # For articles that don't exist, add them (they might need details fetched)
            articles_from_listing = []
            for article in latest_articles:
                if article.url in existing_by_url:
                    # Use existing article (has full content)
                    articles_from_listing.append(existing_by_url[article.url])
                else:
                    # Article from listing not in our dataset - add it
                    articles_from_listing.append(article)
            
            # Merge: prioritize existing articles, then add new ones from listing
            all_articles = existing_articles + articles_from_listing
            
            # Deduplicate by URL (keep first occurrence, which prioritizes existing articles)
            seen_urls = set()
            unique_articles = []
            for article in all_articles:
                if article.url and article.url not in seen_urls:
                    unique_articles.append(article)
                    seen_urls.add(article.url)
            
            # Sort by date (newest first)
            unique_articles.sort(key=lambda a: parse_date(a.date), reverse=True)
            
            # Keep only target_count articles
            final_articles = unique_articles[:target_count]
            
            # Check if articles from listing need details fetched
            articles_needing_details = [
                a for a in final_articles 
                if a.url and a.url not in existing_by_url and (not a.content or len(a.content.strip()) == 0)
            ]
            
            if articles_needing_details and fetch_details:
                print(f"   📄 Fetching details for {len(articles_needing_details)} articles from listing...")
                articles_needing_details = self.fetch_article_details(articles_needing_details)
                # Update final_articles with fetched details
                articles_by_url = {a.url: a for a in final_articles}
                for article in articles_needing_details:
                    if article.url in articles_by_url:
                        articles_by_url[article.url] = article
                final_articles = list(articles_by_url.values())
                final_articles.sort(key=lambda a: parse_date(a.date), reverse=True)
                final_articles = final_articles[:target_count]
            
            if len(final_articles) < target_count:
                print(f"   ⚠️ Only have {len(final_articles)} articles, need {target_count}")
                print(f"   This might mean there aren't enough articles available on the website")
            else:
                print(f"   ✅ Expanded to {len(final_articles)} articles")
            
            removed_count = len(unique_articles) - len(final_articles)
            if removed_count > 0:
                print(f"🗑️  Removed {removed_count} oldest articles to maintain target count")
            
            print(f"✅ Final dataset: {len(final_articles)} articles")
            new_from_listing = len([a for a in final_articles if a.url not in existing_by_url])
            print(f"   - New articles from listing: {new_from_listing}")
            print(f"   - Existing articles kept: {len(final_articles) - new_from_listing}")
            
            return final_articles

        # If no new articles and no expansion needed, just return existing
        if not new_articles:
            print("   No new articles to fetch!")
            existing_articles.sort(key=lambda a: parse_date(a.date), reverse=True)
            return existing_articles[:target_count]

        # Step 4: Fetch details for new articles only
        if fetch_details:
            new_articles = self.fetch_article_details(new_articles)

        # Step 5: Clean new articles
        print("🧹 Cleaning new articles...")
        new_articles = [self.cleaner.clean_article(a) for a in new_articles]
        new_articles, _ = self.cleaner.filter_valid_articles(new_articles, track_filtered=False)

        # Step 6: Merge and sort by date (newest first)
        print("🔀 Merging datasets...")
        all_articles = new_articles + existing_articles

        # Deduplicate by URL
        seen_urls = set()
        unique_articles = []
        for article in all_articles:
            if article.url and article.url not in seen_urls:
                unique_articles.append(article)
                seen_urls.add(article.url)

        # Sort by date (newest first)
        unique_articles.sort(key=lambda a: parse_date(a.date), reverse=True)

        # Keep only target_count articles
        final_articles = unique_articles[:target_count]

        removed_count = len(unique_articles) - len(final_articles)
        if removed_count > 0:
            print(f"🗑️  Removed {removed_count} oldest articles to maintain target count")

        print(f"✅ Final dataset: {len(final_articles)} articles")
        print(f"   - New articles added: {len(new_articles)}")
        print(f"   - Old articles kept: {len(final_articles) - len(new_articles)}")

        return final_articles

    def close(self):
        """Close the HTTP client."""
        self.client.close()
