"""
Data Cleaner Module for BMW PressClub Scraper

This module handles text normalization, deduplication, and quality filtering.
"""

import re

from .schemas import Article


class DataCleaner:
    """
    Cleans and processes scraped data for LLM fine-tuning.

    Handles text normalization, deduplication, and quality filtering.

    This class is independent and can be used with any text data.
    """

    # Minimum content length for valid articles (content + summary)
    MIN_CONTENT_LENGTH = 50
    
    # Minimum title length
    MIN_TITLE_LENGTH = 10

    def clean_article(self, article: Article) -> Article:
        """
        Clean a single article.

        Args:
            article: Article to clean

        Returns:
            Article: Cleaned article
        """
        # Clean title
        article.title = self._normalize_text(article.title)
        
        # Clean summary
        article.summary = self._normalize_text(article.summary)
        
        # Clean content
        article.content = self._normalize_content(article.content)
        
        # Clean tags
        article.tags = self._clean_tags(article.tags)
        
        return article

    def filter_valid_articles(
        self, 
        articles: list[Article], 
        track_filtered: bool = False
    ) -> tuple[list[Article], list[dict]]:
        """
        Filter articles to keep only valid ones for training.

        Args:
            articles: List of articles to filter
            track_filtered: If True, return information about filtered articles

        Returns:
            tuple[list[Article], list[dict]]: Filtered list of valid articles and 
                list of filtered articles with reasons (if track_filtered=True)
        """
        valid = []
        filtered = []
        seen_titles = set()
        
        for article in articles:
            reason = None
            
            # Skip if title is too short
            if len(article.title) < self.MIN_TITLE_LENGTH:
                reason = f"Title too short ({len(article.title)} < {self.MIN_TITLE_LENGTH} chars)"
                if track_filtered:
                    filtered.append({
                        "title": article.title,
                        "url": article.url,
                        "reason": reason
                    })
                continue
            
            # Skip duplicates
            title_key = article.title.lower().strip()
            if title_key in seen_titles:
                reason = "Duplicate title (case-insensitive)"
                if track_filtered:
                    filtered.append({
                        "title": article.title,
                        "url": article.url,
                        "reason": reason
                    })
                continue
            seen_titles.add(title_key)
            
            # Skip if no meaningful content
            content_length = len(article.content) + len(article.summary)
            if content_length < self.MIN_CONTENT_LENGTH:
                reason = f"Insufficient content ({content_length} < {self.MIN_CONTENT_LENGTH} chars)"
                if track_filtered:
                    filtered.append({
                        "title": article.title,
                        "url": article.url,
                        "reason": reason,
                        "content_length": content_length
                    })
                continue
            
            valid.append(article)
        
        if track_filtered:
            return valid, filtered
        return valid, []

    def deduplicate(
        self, 
        articles: list[Article], 
        track_filtered: bool = False
    ) -> tuple[list[Article], list[dict]]:
        """
        Remove duplicate articles based on title similarity.

        Args:
            articles: List of articles
            track_filtered: If True, return information about filtered articles

        Returns:
            tuple[list[Article], list[dict]]: Deduplicated list and list of 
                filtered duplicates with reasons (if track_filtered=True)
        """
        seen = set()
        unique = []
        filtered = []
        
        for article in articles:
            # Create a normalized key for comparison
            key = re.sub(r'\W+', '', article.title.lower())
            
            if key not in seen:
                seen.add(key)
                unique.append(article)
            else:
                if track_filtered:
                    filtered.append({
                        "title": article.title,
                        "url": article.url,
                        "reason": "Duplicate (normalized title match)"
                    })
        
        if track_filtered:
            return unique, filtered
        return unique, []

    def _normalize_text(self, text: str) -> str:
        """
        Normalize text by removing extra whitespace and special characters.

        Args:
            text: Text to normalize

        Returns:
            str: Normalized text
        """
        if not text:
            return ""
        
        # Replace multiple whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Remove strange characters
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        
        return text

    def _normalize_content(self, content: str) -> str:
        """
        Normalize article content for training.

        Args:
            content: Article content

        Returns:
            str: Normalized content
        """
        if not content:
            return ""
        
        # Split into paragraphs
        paragraphs = content.split("\n\n")
        
        # Clean each paragraph
        cleaned = []
        for p in paragraphs:
            p = self._normalize_text(p)
            # Skip paragraphs that contain social media links or website URLs
            if self._contains_social_media_links(p):
                continue
            if len(p) > 20:
                cleaned.append(p)
        
        return "\n\n".join(cleaned)
    
    def _contains_social_media_links(self, text: str) -> bool:
        """
        Check if text contains social media links or website URLs.

        Args:
            text: Text to check

        Returns:
            bool: True if contains social media links
        """
        if not text:
            return False
        
        # Patterns for social media links and website URLs
        patterns = [
            r'www\.bmwgroup\.com',
            r'linkedin\.com/company/bmw',
            r'youtube\.com/bmwgroup',
            r'instagram\.com/bmwgroup',
            r'facebook\.com/bmwgroup',
            r'x\.com/bmwgroup',
            r'twitter\.com/bmwgroup',
            r'LinkedIn:\s*http',
            r'YouTube:\s*http',
            r'Instagram:\s*http',
            r'Facebook:\s*http',
            r'X:\s*http',
        ]
        
        text_lower = text.lower()
        for pattern in patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True
        
        return False

    def _clean_tags(self, tags: list[str]) -> list[str]:
        """
        Clean and normalize tags.

        Args:
            tags: List of tags

        Returns:
            list[str]: Cleaned tags
        """
        cleaned = []
        seen = set()
        
        for tag in tags:
            # Normalize tag
            tag = self._normalize_text(tag)
            
            # Skip empty or too long tags
            if not tag or len(tag) > 50:
                continue
            
            # Skip duplicates (case-insensitive)
            tag_lower = tag.lower()
            if tag_lower in seen:
                continue
            seen.add(tag_lower)
            
            cleaned.append(tag)
        
        return cleaned
