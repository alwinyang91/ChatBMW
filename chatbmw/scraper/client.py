"""
HTTP Client Module for BMW PressClub Scraper

This module handles all HTTP communication with retry logic and rate limiting.
"""

import time
import httpx
from .config import config


class HttpClient:
    """
    HTTP client for making web requests.

    Handles all HTTP communication with retry logic and rate limiting.

    This class is independent and can be reused for other scrapers.

    Example:
        client = HttpClient()
        html = client.get("https://example.com")
        client.close()
    """

    def __init__(self):
        """
        Initialize HTTP client with configuration from config.yaml.
        """
        self._client = httpx.Client(
            headers={
                "User-Agent": config.scraper.user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
            },
            follow_redirects=True,
            timeout=config.scraper.timeout,
        )

    def get(self, url: str) -> str:
        """
        Fetch HTML content from URL with retry logic.

        Implements exponential backoff for failed requests.

        Args:
            url: The URL to fetch

        Returns:
            str: HTML content as string

        Raises:
            httpx.HTTPError: If all retries fail
        """
        last_error = None
        
        for attempt in range(config.scraper.max_retries):
            try:
                response = self._client.get(url)
                response.raise_for_status()
                return response.text
            except httpx.HTTPError as e:
                last_error = e
                # Exponential backoff: 1s, 2s, 4s...
                wait_time = 2 ** attempt
                print(f"  ⚠️ Request failed (attempt {attempt + 1}), retrying in {wait_time}s...")
                time.sleep(wait_time)
        
        raise last_error  # type: ignore

    def close(self):
        """
        Close the HTTP client and release resources.
        """
        self._client.close()
