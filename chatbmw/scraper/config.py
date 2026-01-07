"""
Configuration Module for BMW PressClub Scraper

This module loads configuration from YAML file.
"""

from pathlib import Path
from typing import Any
import yaml


# Default config file path (project root)
DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"


class Config:
    """
    Configuration loader for the scraper.

    Loads settings from a YAML file and provides easy access.

    Usage:
        from scraper.config import config

        # Access settings
        delay = config.scraper.delay
        base_url = config.urls.base
    """

    def __init__(self, config_path: Path | str | None = None):
        """
        Initialize config from YAML file.

        Args:
            config_path: Path to config file, uses default if None
        """
        self._path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        self._data: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        """Load config from YAML file."""
        if self._path.exists():
            with open(self._path, "r", encoding="utf-8") as f:
                self._data = yaml.safe_load(f) or {}
        else:
            # Use defaults if config file doesn't exist
            self._data = self._get_defaults()

    def _get_defaults(self) -> dict[str, Any]:
        """Return default configuration."""
        return {
            "scraper": {
                "delay": 1.5,
                "timeout": 30.0,
                "max_retries": 3,
                "user_agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                "fetch_details": False,
                "limit": 10,
            },
            "output": {
                "directory": "datasets",
                "format": "json",
            },
            "urls": {
                "base": "https://www.press.bmwgroup.com",
                "article_listing": "https://www.press.bmwgroup.com/global/article",
            },
        }

    def reload(self) -> None:
        """Reload config from file."""
        self._load()

    @property
    def scraper(self) -> "ScraperSettings":
        """Get scraper settings."""
        return ScraperSettings(self._data.get("scraper", {}))

    @property
    def output(self) -> "OutputSettings":
        """Get output settings."""
        return OutputSettings(self._data.get("output", {}))

    @property
    def urls(self) -> "UrlSettings":
        """Get URL settings."""
        return UrlSettings(self._data.get("urls", {}))

    def get(self, key: str, default: Any = None) -> Any:
        """Get a config value by dot-notation key (e.g., 'scraper.delay')."""
        keys = key.split(".")
        value = self._data
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        return value if value is not None else default


class ScraperSettings:
    """Scraper-specific settings."""

    def __init__(self, data: dict[str, Any]):
        self._data = data

    @property
    def delay(self) -> float:
        """Delay between requests in seconds."""
        return float(self._data.get("delay", 1.5))

    @property
    def timeout(self) -> float:
        """HTTP request timeout in seconds."""
        return float(self._data.get("timeout", 30.0))

    @property
    def max_retries(self) -> int:
        """Maximum retry attempts."""
        return int(self._data.get("max_retries", 3))

    @property
    def user_agent(self) -> str:
        """Browser user agent string."""
        return self._data.get(
            "user_agent",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )

    @property
    def fetch_details(self) -> bool:
        """Whether to fetch full article content."""
        return bool(self._data.get("fetch_details", False))

    @property
    def limit(self) -> int | None:
        """Maximum number of articles to scrape (None for no limit)."""
        limit_value = self._data.get("limit")
        if limit_value is None or limit_value == "":
            return None
        return int(limit_value) if limit_value > 0 else None


class OutputSettings:
    """Output-specific settings."""

    def __init__(self, data: dict[str, Any]):
        self._data = data

    @property
    def directory(self) -> str:
        """Default output directory."""
        return self._data.get("directory", "datasets")

    @property
    def subdirectory(self) -> str:
        """Subdirectory for article files."""
        return self._data.get("subdirectory", "")

    @property
    def format(self) -> list[str]:
        """Default output format(s)."""
        fmt = self._data.get("format", "json")
        # Support both single string and list of formats
        if isinstance(fmt, list):
            return fmt
        return [fmt]


class UrlSettings:
    """URL settings."""

    def __init__(self, data: dict[str, Any]):
        self._data = data

    @property
    def base(self) -> str:
        """Base URL for BMW PressClub."""
        return self._data.get("base", "https://www.press.bmwgroup.com")

    @property
    def article_listing(self) -> str:
        """Article listing page URL."""
        return self._data.get("article_listing", "https://www.press.bmwgroup.com/global/article")


# Global config instance
# Import this in other modules: from scraper.config import config
config = Config()
