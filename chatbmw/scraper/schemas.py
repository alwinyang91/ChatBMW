"""
Data Schemas for BMW PressClub Scraper

This module contains data classes for articles and conversion utilities.
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


@dataclass
class Article:
    """
    Data class representing a BMW press article.

    This class stores all information about an article that can be used
    for LLM fine-tuning datasets.

    Attributes:
        title: The headline of the article
        date: Publication date
        type: Type of article (Press Release, Press Kit, Speech, Fact & Figures)
        summary: Brief description or excerpt
        tags: List of category tags
        url: Link to the full article
        content: Full article text (when fetched)
    """
    title: str
    date: str
    type: str = ""
    summary: str = ""
    tags: list[str] = field(default_factory=list)
    url: str = ""
    content: str = ""

    def to_dict(self) -> dict[str, Any]:
        """
        Convert Article to dictionary.

        Returns:
            Dictionary representation of the article
        """
        return asdict(self)


@dataclass
class ArticleCollection:
    """
    Data class representing a collection of BMW press articles.

    This class provides methods to convert articles to raw JSON and Alpaca JSON formats.

    Attributes:
        articles: List of Article objects
        source: Source URL of the articles
        scraped_at: Timestamp when articles were scraped
    """
    articles: list[Article] = field(default_factory=list)
    source: str = "https://www.press.bmwgroup.com/global/article"
    scraped_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_raw_json(self) -> dict[str, Any]:
        """
        Convert ArticleCollection to raw JSON dictionary.

        Output format:
        {
            "scraped_at": "2025-12-17T...",
            "source": "https://www.press.bmwgroup.com/global/article",
            "count": 50,
            "articles": [...]
        }

        Returns:
            Dictionary with metadata and articles
        """
        return {
            "scraped_at": self.scraped_at,
            "source": self.source,
            "count": len(self.articles),
            "articles": [article.to_dict() for article in self.articles],
        }

    def to_alpaca_json(
        self,
        tasks: Optional[list[str]] = None
    ) -> list[dict[str, Any]]:
        """
        Convert articles to Alpaca JSON format for LLM training.

        Alpaca format consists of:
        - instruction: The task description
        - input: The input text (article title and content)
        - output: The expected output (summary, tags, etc.)

        Args:
            tasks: List of tasks to generate. Available tasks:
                - 'summarization': Generate article summaries
                - 'tag_extraction': Extract tags/categories
                - 'type_classification': Classify article type
                - 'title_generation': Generate article titles
                Default is None, which includes all four tasks.

        Returns:
            List of dictionaries in Alpaca format
        """
        # Default to all tasks if not specified
        if tasks is None:
            tasks = ['summarization', 'title_generation', 'tag_extraction', 'type_classification']

        # Normalize task names to lowercase for case-insensitive matching
        tasks = [task.lower() for task in tasks]

        alpaca_data = []

        for article in self.articles:
            title = article.title
            content = article.content
            summary = article.summary
            tags = article.tags
            type = article.type

            # Skip articles without essential content
            if not title and not content:
                continue

            # Create multiple training examples from each article for different tasks

            # Task 1: Summarization
            if 'summarization' in tasks and summary:
                alpaca_data.append({
                    "instruction": "Summarize the following BMW news article in a concise way.",
                    "input": content,
                    "output": summary
                })

            # Task 2: Title generation
            if 'title_generation' in tasks and title and content:
                # Combine summary and content as input
                if summary and content:
                    title_input = f"{summary}\n\n{content}"
                elif summary:
                    title_input = summary
                else:
                    title_input = content

                alpaca_data.append({
                    "instruction": "Generate a concise and informative title for the following BMW news article.",
                    "input": title_input,
                    "output": title
                })

            # Task 3: Tag extraction/classification
            if 'tag_extraction' in tasks and tags:
                tags_string = ", ".join(tags)
                # Combine title, summary and content as input
                tag_input_parts = []
                if title:
                    tag_input_parts.append(f"Title: {title}")
                if summary:
                    tag_input_parts.append(f"Summary: {summary}")
                if content:
                    tag_input_parts.append(f"Content: {content}")
                tag_input = "\n\n".join(tag_input_parts) if tag_input_parts else content
                
                alpaca_data.append({
                    "instruction": "Extract and list the key topics and categories for the following BMW news article.",
                    "input": tag_input,
                    "output": tags_string
                })

            # Task 4: Article type classification
            if 'type_classification' in tasks and type:
                # Combine title, summary, content and tags as input
                type_input_parts = []
                if title:
                    type_input_parts.append(f"Title: {title}")
                if summary:
                    type_input_parts.append(f"Summary: {summary}")
                if content:
                    type_input_parts.append(f"Content: {content}")
                if tags:
                    tags_string = ", ".join(tags)
                    type_input_parts.append(f"Tags: {tags_string}")
                type_input = "\n\n".join(type_input_parts) if type_input_parts else content
                
                alpaca_data.append({
                    "instruction": "Identify the type of the following BMW news article.",
                    "input": type_input,
                    "output": type
                })

        return alpaca_data

    def save_json(self, filepath: str) -> None:
        """
        Save to JSON file.

        Args:
            filepath: Output file path
        """
        # Ensure output directory exists
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_raw_json(), f, ensure_ascii=False, indent=2)
        print(f"💾 Saved {len(self.articles)} articles to {filepath}")

    def save_alpaca_jsonl(
        self,
        filepath: str,
        tasks: Optional[list[str]] = None
    ) -> None:
        """
        Save to JSONL file in Alpaca format.

        Args:
            filepath: Output file path
            tasks: List of tasks to generate
        """
        # Ensure output directory exists
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        alpaca_data = self.to_alpaca_json(tasks=tasks)
        with open(filepath, "w", encoding="utf-8") as f:
            for item in alpaca_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"💾 Saved {len(alpaca_data)} training samples to {filepath}")

    @classmethod
    def from_json_file(cls, filepath: str) -> "ArticleCollection":
        """
        Load from JSON file.

        Args:
            filepath: Input file path

        Returns:
            ArticleCollection instance
        """
        with open(filepath, "r", encoding="utf-8") as f:
            return cls.from_raw_json(json.load(f))

    @classmethod
    def from_raw_json(cls, raw_json: dict[str, Any]) -> "ArticleCollection":
        """
        Create from raw JSON dictionary.

        Args:
            raw_json: Dictionary with articles data

        Returns:
            ArticleCollection instance
        """
        articles = [
            Article(
                title=article_dict.get('title', ''),
                date=article_dict.get('date', ''),
                type=article_dict.get('type', article_dict.get('article_type', '')),
                summary=article_dict.get('summary', ''),
                tags=article_dict.get('tags', []),
                url=article_dict.get('url', ''),
                content=article_dict.get('content', '')
            )
            for article_dict in raw_json.get('articles', [])
        ]
        return cls(
            articles=articles,
            source=raw_json.get('source', 'https://www.press.bmwgroup.com/global/article'),
            scraped_at=raw_json.get('scraped_at', datetime.now().isoformat())
        )
