"""
HTML Parser Module for BMW PressClub Scraper

This module handles all HTML parsing logic using BeautifulSoup.
"""

import re

from bs4 import BeautifulSoup, Tag

from .schemas import Article


class HtmlParser:
    """
    Parser for extracting data from HTML content.

    Uses BeautifulSoup for parsing and provides methods specific to
    BMW PressClub page structure.

    This class is responsible for all HTML parsing logic.
    """

    # Date pattern for BMW PressClub dates
    # Example: "Mon Dec 15 09:00:00 CET 2025"
    DATE_PATTERN = re.compile(
        r'(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+'
        r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+'
        r'\d{1,2}\s+\d{2}:\d{2}:\d{2}\s+\w+\s+\d{4}'
    )
    
    # Alternative date pattern: DD.MM.YYYY
    DATE_PATTERN_ALT = re.compile(r'\d{2}\.\d{2}\.\d{4}')

    def __init__(self, html: str):
        """
        Initialize parser with HTML content.

        Args:
            html: Raw HTML string to parse
        """
        self.soup = BeautifulSoup(html, "lxml")

    def find_article_blocks(self) -> list[Tag]:
        """
        Find all article blocks on the page.

        BMW PressClub uses specific HTML structure for article listings.

        HTML Structure:
        <article class="newsfeed clear-after has-image has-media">
          <div class="text">
            <h3><a href="/global/article/detail/...">Title</a></h3>
            <div class="info">
              <h4>
                <span class="date">17.12.2025</span>
                <span class="category">Press Release</span>
              </h4>
            </div>
            <p class="serif">Summary...</p>
          </div>
          <div class="tagging-info tags">
            <div class="clickable tag tag-searchable">Tag1</div>
            ...
          </div>
        </article>

        Returns:
            list[Tag]: List of BeautifulSoup Tag objects representing articles
        """
        # Primary method: Find <article> elements with class "newsfeed"
        article_blocks = self.soup.find_all("article", class_="newsfeed")
        
        if article_blocks:
            return article_blocks
        
        # Fallback: Find by article-detail links and their parent containers
        for link in self.soup.find_all("a", href=re.compile(r"/article/detail/")):
            parent = link.find_parent(["article", "div", "section", "li"])
            if parent and parent not in article_blocks:
                article_blocks.append(parent)
        
        return article_blocks

    def parse_article_block(self, block: Tag) -> Article | None:
        """
        Parse a single article block into an Article object.

        Args:
            block: BeautifulSoup Tag containing article info

        Returns:
            Article | None: Parsed Article object or None if parsing fails
        """
        try:
            # Extract title and URL from <h3><a> tag inside <div class="text">
            # HTML structure: 
            #   <div class="text">
            #     <h3><a href="/global/article/detail/...">Title</a></h3>
            #   </div>
            # Note: There may be empty <a> tags in <div class="image">, so we
            # specifically look for the link inside <h3> or <div class="text">
            
            link = None
            title = ""
            
            # Primary method: Find <h3> first, then get the link inside
            h3_elem = block.find("h3")
            if h3_elem:
                link = h3_elem.find("a", href=re.compile(r"/article/detail/"))
                if link:
                    title = self._clean_text(link.get_text())
            
            # Fallback: Find link in <div class="text">
            if not link or len(title) < 10:
                text_div = block.find("div", class_="text")
                if text_div:
                    link = text_div.find("a", href=re.compile(r"/article/detail/"))
                    if link:
                        title = self._clean_text(link.get_text())
            
            # Last fallback: Find any link with text content
            if not link or len(title) < 10:
                for a in block.find_all("a", href=re.compile(r"/article/detail/")):
                    text = self._clean_text(a.get_text())
                    if len(text) >= 10:
                        link = a
                        title = text
                        break
            
            if not link or len(title) < 10:
                return None

            # Extract URL from the same link
            url = ""
            href = link.get("href", "")
            if href.startswith("/"):
                url = f"https://www.press.bmwgroup.com{href}"
            elif href.startswith("http"):
                url = href

            # Extract date from <span class="date">
            date = ""
            date_elem = block.find("span", class_="date")
            if date_elem:
                date = self._clean_text(date_elem.get_text())
            else:
                # Fallback to regex extraction
                date = self._extract_date(block.get_text())

            # Extract article type from <span class="category">
            # Available types: Press Release, Press Kit, Speech, Fact & Figures
            article_type = ""
            category_elem = block.find("span", class_="category")
            if category_elem:
                raw_type = self._clean_text(category_elem.get_text())
                # Normalize article type
                if "press release" in raw_type.lower():
                    article_type = "Press Release"
                elif "press kit" in raw_type.lower():
                    article_type = "Press Kit"
                elif "speech" in raw_type.lower():
                    article_type = "Speech"
                elif "fact" in raw_type.lower():
                    article_type = "Fact & Figures"

            # Extract summary from <p class="serif">
            summary = ""
            summary_elem = block.find("p", class_="serif")
            if summary_elem:
                summary = self._clean_text(summary_elem.get_text())
            else:
                # Fallback: Look for any paragraph with content
                for elem in block.find_all("p"):
                    text = self._clean_text(elem.get_text())
                    if 30 < len(text) < 500 and text != title:
                        summary = text
                        break

            # Extract tags from <div class="tagging-info">
            # HTML: <div class="tagging-info"><div class="clickable tag">Tag1</div>...</div>
            tags = []
            tagging_info = block.find("div", class_="tagging-info")
            if tagging_info:
                tag_elems = tagging_info.find_all("div", class_=re.compile(r"clickable.*tag|tag.*clickable"))
                seen = set()
                for elem in tag_elems:
                    if "separator" in elem.get("class", []):
                        continue
                    tag_text = self._clean_text(elem.get_text())
                    tag_lower = tag_text.lower()
                    if tag_text and 2 < len(tag_text) < 50 and tag_lower not in seen:
                        seen.add(tag_lower)
                        tags.append(tag_text)

            return Article(
                title=title,
                date=date,
                type=article_type,
                summary=summary,
                tags=tags,
                url=url,
            )

        except Exception as e:
            print(f"  ⚠️ Error parsing article block: {e}")
            return None

    def parse_article_content(self) -> str:
        """
        Parse full article content from an article detail page.

        Based on BMW PressClub HTML structure:
        - div.article-detail: contains h1 (title), h2.teaser (subtitle)
        - div#article-text: main article body text
        - div.text.co2_summary: CO2 emissions info

        Returns:
            str: Cleaned article content
        """
        content_parts = []
        
        # 1. Extract main article text from div#article-text
        article_text_elem = self.soup.select_one("div#article-text")
        if article_text_elem:
            # Remove unwanted elements (buttons, scripts, etc.)
            for tag in article_text_elem.find_all(["script", "style", "button", "nav"]):
                tag.decompose()
            
            # Extract paragraphs from article text
            for p in article_text_elem.find_all("p"):
                text = self._clean_text(p.get_text())
                if len(text) > 20 and not self._is_noise(text):
                    content_parts.append(text)
            
            # Also check for direct text in divs (some articles use divs instead of p)
            if not content_parts or len(content_parts) <= 1:
                for div in article_text_elem.find_all("div", recursive=False):
                    text = self._clean_text(div.get_text())
                    if len(text) > 30 and not self._is_noise(text):
                        content_parts.append(text)
        
        # 2. Extract CO2 emissions summary if present
        co2_elem = self.soup.select_one("div.text.co2_summary")
        if co2_elem:
            co2_text = self._clean_text(co2_elem.get_text())
            if co2_text and len(co2_text) > 20:
                content_parts.append(f"CO2 Emissions & Consumption: {co2_text}")
        
        # 3. Fallback: if no content found, try broader selectors
        if not content_parts:
            # Try div.article-detail or div.content-left
            fallback_selectors = ["div.article-detail", "div.content-left", "div.content-block"]
            for selector in fallback_selectors:
                content_elem = self.soup.select_one(selector)
                if content_elem:
                    for tag in content_elem.find_all(["script", "style", "button", "nav", "aside"]):
                        tag.decompose()
                    for p in content_elem.find_all("p"):
                        text = self._clean_text(p.get_text())
                        if len(text) > 30 and not self._is_noise(text):
                            content_parts.append(text)
                    if content_parts:
                        break
        
        return "\n\n".join(content_parts)

    def _extract_date(self, text: str) -> str:
        """
        Extract date from text.

        Args:
            text: Text that may contain a date

        Returns:
            str: Extracted date or empty string
        """
        # Try main pattern first
        match = self.DATE_PATTERN.search(text)
        if match:
            return match.group(0)
        
        # Try alternative pattern
        match = self.DATE_PATTERN_ALT.search(text)
        if match:
            return match.group(0)
        
        return ""

    def _clean_text(self, text: str) -> str:
        """
        Clean text by normalizing whitespace.

        Args:
            text: Text to clean

        Returns:
            str: Cleaned text
        """
        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _is_noise(self, text: str) -> bool:
        """
        Check if text is noise (contact info, navigation, etc.).

        Args:
            text: Text to check

        Returns:
            bool: True if text is noise
        """
        noise_patterns = [
            r"Tel:\s*\+?[\d\-\s]+",
            r"send an e-mail",
            r"@.*\.(com|de|org)",
            r"Click here",
            r"Show all",
            r"Show more",
            r"CO2 emission",
            r"Media Set",
            r"Attachments\(",
            r"Photos\(",
        ]
        
        text_lower = text.lower()
        for pattern in noise_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return False
