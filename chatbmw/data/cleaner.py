"""
Data cleaning utilities for BMW articles dataset.
"""
import re
import json
from pathlib import Path
from typing import Union


def clean_text(text: str) -> str:
    """
    Clean text by removing URLs, contact info, boilerplate, and normalizing whitespace.
    
    Args:
        text: The text to clean.
        
    Returns:
        Cleaned text with URLs, phone numbers, emails, and boilerplate removed.
    """
    if not isinstance(text, str):
        return text
    
    # Remove URLs (http://, https://, www.)
    url_pattern = r'(?:https?://|www\.)[^\s]+'
    cleaned = re.sub(url_pattern, '', text)
    
    # Remove social media domain patterns
    social_patterns = [
        r'facebook\.com/[^\s]+',
        r'instagram\.com/[^\s]+',
        r'youtube\.com/[^\s]+',
        r'linkedin\.com/[^\s]+',
        r'x\.com/[^\s]+',
        r'tiktok\.com/[^\s]+',
        r'press\.bmwgroup\.com[^\s]*',
        r'press\.bmw\.de[^\s]*',
    ]
    for pattern in social_patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    
    # Remove phone numbers (various formats)
    cleaned = re.sub(r'(?:Phone|Telephone|Tel\.?|Fax):\s*\+?[\d\s\-()]+', '', cleaned)
    
    # Remove email addresses
    cleaned = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '', cleaned)
    
    # Remove common boilerplate sections
    boilerplate_patterns = [
        r'If you have any questions, please contact:.*?(?=\n\n|\Z)',
        r'BMW Group Corporate Communications.*?(?=\n\n|\Z)',
        r'With its four brands BMW, MINI, Rolls-Royce and BMW Motorrad.*?(?=\n\n|\Z)',
    ]
    for pattern in boilerplate_patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.DOTALL | re.IGNORECASE)
    
    # Clean up multiple consecutive newlines and spaces
    cleaned = re.sub(r'\n\n\n+', '\n\n', cleaned)
    cleaned = re.sub(r' +', ' ', cleaned)
    cleaned = re.sub(r'\n +', '\n', cleaned)
    cleaned = cleaned.strip()
    
    return cleaned


def clean_article(article: dict) -> dict:
    """
    Clean a single article dictionary.
    
    Args:
        article: Dictionary containing article data with 'content' and 'summary' fields.
        
    Returns:
        Cleaned article dictionary.
    """
    cleaned = article.copy()
    
    # Clean content and summary fields
    if 'content' in cleaned:
        cleaned['content'] = clean_text(cleaned['content'])
    if 'summary' in cleaned:
        cleaned['summary'] = clean_text(cleaned['summary'])
    
    # Ensure tags is a list
    if 'tags' in cleaned and isinstance(cleaned['tags'], str):
        cleaned['tags'] = [t.strip() for t in cleaned['tags'].split(',')]
    
    return cleaned


def process_jsonl_file(input_path: Union[str, Path], output_path: Union[str, Path]) -> int:
    """
    Process a JSONL file and save cleaned version.
    
    Args:
        input_path: Path to input JSONL file.
        output_path: Path to output cleaned JSONL file.
        
    Returns:
        Number of articles processed.
    """
    articles = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                article = json.loads(line)
                cleaned = clean_article(article)
                articles.append(cleaned)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for article in articles:
            f.write(json.dumps(article, ensure_ascii=False) + '\n')
    
    return len(articles)


def process_json_file(input_path: Union[str, Path], output_path: Union[str, Path]) -> int:
    """
    Process a JSON file and save cleaned version.
    
    Args:
        input_path: Path to input JSON file.
        output_path: Path to output cleaned JSON file.
        
    Returns:
        Number of articles processed.
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle both formats: list of articles or dict with 'articles' key
    if isinstance(data, list):
        articles = data
        cleaned_articles = [clean_article(a) for a in articles]
        output_data = cleaned_articles
    else:
        articles = data.get('articles', [])
        cleaned_articles = [clean_article(a) for a in articles]
        output_data = {
            'count': len(cleaned_articles),
            'articles': cleaned_articles
        }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, ensure_ascii=False, fp=f, indent=2)
    
    return len(cleaned_articles)


def main():
    """CLI entry point for data cleaning."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Clean BMW articles dataset by removing URLs, contact info, and boilerplate."
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to input file (JSONL or JSON format)"
    )
    parser.add_argument(
        "-o", "--output",
        help="Path to output file (default: input_clean.jsonl in datasets/clean_data/)"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = Path("datasets/clean_data")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{input_path.stem}_clean.jsonl"
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Process based on file extension
    if input_path.suffix == ".jsonl":
        count = process_jsonl_file(input_path, output_path)
    elif input_path.suffix == ".json":
        count = process_json_file(input_path, output_path)
    else:
        print(f"Error: Unsupported file format '{input_path.suffix}'. Use .json or .jsonl")
        return
    
    print(f"Processed {count} articles")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()

