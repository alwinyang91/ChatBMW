"""
Command-Line Interface for BMW PressClub Scraper

This module provides the CLI entry point for the scraper.
"""

import argparse
from datetime import datetime
from pathlib import Path

from .config import config
from .scraper import BMWArticleScraper


def generate_output_filename(base_path: str, article_count: int, fetch_details: bool = False, format: str = "json") -> str:
    """
    Generate output filename with date, article count, and details flag.
    
    Args:
        base_path: Base file path (can be directory or full path)
        article_count: Number of articles
        fetch_details: Whether full article content was fetched
        format: Output format extension (json, jsonl)
        
    Returns:
        str: Generated file path with date, count, and details flag
    """
    base_path = Path(base_path)
    today = datetime.now().strftime("%Y-%m-%d")
    details_prefix = "_details" if fetch_details else ""
    ext = f".{format}"
    
    # Check if it's an existing directory or a path without extension
    if base_path.exists() and base_path.is_dir():
        # It's a directory - create filename inside it
        output_path = base_path / f"bmw_articles{details_prefix}_{today}_{article_count}{ext}"
    elif not base_path.suffix:
        # No extension - treat as directory
        output_path = Path(base_path) / f"bmw_articles{details_prefix}_{today}_{article_count}{ext}"
    else:
        # It's a file path - insert date and count before extension
        stem = base_path.stem
        file_ext = base_path.suffix
        output_path = base_path.parent / f"{stem}{details_prefix}_{today}_{article_count}{file_ext}"
    
    return str(output_path)


def main():
    """
    Main entry point for the command-line tool.

    Parses arguments, runs the scraper, and exports results.
    """
    # Setup argument parser
    parser = argparse.ArgumentParser(
        description="BMW PressClub Article Scraper for LLM Fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m chatbmw.scraper                          # Basic scraping
  python -m chatbmw.scraper --detail                 # With full content
  python -m chatbmw.scraper --limit 50               # Limit to 50 articles
  python -m chatbmw.scraper -o bmw_training.jsonl --detail  # Output JSONL in Alpaca format

Configuration is loaded from config.yaml in project root.
        """,
    )
    
    parser.add_argument(
        "--detail", "-d",
        action="store_true",
        help=f"Fetch full article content (slower, default from config: {config.scraper.fetch_details})",
    )
    
    # Build default output path using config's output directory and subdirectory
    # Format is determined by config (can be multiple)
    output_base = Path(config.output.directory)
    if config.output.subdirectory:
        output_base = output_base / config.output.subdirectory
    default_output = str(output_base)
    
    parser.add_argument(
        "--output", "-o",
        default=default_output,
        help=f"Output base path or directory (default: {default_output}). "
             f"Exports to formats: {', '.join(config.output.format)}. "
             f"Filename will include date and article count automatically.",
    )
    
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help=f"Limit number of articles to scrape (default from config: {config.scraper.limit or 'no limit'})",
    )
    
    parser.add_argument(
        "--exact",
        action="store_true",
        help="Use exact output path without adding date/count suffix (useful for CI/CD pipelines)",
    )
    
    # Note: pages parameter removed - pagination is handled automatically via limit
    
    args = parser.parse_args()
    
    # Print header
    print("=" * 60)
    print("🚗 BMW PressClub Article Scraper")
    print("   For LLM Fine-tuning Dataset Generation")
    print("=" * 60)
    print(f"   Config: delay={config.scraper.delay}s, timeout={config.scraper.timeout}s")
    print(f"           fetch_details={config.scraper.fetch_details}, limit={config.scraper.limit or 'no limit'}")
    
    # Create and run scraper
    scraper = BMWArticleScraper()
    
    try:
        # Determine fetch_details: use flag value if provided, otherwise None (so scraper uses config)
        fetch_details_arg = args.detail if args.detail else None
        
        # Scrape articles
        articles = scraper.scrape(
            fetch_details=fetch_details_arg,
            limit=args.limit,
        )
        
        if articles:
            # Show preview
            print("\n" + "-" * 60)
            print("📋 Preview of scraped articles:")
            print("-" * 60)
            
            for i, article in enumerate(articles[:5], 1):
                print(f"\n{i}. {article.title}")
                if article.date:
                    print(f"   📅 {article.date}")
                if article.tags:
                    print(f"   🏷️  {', '.join(article.tags[:3])}")
                if article.content:
                    print(f"   📄 Content: {len(article.content)} chars")
            
            if len(articles) > 5:
                print(f"\n... and {len(articles) - 5} more articles")
            
            # Determine if details were fetched (same logic as scraper uses)
            fetch_details = args.detail if args.detail else config.scraper.fetch_details
            
            # Export to all configured formats
            print("\n" + "-" * 60)
            formats = config.output.format
            
            for fmt in formats:
                # Generate output filename with date, count, and details flag
                if args.exact:
                    output_path = args.output
                else:
                    output_path = generate_output_filename(args.output, len(articles), fetch_details, fmt)
                
                scraper.export(articles, output_path, format=fmt)
        else:
            print("\n❌ No articles found.")
    
    finally:
        scraper.close()
    
    # Print footer
    print("\n" + "=" * 60)
    print("✨ Scraping complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
