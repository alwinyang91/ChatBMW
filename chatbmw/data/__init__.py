"""
Data processing module for BMW articles dataset.

Includes:
- cleaner: Data cleaning utilities
- processor: Chat template processing utilities
"""

__all__ = [
    # Cleaner functions
    "clean_text",
    "clean_article",
    "process_jsonl_file",
    "process_json_file",
    # Processor functions
    "create_chat_message",
    "create_chat_conversation",
    "article_to_chat_data",
    "process_jsonl_to_chat",
    "process_jsonl_to_chat_by_task",
    "convert_dataset_to_chat",
    "process_dataset_to_chat_splits",
    "SYSTEM_PROMPTS",
    "USER_INSTRUCTIONS",
]


def __getattr__(name):
    """Lazy import to avoid RuntimeWarning when running modules as __main__."""
    if name in (
        "clean_text",
        "clean_article", 
        "process_jsonl_file",
        "process_json_file",
    ):
        from chatbmw.data import cleaner
        return getattr(cleaner, name)
    
    if name in (
        "create_chat_message",
        "create_chat_conversation",
        "article_to_chat_data",
        "process_jsonl_to_chat",
        "process_jsonl_to_chat_by_task",
        "convert_dataset_to_chat",
        "process_dataset_to_chat_splits",
        "SYSTEM_PROMPTS",
        "USER_INSTRUCTIONS",
    ):
        from chatbmw.data import processor
        return getattr(processor, name)
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

