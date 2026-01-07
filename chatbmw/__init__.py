"""
ChatBMW - BMW Article Processing and Chat Template Generation

Modules:
- data: Data cleaning and chat template processing
- model: Evaluation metrics and model training
- scraper: Web scraping utilities
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
    # Evaluator functions
    "identify_task",
    "compute_rouge_scores",
    "compute_bert_score",
    "compute_bleu_score",
    "compute_chrf_score",
    "evaluate_text_generation",
    "compute_perplexity",
    "compute_perplexity_from_dataset",
    "compute_classification_metrics",
    "compute_tag_metrics",
    "parse_tags",
    "compute_length_metrics",
    "evaluate_by_task",
    "load_and_evaluate",
    "save_metrics",
    "print_summary",
    "assess_quality",
    "QUALITY_THRESHOLDS",
]

__version__ = "0.1.0"

# Data module exports
_DATA_EXPORTS = {
    "clean_text",
    "clean_article",
    "process_jsonl_file",
    "process_json_file",
    "create_chat_message",
    "create_chat_conversation",
    "article_to_chat_data",
    "process_jsonl_to_chat",
    "process_jsonl_to_chat_by_task",
    "convert_dataset_to_chat",
    "process_dataset_to_chat_splits",
    "SYSTEM_PROMPTS",
    "USER_INSTRUCTIONS",
}

# Model module exports
_MODEL_EXPORTS = {
    "identify_task",
    "compute_rouge_scores",
    "compute_bert_score",
    "compute_bleu_score",
    "compute_chrf_score",
    "evaluate_text_generation",
    "compute_perplexity",
    "compute_perplexity_from_dataset",
    "compute_classification_metrics",
    "compute_tag_metrics",
    "parse_tags",
    "compute_length_metrics",
    "evaluate_by_task",
    "load_and_evaluate",
    "save_metrics",
    "print_summary",
    "assess_quality",
    "QUALITY_THRESHOLDS",
}


def __getattr__(name):
    """Lazy import to avoid RuntimeWarning when running submodules as __main__."""
    if name in _DATA_EXPORTS:
        from chatbmw import data
        return getattr(data, name)
    
    if name in _MODEL_EXPORTS:
        from chatbmw import model
        return getattr(model, name)
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
