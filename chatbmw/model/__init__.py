"""
Model module for BMW article processing.

Includes:
- evaluator: Evaluation metrics for all tasks
- trainer: Model training utilities
"""
from chatbmw.model.evaluator import (
    # Task identification
    identify_task,
    # Text generation metrics
    compute_rouge_scores,
    compute_bert_score,
    compute_bleu_score,
    compute_chrf_score,
    evaluate_text_generation,
    # Perplexity metrics
    compute_perplexity,
    compute_perplexity_from_dataset,
    # Classification metrics
    compute_classification_metrics,
    # Multi-label metrics
    compute_tag_metrics,
    parse_tags,
    # Length metrics
    compute_length_metrics,
    # Main evaluation functions
    evaluate_by_task,
    load_and_evaluate,
    save_metrics,
    print_summary,
    assess_quality,
    QUALITY_THRESHOLDS,
)

__all__ = [
    # Task identification
    "identify_task",
    # Text generation metrics
    "compute_rouge_scores",
    "compute_bert_score",
    "compute_bleu_score",
    "compute_chrf_score",
    "evaluate_text_generation",
    # Perplexity metrics
    "compute_perplexity",
    "compute_perplexity_from_dataset",
    # Classification metrics
    "compute_classification_metrics",
    # Multi-label metrics
    "compute_tag_metrics",
    "parse_tags",
    # Length metrics
    "compute_length_metrics",
    # Main evaluation functions
    "evaluate_by_task",
    "load_and_evaluate",
    "save_metrics",
    "print_summary",
    "assess_quality",
    "QUALITY_THRESHOLDS",
]

