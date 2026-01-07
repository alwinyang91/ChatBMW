"""
Evaluation metrics for BMW article processing tasks.

This module provides comprehensive evaluation metrics for:
- Text generation tasks (summarization, title generation)
- Multi-label classification (tag extraction)
- Single-label classification (type classification)
- Model-level evaluation (perplexity)
"""
import os
import json
import numpy as np

# Enable logits return for Unsloth models (required for perplexity computation)
os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

from chatbmw.data.procesor import USER_INSTRUCTIONS


# ============================================================================
# Task Identification
# ============================================================================
def identify_task(user_message: str) -> str:
    """
    Identify task type from user message.
    
    Args:
        user_message: The user input message from the conversation.
        
    Returns:
        Task type string: 'summarization', 'title_generation', 
        'tag_extraction', 'type_classification', or 'unknown'.
    """
    user_lower = user_message.lower()
    if "summarize" in user_lower:
        return "summarization"
    elif "title" in user_lower:
        return "title_generation"
    elif "topics" in user_lower or "categories" in user_lower or "extract" in user_lower:
        return "tag_extraction"
    elif "type" in user_lower or "identify the type" in user_lower:
        return "type_classification"
    return "unknown"


# ============================================================================
# Text Generation Metrics (Summarization & Title Generation)
# ============================================================================
def compute_rouge_scores(
    predictions: List[str], 
    references: List[str],
    use_stemmer: bool = True,
) -> Dict[str, float]:
    """
    Compute ROUGE-1, ROUGE-2, and ROUGE-L scores.
    Recall-Oriented Understudy for Gisting Evaluation
    ROUGE-1: Unigram precision and recall
    ROUGE-2: Bigram precision and recall
    ROUGE-L: Longest common subsequence precision and recall
    Args:
        predictions: List of generated texts.
        references: List of reference texts.
        use_stemmer: Whether to use stemming for token matching.
        
    Returns:
        Dictionary with ROUGE precision, recall, and F1 scores.
    """
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        raise ImportError(
            "rouge-score is required for ROUGE metrics. "
            "Install it with: pip install rouge-score"
        )
    
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'], 
        use_stemmer=use_stemmer
    )
    
    scores = defaultdict(list)
    for pred, ref in zip(predictions, references):
        result = scorer.score(ref, pred)
        for key in result:
            scores[f"{key}_precision"].append(result[key].precision)
            scores[f"{key}_recall"].append(result[key].recall)
            scores[f"{key}_fmeasure"].append(result[key].fmeasure)
    
    return {k: float(np.mean(v)) for k, v in scores.items()}


def compute_bert_score(
    predictions: List[str], 
    references: List[str],
    lang: str = "en",
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Compute BERTScore for semantic similarity.
    
    Args:
        predictions: List of generated texts.
        references: List of reference texts.
        lang: Language code for BERTScore.
        verbose: Whether to show progress.
        
    Returns:
        Dictionary with BERTScore precision, recall, and F1.
    """
    try:
        from bert_score import score as bert_score_fn
    except ImportError:
        raise ImportError(
            "bert-score is required for BERTScore metrics. "
            "Install it with: pip install bert-score"
        )
    
    P, R, F1 = bert_score_fn(predictions, references, lang=lang, verbose=verbose)
    return {
        "bertscore_precision": float(P.mean().item()),
        "bertscore_recall": float(R.mean().item()),
        "bertscore_f1": float(F1.mean().item())
    }


def compute_bleu_score(
    predictions: List[str], 
    references: List[str],
) -> Dict[str, float]:
    """
    BLEU: Bilingual Evaluation Understudy
    Compute BLEU score for text generation quality.
    
    Measures n-gram precision with brevity penalty.
    Originally designed for machine translation but widely used for generation tasks.
    
    Strengths:
    - Fast computation
    - Language-independent
    - Well-established benchmark
    
    Weaknesses:
    - Doesn't consider synonyms or paraphrases
    - Precision-focused (may penalize longer outputs)
    - No semantic understanding
    
    Args:
        predictions: List of generated texts.
        references: List of reference texts.
        
    Returns:
        Dictionary with BLEU score.
    """
    try:
        from sacrebleu.metrics import BLEU
    except ImportError:
        raise ImportError(
            "sacrebleu is required for BLEU metrics. "
            "Install it with: pip install sacrebleu"
        )
    
    bleu = BLEU()
    # sacrebleu expects references as list of lists
    refs_formatted = [[ref] for ref in references]
    result = bleu.corpus_score(predictions, list(zip(*refs_formatted)))
    
    return {
        "bleu": float(result.score),
        "bleu_bp": float(result.bp),  # brevity penalty
    }


def compute_chrf_score(
    predictions: List[str], 
    references: List[str],
) -> Dict[str, float]:
    """
    chrF: Character n-gram F-score
    
    Measures character-level similarity rather than word-level.
    
    Strengths:
    - Language-independent (no tokenization needed)
    - Robust to morphological variations
    - Good for agglutinative languages
    - Captures sub-word similarities
    
    Weaknesses:
    - May reward superficial character overlap
    - Less interpretable than word-level metrics
    
    Args:
        predictions: List of generated texts.
        references: List of reference texts.
        
    Returns:
        Dictionary with chrF score.
    """
    try:
        from sacrebleu.metrics import CHRF
    except ImportError:
        raise ImportError(
            "sacrebleu is required for chrF metrics. "
            "Install it with: pip install sacrebleu"
        )
    
    chrf = CHRF()
    refs_formatted = [[ref] for ref in references]
    result = chrf.corpus_score(predictions, list(zip(*refs_formatted)))
    
    return {
        "chrf": float(result.score),
    }


# ============================================================================
# Perplexity Metrics (Model-Level Evaluation)
# ============================================================================
def compute_perplexity(
    model,
    tokenizer,
    texts: List[str],
    batch_size: int = 4,
    max_length: Optional[int] = None,
    device: Optional[str] = None,
) -> Dict[str, Union[float, List[float]]]:
    """
    Compute perplexity of a language model on given texts.
    
    Perplexity measures how "surprised" the model is by the text.
    Lower perplexity = better language modeling ability.
    
    PPL = exp(average negative log-likelihood)
    
    Strengths:
    - Directly measures language modeling quality
    - Model-intrinsic metric (no reference needed)
    - Good for comparing models on same data
    
    Weaknesses:
    - Requires model access (not just text outputs)
    - Computationally expensive
    - Not directly comparable across different tokenizers
    
    Args:
        model: HuggingFace model (AutoModelForCausalLM).
        tokenizer: HuggingFace tokenizer.
        texts: List of texts to evaluate.
        batch_size: Batch size for processing.
        max_length: Maximum sequence length (defaults to model's max).
        device: Device to use ('cuda', 'cpu', or None for auto-detect).
        
    Returns:
        Dictionary with:
        - perplexity: Average perplexity across all texts
        - perplexities: List of per-sample perplexity values
        - avg_loss: Average cross-entropy loss
    """
    try:
        import torch
    except ImportError:
        raise ImportError(
            "PyTorch is required for perplexity computation. "
            "Install it with: pip install torch"
        )
    
    # Auto-detect device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = model.to(device)
    model.eval()
    
    # Set max_length
    if max_length is None:
        max_length = getattr(tokenizer, 'model_max_length', 2048)
        # Clamp to reasonable value
        max_length = min(max_length, 4096)
    
    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    all_losses = []
    all_perplexities = []
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            encodings = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)
            
            input_ids = encodings.input_ids
            attention_mask = encodings.attention_mask
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )
            
            # Get per-sample loss
            # outputs.loss is the mean loss, we need per-sample
            logits = outputs.logits
            
            # Shift for causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            shift_mask = attention_mask[..., 1:].contiguous()
            
            # Compute per-token loss
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            loss = loss.view(shift_labels.size())
            
            # Mask padding and compute per-sample loss
            loss = loss * shift_mask
            sample_losses = loss.sum(dim=1) / shift_mask.sum(dim=1)
            
            # Convert to perplexity
            sample_perplexities = torch.exp(sample_losses)
            
            all_losses.extend(sample_losses.cpu().tolist())
            all_perplexities.extend(sample_perplexities.cpu().tolist())
    
    avg_loss = float(np.mean(all_losses))
    avg_perplexity = float(np.exp(avg_loss))
    
    return {
        "perplexity": avg_perplexity,
        "perplexities": all_perplexities,
        "avg_loss": avg_loss,
        "num_samples": len(texts),
    }


def compute_perplexity_from_dataset(
    model,
    tokenizer,
    dataset_path: Union[str, Path],
    text_field: str = "text",
    batch_size: int = 4,
    max_samples: Optional[int] = None,
    device: Optional[str] = None,
) -> Dict[str, Union[float, List[float]]]:
    """
    Compute perplexity on a dataset file (JSON or JSONL).
    
    Args:
        model: HuggingFace model.
        tokenizer: HuggingFace tokenizer.
        dataset_path: Path to JSON/JSONL file.
        text_field: Field name containing text to evaluate.
        batch_size: Batch size for processing.
        max_samples: Maximum number of samples to evaluate.
        device: Device to use.
        
    Returns:
        Dictionary with perplexity metrics.
    """
    dataset_path = Path(dataset_path)
    texts = []
    
    # Load data
    if dataset_path.suffix == '.jsonl':
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    if text_field in item:
                        texts.append(item[text_field])
    else:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                texts = [item.get(text_field, "") for item in data if text_field in item]
            elif text_field in data:
                texts = data[text_field] if isinstance(data[text_field], list) else [data[text_field]]
    
    if max_samples:
        texts = texts[:max_samples]
    
    if not texts:
        raise ValueError(f"No texts found in {dataset_path} with field '{text_field}'")
    
    print(f"📊 Computing perplexity on {len(texts)} samples...")
    
    return compute_perplexity(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        batch_size=batch_size,
        device=device,
    )


# ============================================================================
# Metrics Comparison Guide
# ============================================================================
"""
METRICS COMPARISON FOR TEXT GENERATION TASKS:

| Metric     | Speed  | Synonyms | Semantics | Needs Model | Best For                    |
|------------|--------|----------|-----------|-------------|------------------------------|
| ROUGE      | Fast   | No       | No        | No          | Summarization, recall-focus  |
| BLEU       | Fast   | No       | No        | No          | Short texts, precision-focus |
| chrF       | Fast   | No       | No        | No          | Morphologically rich text    |
| BERTScore  | Slow   | Yes      | Yes       | No          | Semantic similarity          |
| Perplexity | Medium | N/A      | N/A       | Yes         | Model quality, fluency       |

RECOMMENDATIONS BY TASK:
- Summarization: ROUGE-L, BERTScore
- Title Generation: ROUGE-1, BLEU, BERTScore  
- Tag Extraction: Jaccard, F1 (multi-label)
- Classification: Accuracy, F1 (per-class)
- Model Comparison: Perplexity (lower is better)

PERPLEXITY NOTES:
- Measures model's language modeling ability (lower = better)
- Requires model + tokenizer access, not just text outputs
- Not comparable across models with different tokenizers
- Typical ranges: Good (<20), Acceptable (20-50), Needs work (>50)

For comprehensive evaluation, use multiple metrics as they capture different aspects.
"""


# ============================================================================
# Classification Metrics (Type Classification)
# ============================================================================
def compute_classification_metrics(
    predictions: List[str], 
    references: List[str],
) -> Dict[str, Union[float, int, Dict]]:
    """
    Compute classification metrics for single-label tasks.
    
    Args:
        predictions: List of predicted labels.
        references: List of reference labels.
        
    Returns:
        Dictionary with accuracy and class distribution.
    """
    from collections import Counter
    
    # Normalize strings for comparison
    preds_normalized = [p.strip().lower() for p in predictions]
    refs_normalized = [r.strip().lower() for r in references]
    
    # Exact match accuracy
    correct = sum(p == r for p, r in zip(preds_normalized, refs_normalized))
    accuracy = correct / len(predictions) if predictions else 0.0
    
    # Per-class metrics
    class_counts = Counter(refs_normalized)
    class_correct = defaultdict(int)
    class_predicted = defaultdict(int)
    
    for pred, ref in zip(preds_normalized, refs_normalized):
        class_predicted[pred] += 1
        if pred == ref:
            class_correct[ref] += 1
    
    # Per-class accuracy
    class_accuracy = {}
    for cls, count in class_counts.items():
        class_accuracy[cls] = class_correct[cls] / count if count > 0 else 0.0
    
    return {
        "accuracy": float(accuracy),
        "correct": int(correct),
        "total": len(predictions),
        "class_distribution": dict(class_counts),
        "class_accuracy": class_accuracy,
    }


# ============================================================================
# Multi-label Metrics (Topic/Tag Extraction)
# ============================================================================
def parse_tags(text: str) -> set:
    """
    Parse comma-separated tags into a set.
    
    Args:
        text: Comma-separated string of tags.
        
    Returns:
        Set of normalized tag strings.
    """
    return set(tag.strip().lower() for tag in text.split(',') if tag.strip())


def compute_tag_metrics(
    predictions: List[str], 
    references: List[str],
) -> Dict[str, float]:
    """
    Compute metrics for multi-label tag extraction.
    
    Args:
        predictions: List of predicted tag strings (comma-separated).
        references: List of reference tag strings (comma-separated).
        
    Returns:
        Dictionary with exact match, Jaccard, precision, recall, and F1.
    """
    exact_matches = 0
    jaccard_scores = []
    all_precisions = []
    all_recalls = []
    all_f1s = []
    
    for pred, ref in zip(predictions, references):
        pred_tags = parse_tags(pred)
        ref_tags = parse_tags(ref)
        
        # Exact match
        if pred_tags == ref_tags:
            exact_matches += 1
        
        # Jaccard similarity
        if pred_tags or ref_tags:
            intersection = len(pred_tags & ref_tags)
            union = len(pred_tags | ref_tags)
            jaccard = intersection / union if union > 0 else 0
            jaccard_scores.append(jaccard)
            
            # Precision, Recall, F1
            precision = intersection / len(pred_tags) if pred_tags else 0
            recall = intersection / len(ref_tags) if ref_tags else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            all_precisions.append(precision)
            all_recalls.append(recall)
            all_f1s.append(f1)
    
    n = len(predictions)
    return {
        "exact_match_accuracy": float(exact_matches / n) if n > 0 else 0.0,
        "jaccard_similarity": float(np.mean(jaccard_scores)) if jaccard_scores else 0.0,
        "precision": float(np.mean(all_precisions)) if all_precisions else 0.0,
        "recall": float(np.mean(all_recalls)) if all_recalls else 0.0,
        "f1": float(np.mean(all_f1s)) if all_f1s else 0.0,
    }


# ============================================================================
# Length and Coverage Metrics
# ============================================================================
def compute_length_metrics(
    predictions: List[str], 
    references: List[str],
) -> Dict[str, float]:
    """
    Compute length-based metrics for generated text.
    
    Args:
        predictions: List of generated texts.
        references: List of reference texts.
        
    Returns:
        Dictionary with length statistics.
    """
    pred_lengths = [len(p.split()) for p in predictions]
    ref_lengths = [len(r.split()) for r in references]
    length_ratios = [
        p / r if r > 0 else 0 
        for p, r in zip(pred_lengths, ref_lengths)
    ]
    
    return {
        "avg_pred_length": float(np.mean(pred_lengths)),
        "avg_ref_length": float(np.mean(ref_lengths)),
        "avg_length_ratio": float(np.mean(length_ratios)),
        "std_length_ratio": float(np.std(length_ratios)),
    }


# ============================================================================
# Main Evaluation Functions
# ============================================================================
def evaluate_text_generation(
    predictions: List[str],
    references: List[str],
    compute_bertscore: bool = True,
    compute_bleu: bool = False,
    compute_chrf: bool = False,
) -> Dict[str, float]:
    """
    Evaluate text generation quality with multiple metrics.
    
    Args:
        predictions: List of generated texts.
        references: List of reference texts.
        compute_bertscore: Whether to compute BERTScore (slower but semantic).
        compute_bleu: Whether to compute BLEU score (precision-focused).
        compute_chrf: Whether to compute chrF (character-level).
        
    Returns:
        Dictionary with all computed metrics.
        
    Metrics Guide:
        - ROUGE: Always computed. Good baseline for overlap.
        - BERTScore: Recommended for semantic similarity. Slower.
        - BLEU: Optional. Better for shorter, precise outputs.
        - chrF: Optional. Good for morphologically rich text.
    """
    metrics = {}
    
    # ROUGE scores (always computed - fast and reliable baseline)
    rouge_metrics = compute_rouge_scores(predictions, references)
    metrics.update(rouge_metrics)
    
    # Length metrics (always computed - useful for analysis)
    length_metrics = compute_length_metrics(predictions, references)
    metrics.update(length_metrics)
    
    # BERTScore (optional, slower but more semantic)
    if compute_bertscore:
        try:
            bert_metrics = compute_bert_score(predictions, references)
            metrics.update(bert_metrics)
        except ImportError as e:
            print(f"⚠️ BERTScore skipped: {e}")
    
    # BLEU score (optional, precision-focused)
    if compute_bleu:
        try:
            bleu_metrics = compute_bleu_score(predictions, references)
            metrics.update(bleu_metrics)
        except ImportError as e:
            print(f"⚠️ BLEU skipped: {e}")
    
    # chrF score (optional, character-level)
    if compute_chrf:
        try:
            chrf_metrics = compute_chrf_score(predictions, references)
            metrics.update(chrf_metrics)
        except ImportError as e:
            print(f"⚠️ chrF skipped: {e}")
    
    return metrics


def evaluate_by_task(
    results: List[Dict],
    compute_bertscore: bool = True,
    verbose: bool = True,
) -> Dict[str, Dict]:
    """
    Evaluate results grouped by task type.
    
    Args:
        results: List of result dictionaries with 'user_message', 
                 'expected', and 'generated' keys.
        compute_bertscore: Whether to compute BERTScore for text generation tasks.
        verbose: Whether to print results.
        
    Returns:
        Dictionary with metrics for each task type.
    """
    # Group results by task
    task_groups = defaultdict(lambda: {"predictions": [], "references": []})
    
    for r in results:
        task = identify_task(r["user_message"])
        task_groups[task]["predictions"].append(r["generated"])
        task_groups[task]["references"].append(r["expected"])
    
    all_metrics = {}
    
    for task, data in task_groups.items():
        preds = data["predictions"]
        refs = data["references"]
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Task: {task.upper()} ({len(preds)} samples)")
            print(f"{'='*60}")
        
        if task in ["summarization", "title_generation"]:
            # Text generation metrics
            metrics = evaluate_text_generation(
                preds, refs, 
                compute_bertscore=compute_bertscore,
            )
            
            if verbose:
                print("\n📊 ROUGE Scores:")
                print(f"  ROUGE-1 F1: {metrics['rouge1_fmeasure']:.4f}")
                print(f"  ROUGE-2 F1: {metrics['rouge2_fmeasure']:.4f}")
                print(f"  ROUGE-L F1: {metrics['rougeL_fmeasure']:.4f}")
                
                if compute_bertscore and 'bertscore_f1' in metrics:
                    print("\n📊 BERTScore (Semantic Similarity):")
                    print(f"  Precision: {metrics['bertscore_precision']:.4f}")
                    print(f"  Recall: {metrics['bertscore_recall']:.4f}")
                    print(f"  F1: {metrics['bertscore_f1']:.4f}")
                
                print("\n📊 Length Metrics:")
                print(f"  Avg Prediction Length: {metrics['avg_pred_length']:.1f} words")
                print(f"  Avg Reference Length: {metrics['avg_ref_length']:.1f} words")
                print(f"  Length Ratio: {metrics['avg_length_ratio']:.2f}")
            
            all_metrics[task] = metrics
            
        elif task == "tag_extraction":
            metrics = compute_tag_metrics(preds, refs)
            
            if verbose:
                print("\n📊 Tag Extraction Metrics:")
                print(f"  Exact Match: {metrics['exact_match_accuracy']:.4f}")
                print(f"  Jaccard Similarity: {metrics['jaccard_similarity']:.4f}")
                print(f"  Precision: {metrics['precision']:.4f}")
                print(f"  Recall: {metrics['recall']:.4f}")
                print(f"  F1: {metrics['f1']:.4f}")
            
            all_metrics[task] = metrics
            
        elif task == "type_classification":
            metrics = compute_classification_metrics(preds, refs)
            
            if verbose:
                print("\n📊 Classification Metrics:")
                print(f"  Accuracy: {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total']})")
                print(f"\n  Class Distribution: {metrics['class_distribution']}")
                print(f"  Per-class Accuracy: {metrics['class_accuracy']}")
            
            all_metrics[task] = metrics
        
        else:
            if verbose:
                print(f"  ⚠️ Unknown task type, skipping evaluation")
    
    return all_metrics


def load_and_evaluate(
    results_path: Union[str, Path],
    compute_bertscore: bool = True,
    verbose: bool = True,
) -> Dict[str, Dict]:
    """
    Load results from JSON file and evaluate.
    
    Args:
        results_path: Path to the JSON file containing test results.
        compute_bertscore: Whether to compute BERTScore.
        verbose: Whether to print results.
        
    Returns:
        Dictionary with metrics for each task type.
    """
    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    if verbose:
        print(f"📂 Loaded {len(results)} results from {results_path}")
    
    return evaluate_by_task(results, compute_bertscore, verbose)


def save_metrics(
    metrics: Dict,
    output_path: Union[str, Path],
) -> None:
    """
    Save evaluation metrics to a JSON file.
    
    Args:
        metrics: Dictionary of metrics to save.
        output_path: Path to save the metrics JSON file.
    """
    # Ensure all values are JSON serializable
    def convert_to_serializable(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    serializable_metrics = convert_to_serializable(metrics)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_metrics, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Metrics saved to: {output_path}")


def print_summary(metrics: Dict[str, Dict]) -> None:
    """
    Print a summary table of all metrics.
    
    Args:
        metrics: Dictionary of metrics from evaluate_by_task.
    """
    print("\n" + "="*70)
    print("📊 EVALUATION SUMMARY")
    print("="*70)
    
    for task, task_metrics in metrics.items():
        print(f"\n{task.upper()}:")
        
        if task in ["summarization", "title_generation"]:
            print(f"  ROUGE-L F1: {task_metrics.get('rougeL_fmeasure', 0):.4f}")
            if 'bertscore_f1' in task_metrics:
                print(f"  BERTScore F1: {task_metrics.get('bertscore_f1', 0):.4f}")
        elif task == "tag_extraction":
            print(f"  F1: {task_metrics.get('f1', 0):.4f}")
            print(f"  Jaccard: {task_metrics.get('jaccard_similarity', 0):.4f}")
        elif task == "type_classification":
            print(f"  Accuracy: {task_metrics.get('accuracy', 0):.4f}")
        elif task == "perplexity":
            print(f"  Perplexity: {task_metrics.get('perplexity', 0):.4f}")
            print(f"  Avg Loss: {task_metrics.get('avg_loss', 0):.4f}")
    
    print("\n" + "="*70)


# ============================================================================
# Quality Thresholds Reference
# ============================================================================
QUALITY_THRESHOLDS = {
    "summarization": {
        "good": {"rougeL_fmeasure": 0.30},
        "acceptable": {"rougeL_fmeasure": 0.20},
    },
    "title_generation": {
        # Title generation typically has lower overlap due to creativity
        "good": {"rouge1_fmeasure": 0.25},
        "acceptable": {"rouge1_fmeasure": 0.15},
    },
    "tag_extraction": {
        "good": {"f1": 0.60, "jaccard_similarity": 0.50},
        "acceptable": {"f1": 0.40, "jaccard_similarity": 0.35},
    },
    "type_classification": {
        "good": {"accuracy": 0.90},
        "acceptable": {"accuracy": 0.80},
    },
}


def assess_quality(metrics: Dict[str, Dict]) -> Dict[str, str]:
    """
    Assess the quality of results based on thresholds.
    
    Args:
        metrics: Dictionary of metrics from evaluate_by_task.
        
    Returns:
        Dictionary mapping task names to quality ratings.
    """
    ratings = {}
    
    for task, task_metrics in metrics.items():
        if task not in QUALITY_THRESHOLDS:
            ratings[task] = "unknown"
            continue
        
        thresholds = QUALITY_THRESHOLDS[task]
        
        # Check good thresholds
        is_good = all(
            task_metrics.get(metric, 0) >= threshold
            for metric, threshold in thresholds["good"].items()
        )
        
        if is_good:
            ratings[task] = "good ✅"
            continue
        
        # Check acceptable thresholds
        is_acceptable = all(
            task_metrics.get(metric, 0) >= threshold
            for metric, threshold in thresholds["acceptable"].items()
        )
        
        if is_acceptable:
            ratings[task] = "acceptable 🟡"
        else:
            ratings[task] = "needs improvement 🔴"
    
    return ratings


# ============================================================================
# CLI Functions
# ============================================================================
def generate_response_from_messages(
    model,
    tokenizer,
    messages: List[Dict],
    max_new_tokens: int = 512,
) -> str:
    """
    Generate response for a given list of messages (chat format).
    
    Args:
        model: The language model.
        tokenizer: The tokenizer.
        messages: List of message dictionaries with 'role' and 'content' keys.
        max_new_tokens: Maximum number of tokens to generate.
        
    Returns:
        Generated response string.
    """
    import torch
    
    # Use only system and user messages for input (exclude assistant response)
    input_messages = [msg for msg in messages if msg["role"] != "assistant"]
    
    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        input_messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Decode only the generated part (remove the prompt)
    generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return generated_text.strip()


def run_inference(
    model,
    tokenizer,
    test_dataset,
    max_new_tokens: int = 512,
    verbose: bool = True,
) -> List[Dict]:
    """
    Run inference on test dataset and collect results.
    
    Args:
        model: The language model.
        tokenizer: The tokenizer.
        test_dataset: HuggingFace Dataset with 'messages' field.
        max_new_tokens: Maximum number of tokens to generate.
        verbose: Whether to show progress.
        
    Returns:
        List of result dictionaries.
    """
    from tqdm import tqdm
    import time
    
    results = []
    start_time = time.time()
    
    iterator = tqdm(range(len(test_dataset)), desc="Generating responses") if verbose else range(len(test_dataset))
    
    for idx in iterator:
        example = test_dataset[idx]
        messages = example["messages"]
        
        # Extract user and expected assistant messages
        user_content = next((msg["content"] for msg in messages if msg["role"] == "user"), "")
        expected_output = next((msg["content"] for msg in messages if msg["role"] == "assistant"), "")
        
        try:
            generated_output = generate_response_from_messages(
                model, tokenizer, messages, max_new_tokens
            )
            
            results.append({
                "index": idx,
                "user_message": user_content,
                "expected": expected_output,
                "generated": generated_output,
                "success": True
            })
        except Exception as e:
            results.append({
                "index": idx,
                "user_message": user_content,
                "expected": expected_output,
                "generated": f"Error: {str(e)}",
                "success": False
            })
    
    elapsed_time = time.time() - start_time
    if verbose:
        print(f"\n✓ Inference completed in {elapsed_time:.2f} seconds")
        print(f"✓ Processed {len(results)} examples")
        print(f"✓ Success rate: {sum(1 for r in results if r['success'])}/{len(results)}")
    
    return results


def main():
    """CLI entry point for model evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned BMW chat model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate with a results file (already generated)
  python -m chatbmw.model.evaluator --results-file checkpoints/test_results.json

  # Run full evaluation (inference + metrics)
  python -m chatbmw.model.evaluator \\
      --model-path checkpoints/unsloth/Llama-3.2-1B-Instruct/merged_model \\
      --test-data datasets/chat_data_2000/test_chat.jsonl \\
      --output-dir checkpoints/evaluation

  # Skip BERTScore for faster evaluation
  python -m chatbmw.model.evaluator --results-file results.json --no-bertscore
        """
    )
    
    # Input options (either results file OR model + test data)
    input_group = parser.add_argument_group("Input Options")
    input_group.add_argument(
        "--results-file",
        type=str,
        help="Path to existing results JSON file (skip inference)"
    )
    input_group.add_argument(
        "--model-path",
        type=str,
        help="Path to fine-tuned model (for running inference)"
    )
    input_group.add_argument(
        "--test-data",
        type=str,
        help="Path to test dataset JSONL file (for running inference)"
    )
    
    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory to save output files (default: current directory)"
    )
    output_group.add_argument(
        "--save-results",
        action="store_true",
        help="Save inference results to JSON file"
    )
    output_group.add_argument(
        "--save-metrics",
        action="store_true",
        default=True,
        help="Save evaluation metrics to JSON file (default: True)"
    )
    
    # Evaluation options
    eval_group = parser.add_argument_group("Evaluation Options")
    eval_group.add_argument(
        "--no-bertscore",
        action="store_true",
        help="Skip BERTScore computation (faster)"
    )
    eval_group.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate (default: 512)"
    )
    eval_group.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.results_file is None and (args.model_path is None or args.test_data is None):
        parser.error("Either --results-file OR both --model-path and --test-data are required")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    verbose = not args.quiet
    compute_bertscore = not args.no_bertscore
    
    # Get results (either load from file or run inference)
    perplexity_metrics = None  # Only computed when model is loaded
    
    if args.results_file:
        # Load existing results
        results_path = Path(args.results_file)
        if verbose:
            print(f"📂 Loading results from {results_path}...")
        
        with open(results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        if verbose:
            print(f"✓ Loaded {len(results)} results")
    else:
        # Run inference
        if verbose:
            print("🚀 Loading model and running inference...")
            print(f"   Model: {args.model_path}")
            print(f"   Test data: {args.test_data}")
        
        # Import unsloth for model loading
        try:
            from unsloth import FastLanguageModel
        except ImportError:
            raise ImportError(
                "unsloth is required for model loading. "
                "Install it with: pip install unsloth"
            )
        
        from datasets import load_dataset
        
        # Load model
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model_path,
            max_seq_length=10240,
            dtype=None,
            load_in_4bit=False,
        )
        FastLanguageModel.for_inference(model)
        
        if verbose:
            print("✓ Model loaded successfully")
        
        # Load test dataset
        test_dataset = load_dataset("json", data_files=args.test_data, split="train")
        
        if verbose:
            print(f"✓ Loaded {len(test_dataset)} test examples")
        
        # Run inference
        results = run_inference(
            model, tokenizer, test_dataset,
            max_new_tokens=args.max_new_tokens,
            verbose=verbose
        )
        
        # Save results if requested
        if args.save_results:
            results_output_path = output_dir / "test_results.json"
            results_output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(results_output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            if verbose:
                print(f"✓ Results saved to {results_output_path}")
        
        # Compute perplexity while model is loaded
        if verbose:
            print("\n📊 Computing perplexity...")
        
        # Get expected texts for perplexity computation
        expected_texts = [r["expected"] for r in results if r["success"]]
        perplexity_results = compute_perplexity(
            model=model,
            tokenizer=tokenizer,
            texts=expected_texts,
            batch_size=4,
        )
        # Store for later (exclude per-sample perplexities from saved metrics)
        perplexity_metrics = {
            "perplexity": perplexity_results["perplexity"],
            "avg_loss": perplexity_results["avg_loss"],
            "num_samples": perplexity_results["num_samples"],
        }
        
        if verbose:
            print(f"✓ Perplexity: {perplexity_metrics['perplexity']:.4f}")
            print(f"✓ Avg Loss: {perplexity_metrics['avg_loss']:.4f}")
    
    # Run evaluation
    if verbose:
        print("\n🚀 Running evaluation...")
    
    metrics = evaluate_by_task(results, compute_bertscore=compute_bertscore, verbose=verbose)
    
    # Add perplexity metrics if computed
    if perplexity_metrics is not None:
        metrics["perplexity"] = perplexity_metrics
    
    # Print summary
    print_summary(metrics)
    
    # Assess quality
    quality_ratings = assess_quality(metrics)
    print("\n📋 Quality Assessment:")
    for task, rating in quality_ratings.items():
        print(f"  {task}: {rating}")
    
    # Save metrics
    if args.save_metrics:
        metrics_output_path = output_dir / "evaluation_metrics.json"
        metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
        save_metrics(metrics, metrics_output_path)
    
    return metrics


if __name__ == "__main__":
    main()

