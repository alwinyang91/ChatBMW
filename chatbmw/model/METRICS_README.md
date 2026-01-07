# Evaluation Metrics Guide

This document provides a comprehensive explanation of all evaluation metrics implemented in `evaluator.py` for BMW article processing tasks.

## Table of Contents

- [Overview](#overview)
- [Text Generation Metrics](#text-generation-metrics)
  - [ROUGE Scores](#rouge-scores)
  - [BERTScore](#bertscore)
- [Classification Metrics](#classification-metrics)
- [Multi-label Tag Metrics](#multi-label-tag-metrics)
- [Length Metrics](#length-metrics)
- [Metrics Comparison](#metrics-comparison)
- [Quality Thresholds](#quality-thresholds)
- [Task-Specific Recommendations](#task-specific-recommendations)

---

## Overview

The evaluation module supports four main task types:

| Task | Type | Primary Metrics |
|------|------|-----------------|
| Summarization | Text Generation | ROUGE-L, BERTScore |
| Title Generation | Text Generation | ROUGE-1, BERTScore |
| Tag Extraction | Multi-label Classification | Jaccard, F1 |
| Type Classification | Single-label Classification | Accuracy |

---

## Text Generation Metrics

### ROUGE Scores

**ROUGE** (Recall-Oriented Understudy for Gisting Evaluation) measures the overlap between generated text and reference text using n-gram matching.

#### Variants

| Metric | Description | Best For |
|--------|-------------|----------|
| **ROUGE-1** | Unigram (single word) overlap | General content coverage |
| **ROUGE-2** | Bigram (two consecutive words) overlap | Phrase-level matching |
| **ROUGE-L** | Longest Common Subsequence (LCS) | Sentence-level structure |

#### Components

Each ROUGE variant provides three scores:

- **Precision**: What fraction of the generated n-grams appear in the reference?
  ```
  Precision = |Generated ∩ Reference| / |Generated|
  ```

- **Recall**: What fraction of the reference n-grams appear in the generated text?
  ```
  Recall = |Generated ∩ Reference| / |Reference|
  ```

- **F-measure (F1)**: Harmonic mean of precision and recall
  ```
  F1 = 2 × (Precision × Recall) / (Precision + Recall)
  ```

#### Interpretation

| Score Range | Quality |
|-------------|---------|
| 0.4+ | Excellent overlap |
| 0.25-0.4 | Good overlap |
| 0.15-0.25 | Moderate overlap |
| <0.15 | Poor overlap |

#### Strengths & Weaknesses

✅ **Strengths:**
- Fast computation
- Language-independent
- Well-established benchmark
- Good for measuring content coverage

❌ **Weaknesses:**
- No semantic understanding (synonyms treated as different)
- Ignores word order (except ROUGE-L)
- May penalize valid paraphrases

---

### BERTScore

**BERTScore** leverages pre-trained BERT contextual embeddings to compute semantic similarity between texts.

#### How It Works

1. Tokenize both generated and reference texts
2. Obtain contextual embeddings from BERT
3. Compute cosine similarity between token embeddings
4. Aggregate similarities using greedy matching

#### Components

- **Precision**: Average maximum similarity of each generated token to reference tokens
- **Recall**: Average maximum similarity of each reference token to generated tokens
- **F1**: Harmonic mean of precision and recall

#### Interpretation

| Score Range | Quality |
|-------------|---------|
| 0.90+ | Excellent semantic match |
| 0.85-0.90 | Good semantic match |
| 0.80-0.85 | Moderate semantic match |
| <0.80 | Poor semantic match |

#### Strengths & Weaknesses

✅ **Strengths:**
- Captures semantic similarity (synonyms, paraphrases)
- Contextual understanding
- Correlates well with human judgment

❌ **Weaknesses:**
- Computationally expensive (requires GPU for speed)
- Model-dependent results
- May be biased toward BERT's training data

---

## Classification Metrics

For single-label classification tasks (e.g., article type classification).

### Accuracy

The proportion of correct predictions:

```
Accuracy = Correct Predictions / Total Predictions
```

### Per-Class Accuracy

Accuracy computed separately for each class:

```
Class_Accuracy[c] = Correct_c / Total_c
```

This helps identify if the model struggles with specific classes.

### Class Distribution

Counts of each class in the reference data, useful for understanding class imbalance.

#### Interpretation

| Accuracy | Quality |
|----------|---------|
| 90%+ | Good ✅ |
| 80-90% | Acceptable 🟡 |
| <80% | Needs Improvement 🔴 |

---

## Multi-label Tag Metrics

For multi-label classification tasks (e.g., tag/topic extraction).

### Exact Match Accuracy

Percentage of samples where the predicted tag set exactly matches the reference tag set:

```
Exact_Match = Σ(pred_set == ref_set) / N
```

*This is a strict metric—partial matches count as wrong.*

### Jaccard Similarity

Measures the overlap between predicted and reference tag sets:

```
Jaccard = |Pred ∩ Ref| / |Pred ∪ Ref|
```

| Value | Interpretation |
|-------|----------------|
| 1.0 | Perfect match |
| 0.5 | Half overlap |
| 0.0 | No overlap |

### Precision, Recall, F1

Adapted for multi-label:

- **Precision**: What fraction of predicted tags are correct?
  ```
  Precision = |Pred ∩ Ref| / |Pred|
  ```

- **Recall**: What fraction of reference tags were predicted?
  ```
  Recall = |Pred ∩ Ref| / |Ref|
  ```

- **F1**: Harmonic mean
  ```
  F1 = 2 × (Precision × Recall) / (Precision + Recall)
  ```

#### Interpretation

| F1 Score | Jaccard | Quality |
|----------|---------|---------|
| 0.60+ | 0.50+ | Good ✅ |
| 0.40-0.60 | 0.35-0.50 | Acceptable 🟡 |
| <0.40 | <0.35 | Needs Improvement 🔴 |

---

## Length Metrics

Measures the length characteristics of generated text compared to references.

### Metrics Provided

| Metric | Description |
|--------|-------------|
| `avg_pred_length` | Average word count of predictions |
| `avg_ref_length` | Average word count of references |
| `avg_length_ratio` | Mean of (pred_length / ref_length) |
| `std_length_ratio` | Standard deviation of length ratios |

### Interpretation

| Length Ratio | Interpretation |
|--------------|----------------|
| ~1.0 | Similar length to reference |
| >1.0 | Predictions are longer |
| <1.0 | Predictions are shorter |

*High standard deviation indicates inconsistent output lengths.*

---

## Metrics Comparison

| Metric | Speed | Synonyms | Semantics | Best For |
|--------|-------|----------|-----------|----------|
| ROUGE | Fast ⚡ | ❌ | ❌ | Summarization, recall-focus |
| BERTScore | Slow 🐢 | ✅ | ✅ | Semantic similarity |

### When to Use Each

```
┌─────────────────────────────────────────────────────────────┐
│                    METRIC SELECTION GUIDE                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Need semantic understanding?                               │
│  ├── Yes → BERTScore                                        │
│  └── No → Continue below                                    │
│                                                             │
│  Task type?                                                 │
│  ├── Summarization → ROUGE-L (recall matters)               │
│  ├── Title Generation → ROUGE-1 (precision matters)         │
│  └── General → ROUGE + BERTScore                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Quality Thresholds

Pre-defined thresholds for quality assessment:

### Summarization

| Rating | ROUGE-L F1 |
|--------|------------|
| Good ✅ | ≥ 0.30 |
| Acceptable 🟡 | ≥ 0.20 |
| Needs Improvement 🔴 | < 0.20 |

### Title Generation

| Rating | ROUGE-1 F1 |
|--------|------------|
| Good ✅ | ≥ 0.25 |
| Acceptable 🟡 | ≥ 0.15 |
| Needs Improvement 🔴 | < 0.15 |

*Note: Title generation typically has lower overlap due to creative freedom.*

### Tag Extraction

| Rating | F1 | Jaccard |
|--------|-----|---------|
| Good ✅ | ≥ 0.60 | ≥ 0.50 |
| Acceptable 🟡 | ≥ 0.40 | ≥ 0.35 |
| Needs Improvement 🔴 | < 0.40 | < 0.35 |

### Type Classification

| Rating | Accuracy |
|--------|----------|
| Good ✅ | ≥ 90% |
| Acceptable 🟡 | ≥ 80% |
| Needs Improvement 🔴 | < 80% |

---

## Task-Specific Recommendations

### For Summarization

**Primary Metrics:** ROUGE-L F1, BERTScore F1

- ROUGE-L captures the longest matching sequence (sentence structure)
- BERTScore ensures semantic preservation
- Check length ratio to ensure appropriate compression

### For Title Generation

**Primary Metrics:** ROUGE-1 F1, BERTScore F1

- ROUGE-1 captures key word overlap
- BERTScore verifies the title captures the essence

### For Tag Extraction

**Primary Metrics:** F1, Jaccard Similarity

- F1 balances finding all tags vs. precision
- Jaccard is intuitive for set comparison
- Exact match is too strict for most use cases

### For Type Classification

**Primary Metrics:** Accuracy, Per-class Accuracy

- Overall accuracy for general performance
- Per-class accuracy to identify weak spots
- Check class distribution for imbalance issues

---

## Usage Example

```python
from chatbmw.evaluator import (
    load_and_evaluate,
    print_summary,
    assess_quality,
    save_metrics
)

# Load and evaluate results
metrics = load_and_evaluate(
    "test_results.json",
    compute_bertscore=True,
    verbose=True
)

# Print summary table
print_summary(metrics)

# Get quality ratings
ratings = assess_quality(metrics)
print(ratings)

# Save metrics to file
save_metrics(metrics, "evaluation_metrics.json")
```

---
