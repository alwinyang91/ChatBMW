# ChatBMW


A complete pipeline for fine-tuning Large Language Models (LLMs) on BMW news articles. This project includes data scraping, processing, model fine-tuning with Unsloth/LoRA, evaluation, and deployment to Hugging Face.

**🤗 Try the model: [Alwin-Yang/BMW-Llama-3.2-1B](https://huggingface.co/Alwin-Yang/BMW-Llama-3.2-1B)**

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- Nvidia GPU (recommended for training)

# 0. Installation

## 0.1. Install PyTorch (GPU-specific)

Choose the command matching your CUDA version.

Example for Nvidia 5090 GPU:
```bash
uv venv
source .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

## 0.2. Install dependencies
```bash
uv sync
```

# 1. Data Collection
### Option 1: Download from Hugging Face
```bash
# Using huggingface-cli
huggingface-cli download Alwin-Yang/bmw-articles --repo-type dataset --local-dir datasets/raw_data

# Or using Python
python -c "from datasets import load_dataset; load_dataset('Alwin-Yang/bmw-articles')"
```

> **Visit the [BMW-News-Hub](https://alwinyang91.github.io/BMW-News-Hub) website to explore the articles dataset!**

### Option 2: Scrape from scratch
```bash
# Basic scraping (metadata only)
python -m chatbmw.scraper

# Fetch full content, limit to 50 articles
python -m chatbmw.scraper --detail --limit 50

# Custom output path
python -m chatbmw.scraper -d -l 100 -o datasets/my_data
```

| Argument | Short | Description |
|----------|-------|-------------|
| `--detail` | `-d` | Fetch full article content (slower) |
| `--limit` | `-l` | Limit number of articles to scrape |
| `--output` | `-o` | Output path (default: `datasets/raw_data`) |
| `--exact` | - | Use exact output path without date/count suffix |

### Option 3: Collect articles with GitHub Actions
1. Fork the repository [alwinyang91/BMW-Articles](https://github.com/alwinyang91/BMW-Articles)

2. Configure your Hugging Face credentials as GitHub Actions secrets and variables in your forked repository.

3. Trigger the GitHub Actions workflow to automatically collect and upload articles to your Hugging Face dataset.

# 2. Data Processing

## 2.1 Clean the raw data
```bash
python -m chatbmw.data.cleaner -i datasets/raw_data/bmw_articles_20251230_162547_2000articles.jsonl
python -m chatbmw.data.cleaner -i datasets/raw_data/bmw_articles_20251230_162129_1500articles.jsonl
python -m chatbmw.data.cleaner -i datasets/raw_data/bmw_articles_20251230_155816_1000articles.jsonl
python -m chatbmw.data.cleaner -i datasets/raw_data/bmw_articles_20251230_141305_500articles.jsonl
```

| Argument | Short | Description |
|----------|-------|-------------|
| `--input` | `-i` | Path to input file (JSONL or JSON format) |
| `--output` | `-o` | Output path (default: `datasets/clean_data/{input_stem}_clean.jsonl`) |

## 2.2 Transform to chat format and split dataset for fine-tuning
```bash
python -m chatbmw.data.processor -i datasets/clean_data/bmw_articles_20251230_162547_2000articles_clean.jsonl -o datasets/chat_data_2000
python -m chatbmw.data.processor -i datasets/clean_data/bmw_articles_20251230_162129_1500articles_clean.jsonl -o datasets/chat_data_1500
python -m chatbmw.data.processor -i datasets/clean_data/bmw_articles_20251230_155816_1000articles_clean.jsonl -o datasets/chat_data_1000
python -m chatbmw.data.processor -i datasets/clean_data/bmw_articles_20251230_141305_500articles_clean.jsonl -o datasets/chat_data_500
```

| Argument | Short | Description |
|----------|-------|-------------|
| `--input` | `-i` | Path to input JSONL file (cleaned articles) |
| `--output` | `-o` | Output directory (default: `datasets/chat_data/`) |
| `--tasks` | - | Tasks to generate (default: all) |
| `--exclude-types` | - | Article types to exclude (default: `Fact & Figures`) |
| `--test-size` | - | Fraction for test split (default: 0.1) |
| `--val-size` | - | Fraction for validation split (default: 0.1) |
| `--seed` | - | Random seed (default: 42) |

>**Step by step process refer to `notebooks/Data_Process.ipynb`**

# 3. Model Fine-Tuning

To fine-tune the LLM using the CLI, first configure your training parameters in `train_config.yaml`.

Training metrics are logged to **TensorBoard** by default. To also enable [Weights & Biases](https://wandb.ai/) logging, create a `.env` file in the project root with your API key:

```
WANDB_API_KEY=your_api_key_here
```

```bash
python -m chatbmw.model.trainer
```

To view training metrics with TensorBoard:

```bash
tensorboard --logdir=YOUR_CHECKPOINT_DIR
```
>**Step by step process refer to `notebooks/Finetune_Llama-3.2-1B.ipynb`**


# 4. Model Evaluation

### Option 1: Evaluate from existing results file (fastest)
```bash
python -m chatbmw.model.evaluator \
    --results-file results/test_results.json \
    --output-dir results
```

### Option 2: Full evaluation (inference + metrics)
```bash
python -m chatbmw.model.evaluator \
    --model-path checkpoints/unsloth/Llama-3.2-1B-Instruct/merged_model \
    --test-data datasets/chat_data_2000/test_chat.jsonl \
    --output-dir results --save-results
```

### Option 3: Quick evaluation (skip BERTScore)
```bash
python -m chatbmw.model.evaluator \
    --results-file checkpoints/unsloth/Llama-3.2-1B-Instruct/test_results.json \
    --no-bertscore
```

| Argument | Description |
|----------|-------------|
| `--results-file` | Path to existing results JSON file (skip inference) |
| `--model-path` | Path to fine-tuned model (for running inference) |
| `--test-data` | Path to test dataset JSONL file (for running inference) |
| `--output-dir` | Directory to save output files (default: current directory) |
| `--save-results` | Save inference results to JSON file |
| `--save-metrics` | Save evaluation metrics to JSON file (default: True) |
| `--no-bertscore` | Skip BERTScore computation (faster) |
| `--max-new-tokens` | Maximum tokens to generate (default: 512) |
| `--quiet` | Reduce output verbosity |

>**Step by step process and visul refer to `notebooks/Evaulate_Llama-3.2-1B.ipynb`**


# 5. Publish Fine-Tuned Model to Hugging Face

## 5.1 Login to Hugging Face
```bash
huggingface-cli login
```
You'll need a [Hugging Face access token](https://huggingface.co/settings/tokens) with write permissions.

## 5.2 Upload the Model

**Option 1: Using CLI**
```bash
huggingface-cli upload YOUR_USERNAME/BMW-Llama-3.2-1B checkpoints/unsloth/Llama-3.2-1B-Instruct/merged_model
```

**Option 2: Using Python**

Refer to `notebooks/Upload_Llama-3.2-1B.ipynb`

