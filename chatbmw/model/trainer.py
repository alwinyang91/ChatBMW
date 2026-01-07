import os
import sys
import json
import random
from pathlib import Path

import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

import wandb
import weave
from datasets import load_dataset, Dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from trl import SFTConfig, SFTTrainer
from transformers import EarlyStoppingCallback, DataCollatorForSeq2Seq


# Load configuration from YAML file
def load_config(config_path: str | Path | None = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / "train_config.yaml"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    return config


def main():
    """Main training function."""
    # Load environment variables from .env file
    load_dotenv()
    
    # Load config
    config = load_config()

    # Extract configuration values
    MODEL_NAME = config["model"]["name"]
    MAX_SEQ_LENGTH = config["model"]["max_seq_length"]
    LOAD_IN_4BIT = config["model"]["load_in_4bit"]
    LOAD_IN_8BIT = config["model"]["load_in_8bit"]
    FULL_FINETUNING = config["model"]["full_finetuning"]
    DTYPE = config["model"].get("dtype")  # Can be None
    HF_TOKEN = config["model"].get("token")  # Optional

    # Format paths with model name
    CHECKPOINT_DIR = config["paths"]["checkpoint_dir"].format(model_name=MODEL_NAME)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    LORA_MODEL_PATH = config["paths"]["lora_model_path"].format(
        checkpoint_dir=CHECKPOINT_DIR
    )
    MERGED_MODEL_PATH = config["paths"]["merged_model_path"].format(
        checkpoint_dir=CHECKPOINT_DIR
    )
    os.makedirs(LORA_MODEL_PATH, exist_ok=True)
    os.makedirs(MERGED_MODEL_PATH, exist_ok=True)

    TRAIN_CHAT_DATA_FILE_NAME = config["data"]["train_file"]
    VAL_CHAT_DATA_FILE_NAME = config["data"]["val_file"]
    TEST_CHAT_DATA_FILE_NAME = config["data"]["test_file"]
    FORMATTING_BATCH_SIZE = config["data"]["formatting_batch_size"]

    # Wandb configuration - prioritize .env file, fall back to config
    WANDB_API_KEY = os.getenv("WANDB_API_KEY") or config["wandb"].get("api_key")
    WANDB_PROJECT = config["wandb"].get("project")
    WANDB_ENTITY = config["wandb"].get("entity")
    WANDB_RUN_NAME = config["wandb"].get("run_name")
    WANDB_CONFIG = config["wandb"].get("config", {})
    
    # Determine report_to: always use tensorboard, add wandb if api_key exists
    if WANDB_API_KEY:
        REPORT_TO = ["tensorboard", "wandb"]
    else:
        REPORT_TO = ["tensorboard"]

    dataset_train = load_dataset("json", data_files=TRAIN_CHAT_DATA_FILE_NAME, split="train")
    dataset_val = load_dataset("json", data_files=VAL_CHAT_DATA_FILE_NAME, split="train")
    dataset_test = load_dataset("json", data_files=TEST_CHAT_DATA_FILE_NAME, split="train")

    train_ds = Dataset.from_list(dataset_train)
    val_ds = Dataset.from_list(dataset_val)
    test_ds = Dataset.from_list(dataset_test)

    base_model, base_tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME, 
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = DTYPE,  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
        load_in_4bit = LOAD_IN_4BIT,
        load_in_8bit = LOAD_IN_8BIT,
        full_finetuning = FULL_FINETUNING,
        token = HF_TOKEN if HF_TOKEN else None,  # use one if using gated models like meta-llama/Llama-2-7b-hf
    )

    tokenizer = get_chat_template(
        base_tokenizer,
        chat_template = config["chat_template"]["name"],  # Use conversational format for Instruct models
    )

    def formatting_chat_prompts_func(examples):
        """
        Format chat conversations for training.
        Input: examples with 'messages' field containing list of {role, content} dicts
        Output: formatted text strings with chat template applied
        """
        messages_list = examples["messages"]
        
        texts = []
        for messages in messages_list:
            # Apply chat template to the messages
            text = base_tokenizer.apply_chat_template(
                messages,
                tokenize = False,
                add_generation_prompt = False
            )
            texts.append(text)
        
        return {"text": texts}


    train_formatted_ds = train_ds.map(
        formatting_chat_prompts_func,
        batched=True,
        batch_size=FORMATTING_BATCH_SIZE,
        desc="Formatting train dataset"
    )
    val_formatted_ds = val_ds.map(
        formatting_chat_prompts_func,
        batched=True,
        batch_size=FORMATTING_BATCH_SIZE,
        desc="Formatting validation dataset"
    )

    # Filter out samples that are too long (where assistant response would be truncated)
    # This prevents eval_loss from being nan due to all labels being -100
    response_part = config["response_training"]["response_part"]
    response_tokens = tokenizer(response_part, add_special_tokens=False)["input_ids"]
    
    def filter_long_samples(example):
        """Keep only samples where assistant response won't be truncated."""
        tokens = tokenizer(example["text"], add_special_tokens=False)["input_ids"]
        if len(tokens) <= MAX_SEQ_LENGTH:
            return True
        # Find where assistant response starts
        for i in range(len(tokens) - len(response_tokens)):
            if tokens[i:i+len(response_tokens)] == response_tokens:
                return i < MAX_SEQ_LENGTH  # Keep if assistant starts before truncation
        return True  # Keep if no assistant marker found (shouldn't happen)
    
    train_before = len(train_formatted_ds)
    val_before = len(val_formatted_ds)
    
    train_formatted_ds = train_formatted_ds.filter(filter_long_samples, desc="Filtering long train samples")
    val_formatted_ds = val_formatted_ds.filter(filter_long_samples, desc="Filtering long val samples")
    
    train_filtered = train_before - len(train_formatted_ds)
    val_filtered = val_before - len(val_formatted_ds)
    if train_filtered > 0 or val_filtered > 0:
        print(f"Filtered out {train_filtered} train and {val_filtered} val samples (too long, assistant would be truncated)")

    lora_config = config["lora"]
    finetuned_model = FastLanguageModel.get_peft_model(
        base_model,
        r = lora_config["r"],
        target_modules = lora_config["target_modules"],
        lora_alpha = lora_config["lora_alpha"],
        lora_dropout = lora_config["lora_dropout"],
        bias = lora_config["bias"],
        use_gradient_checkpointing = lora_config["use_gradient_checkpointing"],
        random_state = lora_config["random_state"],
        use_rslora = lora_config["use_rslora"],
        loftq_config = lora_config["loftq_config"],
    )

    # Initialize wandb only if api_key is provided
    if WANDB_API_KEY:
        os.environ['WANDB_API_KEY'] = WANDB_API_KEY
        wandb.init(
            project=WANDB_PROJECT,
            name=WANDB_RUN_NAME,
            entity=WANDB_ENTITY,  # None means use default account
            config=WANDB_CONFIG
        )



    training_config = config["training"]
    trainer = SFTTrainer(
        model = finetuned_model,
        tokenizer = tokenizer,
        train_dataset = train_formatted_ds,      # Chat format dataset with 'text' field
        eval_dataset = val_formatted_ds,          # Validation dataset
        dataset_text_field = "text",           # The field containing formatted text
        max_seq_length = MAX_SEQ_LENGTH,
        data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
        packing = training_config["packing"],  # Can make training 5x faster for short sequences.
        args = SFTConfig(
            per_device_train_batch_size = training_config["per_device_train_batch_size"],
            per_device_eval_batch_size = training_config["per_device_eval_batch_size"],
            eval_strategy = training_config["eval_strategy"],
            eval_steps = training_config["eval_steps"],
            gradient_accumulation_steps = training_config["gradient_accumulation_steps"],
            warmup_steps = training_config["warmup_steps"],
            num_train_epochs = training_config["num_train_epochs"],
            learning_rate = training_config["learning_rate"],
            logging_steps = training_config["logging_steps"],
            optim = training_config["optim"],
            weight_decay = training_config["weight_decay"],
            lr_scheduler_type = training_config["lr_scheduler_type"],
            seed = training_config["seed"],
            report_to = REPORT_TO,  # tensorboard by default, + wandb if api_key exists
            output_dir = CHECKPOINT_DIR,
            save_strategy = training_config["save_strategy"],
            save_steps = training_config["save_steps"],
            save_total_limit = training_config["save_total_limit"],
            load_best_model_at_end = training_config["load_best_model_at_end"],
            metric_for_best_model = training_config["metric_for_best_model"],
            greater_is_better = training_config["greater_is_better"],
        ),
    )


    response_training_config = config["response_training"]
    trainer = train_on_responses_only(
        trainer,
        instruction_part = response_training_config["instruction_part"],
        response_part = response_training_config["response_part"],
    )


    early_stopping_config = config["early_stopping"]
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience = early_stopping_config["patience"],
        early_stopping_threshold = early_stopping_config["threshold"],
    )
    trainer.add_callback(early_stopping_callback)

    trainer_stats = trainer.train()

    finetuned_model.save_pretrained(LORA_MODEL_PATH)
    base_tokenizer.save_pretrained(LORA_MODEL_PATH)

    # Merge LoRA weights into base model and save
    finetuned_model.save_pretrained_merged(
        MERGED_MODEL_PATH,
        base_tokenizer,
        save_method=config["save"]["merged_save_method"],  # Options: "merged_16bit", "merged_4bit", "lora"
    )

    print("Training completed successfully!")
    print(f"LoRA model saved to: {LORA_MODEL_PATH}")
    print(f"Merged model saved to: {MERGED_MODEL_PATH}")


if __name__ == "__main__":
    main()



