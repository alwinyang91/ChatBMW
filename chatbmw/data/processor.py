"""
Data processor for transforming BMW articles into chat template format for LLM training.
"""
import json
from pathlib import Path
from typing import Union, Optional

from datasets import Dataset
from datasets.features import ClassLabel


# Task-specific system prompts
SYSTEM_PROMPTS = {
    "summarization": "You are an expert at summarizing BMW news articles. Provide concise, informative summaries that capture the key points.",
    "tag_extraction": "You are an expert at analyzing BMW news articles. Extract and categorize the key topics and themes from the content.",
    "type_classification": "You are an expert at classifying BMW news articles. Identify the article type based on its content and structure.",
    "title_generation": "You are an expert at creating headlines for BMW news articles. Generate concise, informative, and engaging titles.",
}

# Task-specific user instructions
USER_INSTRUCTIONS = {
    "summarization": "Summarize the following BMW news article in a concise way.",
    "tag_extraction": "Extract and list the key topics and categories for the following BMW news article.",
    "type_classification": "Identify the type of the following BMW news article.",
    "title_generation": "Generate a concise and informative title for the following BMW news article.",
}



def create_chat_message(
    role: str,
    content: str,
) -> dict:
    """
    Create a single chat message.
    
    Args:
        role: The role of the message sender (system, user, or assistant).
        content: The content of the message.
        
    Returns:
        A dictionary representing the chat message.
    """
    return {"role": role, "content": content}


def create_chat_conversation(
    system_prompt: str,
    user_message: str,
    assistant_response: str,
) -> dict:
    """
    Create a complete chat conversation with system, user, and assistant messages.
    
    Args:
        system_prompt: The system prompt setting the context.
        user_message: The user's input/question.
        assistant_response: The assistant's response/output.
        
    Returns:
        A dictionary with a 'messages' key containing the conversation.
    """
    return {
        "messages": [
            create_chat_message("system", system_prompt),
            create_chat_message("user", user_message),
            create_chat_message("assistant", assistant_response),
        ]
    }


def article_to_chat_data(
    article: dict,
    tasks: Optional[list[str]] = None,
) -> list[dict]:
    """
    Convert a single article to chat template format for specified tasks.
    
    Args:
        article: The article dictionary containing title, content, summary, tags, type.
        tasks: List of tasks to generate. Options: 'summarization', 'tag_extraction',
               'type_classification', 'title_generation'. If None, all tasks are generated.
               
    Returns:
        A list of chat conversation dictionaries for the specified tasks.
    """
    if tasks is None:
        tasks = ["summarization", "tag_extraction", "type_classification", "title_generation"]
    
    # Extract article fields
    title = article.get("title", "")
    content = article.get("content", "")
    summary = article.get("summary", "")
    context = article.get("context", "")
    tags = article.get("tags", [])
    article_type = article.get("type", "")
    
    # For tasks 2-4: use summary plus content
    task_input = summary + content
    

    chat_data = []
    
    # Task 1: Summarization
    # Input: content only (we want to summarize the main text)
    if "summarization" in tasks and summary and content:
        user_message = f"{USER_INSTRUCTIONS['summarization']}\n\n{content}"
        chat_data.append(
            create_chat_conversation(
                system_prompt=SYSTEM_PROMPTS["summarization"],
                user_message=user_message,
                assistant_response=summary,
            )
        )

    # Task 2: Title generation
    # Input: context (or summary if no context), NOT title (avoid data leakage)
    if "title_generation" in tasks and title and task_input:
        user_message = f"{USER_INSTRUCTIONS['title_generation']}\n\n{task_input}"
        chat_data.append(
            create_chat_conversation(
                system_prompt=SYSTEM_PROMPTS["title_generation"],
                user_message=user_message,
                assistant_response=title,
            )
        )

    # Task 3: Tag extraction
    # Input: context (or summary if no context)
    if "tag_extraction" in tasks and tags and task_input:
        tags_str = ", ".join(tags) if isinstance(tags, list) else tags
        user_message = f"{USER_INSTRUCTIONS['tag_extraction']}\n\n{task_input}"
        chat_data.append(
            create_chat_conversation(
                system_prompt=SYSTEM_PROMPTS["tag_extraction"],
                user_message=user_message,
                assistant_response=tags_str,
            )
        )
    
    # Task 4: Article type classification
    # Input: context (or summary if no context)
    if "type_classification" in tasks and article_type and task_input:
        user_message = f"{USER_INSTRUCTIONS['type_classification']}\n\n{task_input}"
        chat_data.append(
            create_chat_conversation(
                system_prompt=SYSTEM_PROMPTS["type_classification"],
                user_message=user_message,
                assistant_response=article_type,
            )
        )
    
    return chat_data


def process_jsonl_to_chat(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    tasks: Optional[list[str]] = None,
) -> dict:
    """
    Process a JSONL file and convert to chat template format.
    
    Args:
        input_path: Path to input JSONL file containing articles.
        output_path: Path to output JSONL file for chat data.
        tasks: List of tasks to generate. If None, all tasks are generated.
        
    Returns:
        A dictionary with statistics about the processing.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    all_chat_data = []
    articles_processed = 0
    task_counts = {
        "summarization": 0,
        "tag_extraction": 0,
        "type_classification": 0,
        "title_generation": 0,
    }
    
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                article = json.loads(line)
                chat_data = article_to_chat_data(article, tasks)
                all_chat_data.extend(chat_data)
                articles_processed += 1
                
                # Count tasks
                for item in chat_data:
                    system_content = item["messages"][0]["content"]
                    for task_name, prompt in SYSTEM_PROMPTS.items():
                        if prompt == system_content:
                            task_counts[task_name] += 1
                            break
    
    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        for item in all_chat_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    return {
        "articles_processed": articles_processed,
        "total_conversations": len(all_chat_data),
        "task_counts": task_counts,
        "output_path": str(output_path),
    }


def process_jsonl_to_chat_by_task(
    input_path: Union[str, Path],
    output_dir: Union[str, Path],
    tasks: Optional[list[str]] = None,
) -> dict:
    """
    Process a JSONL file and create separate output files for each task.
    
    Args:
        input_path: Path to input JSONL file containing articles.
        output_dir: Directory to save output files.
        tasks: List of tasks to generate. If None, all tasks are generated.
        
    Returns:
        A dictionary with statistics about the processing.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if tasks is None:
        tasks = ["summarization", "tag_extraction", "type_classification", "title_generation"]
    
    # Collect data by task
    task_data = {task: [] for task in tasks}
    articles_processed = 0
    
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                article = json.loads(line)
                articles_processed += 1
                
                for task in tasks:
                    chat_data = article_to_chat_data(article, [task])
                    task_data[task].extend(chat_data)
    
    # Write separate files for each task
    output_paths = {}
    for task, data in task_data.items():
        output_path = output_dir / f"chat_{task}.jsonl"
        with open(output_path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        output_paths[task] = str(output_path)
    
    return {
        "articles_processed": articles_processed,
        "task_counts": {task: len(data) for task, data in task_data.items()},
        "output_paths": output_paths,
    }


def convert_dataset_to_chat(
    dataset: Dataset,
    tasks: Optional[list[str]] = None,
) -> list[dict]:
    """
    Convert a HuggingFace dataset to chat format using chatbmw processor.
    
    Args:
        dataset: HuggingFace dataset with article fields (title, content, summary, tags, type)
        tasks: List of tasks to generate. If None, all tasks are generated.
        
    Returns:
        List of chat conversation dictionaries
    """
    all_chat_data = []
    
    for i in range(len(dataset)):
        article = {
            "title": dataset[i].get("title", ""),
            "content": dataset[i].get("content", ""),
            "summary": dataset[i].get("summary", ""),
            "tags": dataset[i].get("tags", []),
            "type": dataset[i].get("type", ""),
        }
        
        # Convert article to chat format
        chat_data = article_to_chat_data(article, tasks)
        all_chat_data.extend(chat_data)
    
    return all_chat_data


def process_dataset_to_chat_splits(
    input_path: Union[str, Path],
    output_dir: Union[str, Path],
    exclude_types: Optional[list[str]] = None,
    test_size: float = 0.1,
    val_size: float = 0.1,
    seed: int = 42,
    tasks: Optional[list[str]] = None,
) -> dict:
    """
    Process a cleaned dataset file into train/val/test chat format splits.
    
    Args:
        input_path: Path to input JSONL file containing cleaned articles.
        output_dir: Directory to save output files (train_chat.jsonl, val_chat.jsonl, test_chat.jsonl).
        exclude_types: List of article types to exclude (e.g., ['Fact & Figures']).
        test_size: Fraction of data for test split.
        val_size: Fraction of remaining data for validation split.
        seed: Random seed for reproducibility.
        tasks: List of tasks to generate. If None, all tasks are generated.
        
    Returns:
        A dictionary with statistics about the processing.
    """
    from datasets import load_dataset
    
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load the dataset
    dataset = load_dataset("json", data_files=str(input_path), split="train")
    original_count = len(dataset)
    
    # Exclude certain types if specified
    if exclude_types:
        for exclude_type in exclude_types:
            dataset = dataset.filter(lambda x: x['type'] != exclude_type)
    
    filtered_count = len(dataset)
    
    # Get all unique type values for stratified splitting
    unique_types = sorted(set(str(t) for t in dataset['type']))
    
    # Add a label column for stratified splitting
    type_to_idx = {t: i for i, t in enumerate(unique_types)}
    dataset = dataset.map(lambda x: {'label': type_to_idx[str(x['type'])]})
    dataset = dataset.cast_column("label", ClassLabel(names=unique_types))
    
    # Try stratified split, fall back to random split if classes have too few samples
    use_stratified = True
    try:
        # Stratified split: train/test first, then train/val
        split_1 = dataset.train_test_split(test_size=test_size, seed=seed, stratify_by_column="label")
        test_ds = split_1["test"]
        train_val_dataset = split_1["train"]
        
        # Adjust val_size to be relative to the remaining data
        adjusted_val_size = val_size / (1 - test_size)
        split_2 = train_val_dataset.train_test_split(test_size=adjusted_val_size, seed=seed, stratify_by_column="label")
        train_ds = split_2["train"]
        val_ds = split_2["test"]
    except ValueError as e:
        if "Minimum class count error" in str(e) or "too few" in str(e):
            # Fall back to regular random split for small datasets
            use_stratified = False
            split_1 = dataset.train_test_split(test_size=test_size, seed=seed, shuffle=True)
            test_ds = split_1["test"]
            train_val_dataset = split_1["train"]
            
            adjusted_val_size = val_size / (1 - test_size)
            split_2 = train_val_dataset.train_test_split(test_size=adjusted_val_size, seed=seed, shuffle=True)
            train_ds = split_2["train"]
            val_ds = split_2["test"]
        else:
            raise
    
    # Convert to chat format
    train_chat = convert_dataset_to_chat(train_ds, tasks)
    val_chat = convert_dataset_to_chat(val_ds, tasks)
    test_chat = convert_dataset_to_chat(test_ds, tasks)
    
    # Create HuggingFace Datasets
    train_chat_ds = Dataset.from_list(train_chat)
    val_chat_ds = Dataset.from_list(val_chat)
    test_chat_ds = Dataset.from_list(test_chat)
    
    # Save to JSONL files
    train_path = output_dir / "train_chat.jsonl"
    val_path = output_dir / "val_chat.jsonl"
    test_path = output_dir / "test_chat.jsonl"
    
    with open(train_path, "w", encoding="utf-8") as f:
        for item in train_chat:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    with open(val_path, "w", encoding="utf-8") as f:
        for item in val_chat:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    with open(test_path, "w", encoding="utf-8") as f:
        for item in test_chat:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    return {
        "original_articles": original_count,
        "filtered_articles": filtered_count,
        "excluded_count": original_count - filtered_count,
        "train_articles": len(train_ds),
        "val_articles": len(val_ds),
        "test_articles": len(test_ds),
        "train_conversations": len(train_chat),
        "val_conversations": len(val_chat),
        "test_conversations": len(test_chat),
        "total_conversations": len(train_chat) + len(val_chat) + len(test_chat),
        "output_dir": str(output_dir),
        "unique_types": unique_types,
        "stratified_split": use_stratified,
    }


def main():
    """CLI entry point for data processing."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Transform BMW articles to chat format and split into train/val/test sets."
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to input JSONL file (cleaned articles)"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output directory (default: datasets/chat_data/)"
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["summarization", "tag_extraction", "type_classification", "title_generation"],
        default=["summarization", "tag_extraction", "type_classification", "title_generation"],
        help="Tasks to generate (default: all)"
    )
    parser.add_argument(
        "--exclude-types",
        nargs="+",
        default=["Fact & Figures"],
        help="Article types to exclude (default: 'Fact & Figures')"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.1,
        help="Fraction of data for test split (default: 0.1)"
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.1,
        help="Fraction of data for validation split (default: 0.1)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path("datasets/chat_data")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stats = process_dataset_to_chat_splits(
        input_path=input_path,
        output_dir=output_dir,
        exclude_types=args.exclude_types,
        tasks=args.tasks,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
    )
    
    print(f"Original articles: {stats['original_articles']}")
    print(f"After filtering: {stats['filtered_articles']} (excluded {stats['excluded_count']})")
    print(f"Split: train={stats['train_articles']}, val={stats['val_articles']}, test={stats['test_articles']}")
    print(f"Total conversations: {stats['total_conversations']}")
    print(f"  - Train: {stats['train_conversations']}")
    print(f"  - Val: {stats['val_conversations']}")
    print(f"  - Test: {stats['test_conversations']}")
    print(f"Stratified split: {stats['stratified_split']}")
    print(f"Output saved to: {stats['output_dir']}")


if __name__ == "__main__":
    main()

