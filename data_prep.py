"""
data_prep.py — Dataset preparation utilities for LLM fine-tuning
Converts raw data into training-ready JSONL format
Supports: CSV, TXT, JSON, HuggingFace datasets, custom scrapers
"""

import json
import csv
import re
import random
import argparse
from pathlib import Path
from typing import Iterator


# ── Format converters ──────────────────────────────────────────────────────────
def csv_to_instruction(
    csv_path: str,
    instruction_col: str,
    output_col: str,
    input_col: str = "",
    system_prompt: str = "",
    out_path: str = "data/train.jsonl",
):
    """Convert a CSV with instruction/output columns to JSONL."""
    records = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            record = {
                "instruction": (system_prompt + " " if system_prompt else "") + row[instruction_col],
                "input": row.get(input_col, "") if input_col else "",
                "output": row[output_col],
            }
            records.append(record)

    _write_jsonl(records, out_path)
    print(f"✅ Converted {len(records)} rows → {out_path}")
    return records


def txt_to_chunks(
    txt_path: str,
    chunk_size: int = 512,
    overlap: int = 64,
    out_path: str = "data/train.jsonl",
):
    """Split a plain text file into overlapping chunks for domain adaptation."""
    text = Path(txt_path).read_text()
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        if len(chunk.split()) > 32:  # skip tiny chunks
            chunks.append({"text": chunk})
        start += chunk_size - overlap

    _write_jsonl(chunks, out_path)
    print(f"✅ Split into {len(chunks)} chunks → {out_path}")
    return chunks


def qa_pairs_to_instruction(
    qa_list: list[dict],   # [{"question": ..., "answer": ...}]
    system_prompt: str = "Answer the following question accurately and concisely.",
    out_path: str = "data/train.jsonl",
):
    """Convert Q&A pairs to instruction format."""
    records = [
        {
            "instruction": system_prompt,
            "input": qa["question"],
            "output": qa["answer"],
        }
        for qa in qa_list
    ]
    _write_jsonl(records, out_path)
    print(f"✅ Converted {len(records)} Q&A pairs → {out_path}")
    return records


def chat_to_instruction(
    conversations: list[list[dict]],  # [[{"role": "user", "content": ...}, ...]]
    out_path: str = "data/train.jsonl",
):
    """Convert multi-turn conversations to single-turn instruction format."""
    records = []
    for conv in conversations:
        for i in range(0, len(conv) - 1, 2):
            if conv[i]["role"] == "user" and i + 1 < len(conv):
                records.append({
                    "instruction": conv[i]["content"],
                    "input": "",
                    "output": conv[i + 1]["content"],
                })
    _write_jsonl(records, out_path)
    print(f"✅ Converted {len(records)} turns → {out_path}")
    return records


# ── Dataset quality filters ────────────────────────────────────────────────────
def filter_dataset(
    records: list[dict],
    min_output_words: int = 5,
    max_output_words: int = 500,
    remove_duplicates: bool = True,
    remove_pii: bool = True,
) -> list[dict]:
    """Apply quality filters to training data."""
    seen = set()
    cleaned = []

    for r in records:
        output = r.get("output", r.get("text", ""))
        words  = len(output.split())

        # Length filter
        if words < min_output_words or words > max_output_words:
            continue

        # Deduplication
        if remove_duplicates:
            key = output[:100].strip().lower()
            if key in seen:
                continue
            seen.add(key)

        # Basic PII removal
        if remove_pii:
            output = _remove_pii(output)
            r = {**r, "output": output}

        cleaned.append(r)

    removed = len(records) - len(cleaned)
    print(f"🧹 Filtered: {len(records)} → {len(cleaned)} samples ({removed} removed)")
    return cleaned


def _remove_pii(text: str) -> str:
    """Remove common PII patterns from text."""
    text = re.sub(r'\b[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}\b', '[EMAIL]', text)
    text = re.sub(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', '[PHONE]', text)
    text = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', '[IP]', text)
    text = re.sub(r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14})\b', '[CARD]', text)
    return text


# ── Sample dataset generator ───────────────────────────────────────────────────
def generate_sample_dataset(
    task: str = "qa",
    n_samples: int = 100,
    out_path: str = "data/train.jsonl",
) -> list[dict]:
    """
    Generate a small sample dataset for testing the pipeline.
    Tasks: qa | summarization | code | classification
    """
    records = []

    if task == "qa":
        templates = [
            ("What is machine learning?",
             "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed."),
            ("Explain gradient descent.",
             "Gradient descent is an optimization algorithm that iteratively adjusts model parameters in the direction that reduces the loss function, following the negative gradient."),
            ("What is a transformer model?",
             "A transformer is a deep learning architecture based on self-attention mechanisms, enabling parallel processing of sequences and capturing long-range dependencies effectively."),
            ("What is overfitting?",
             "Overfitting occurs when a model learns the training data too well, including noise, resulting in poor generalization to unseen data."),
            ("What is fine-tuning?",
             "Fine-tuning is the process of taking a pre-trained model and continuing training it on a smaller, task-specific dataset to adapt it for a particular use case."),
        ]
        for i in range(n_samples):
            q, a = templates[i % len(templates)]
            records.append({
                "instruction": "Answer the following AI/ML question clearly and accurately.",
                "input": q + f" (variant {i})",
                "output": a,
            })

    elif task == "code":
        templates = [
            ("Write a Python function to compute fibonacci numbers.",
             "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)"),
            ("Write a function to check if a string is a palindrome.",
             "def is_palindrome(s):\n    s = s.lower().replace(' ', '')\n    return s == s[::-1]"),
        ]
        for i in range(n_samples):
            inst, code = templates[i % len(templates)]
            records.append({
                "instruction": inst + f" (example {i})",
                "input": "",
                "output": f"```python\n{code}\n```",
            })

    elif task == "summarization":
        for i in range(n_samples):
            records.append({
                "instruction": "Summarize the following text in one sentence.",
                "input": f"This is a sample text about topic {i}. " * 10,
                "output": f"This text discusses topic {i} across multiple related points.",
            })

    _write_jsonl(records, out_path)
    print(f"✅ Generated {len(records)} sample records → {out_path}")
    return records


def _write_jsonl(records: list[dict], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def train_val_split(
    in_path: str,
    train_path: str = "data/train.jsonl",
    val_path: str   = "data/val.jsonl",
    val_ratio: float = 0.1,
    seed: int = 42,
):
    """Split a JSONL file into train and validation sets."""
    with open(in_path) as f:
        records = [json.loads(l) for l in f if l.strip()]

    random.seed(seed)
    random.shuffle(records)
    split = int(len(records) * (1 - val_ratio))
    train_records = records[:split]
    val_records   = records[split:]

    _write_jsonl(train_records, train_path)
    _write_jsonl(val_records,   val_path)
    print(f"✅ Split: {len(train_records)} train | {len(val_records)} val")


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset preparation for LLM fine-tuning")
    parser.add_argument("--generate-sample", action="store_true", help="Generate a sample dataset")
    parser.add_argument("--task",   type=str, default="qa", choices=["qa","code","summarization"])
    parser.add_argument("--n",      type=int, default=100, help="Number of samples to generate")
    parser.add_argument("--output", type=str, default="data/train.jsonl")
    parser.add_argument("--csv",    type=str, help="Convert CSV to JSONL")
    parser.add_argument("--txt",    type=str, help="Chunk TXT file for domain adaptation")
    parser.add_argument("--inst-col",  type=str, default="instruction")
    parser.add_argument("--output-col",type=str, default="output")
    args = parser.parse_args()

    if args.generate_sample:
        generate_sample_dataset(args.task, args.n, args.output)
    elif args.csv:
        csv_to_instruction(args.csv, args.inst_col, args.output_col, out_path=args.output)
    elif args.txt:
        txt_to_chunks(args.txt, out_path=args.output)
