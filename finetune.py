"""
finetune.py — LLM Fine-tuning Pipeline
Full pipeline: data prep → tokenization → LoRA/QLoRA training → evaluation → export
Uses HuggingFace Transformers + PEFT (Parameter Efficient Fine-Tuning)
Runs on FREE hardware: CPU or single consumer GPU (even 8GB VRAM)

Supported techniques:
  - LoRA  (Low-Rank Adaptation)        — fine-tune <1% of params
  - QLoRA (Quantized LoRA)             — 4-bit quantization + LoRA
  - Full fine-tuning                    — for small models only
  - Instruction tuning                  — Alpaca/ChatML format
  - Domain adaptation                   — raw text continuation
"""

import os
import json
import time
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
    PeftModel,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Configuration dataclass ────────────────────────────────────────────────────
@dataclass
class FinetuneConfig:
    """All hyperparameters and settings in one place."""

    # Model
    base_model: str         = "microsoft/phi-2"   # change to any HF model
    output_dir: str         = "outputs/finetuned"
    resume_from: str        = ""                  # checkpoint path to resume

    # Technique
    technique: str          = "qlora"             # lora | qlora | full
    lora_r: int             = 16                  # LoRA rank
    lora_alpha: int         = 32                  # LoRA scaling = alpha/r
    lora_dropout: float     = 0.05
    lora_target_modules: list = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )

    # Data
    dataset_path: str       = "data/train.jsonl"
    val_split: float        = 0.1                 # fraction for validation
    data_format: str        = "instruction"        # instruction | text | chat
    max_seq_len: int        = 512
    num_train_samples: int  = -1                  # -1 = all

    # Training
    epochs: int             = 3
    batch_size: int         = 4
    grad_accumulation: int  = 4                   # effective batch = 4*4=16
    learning_rate: float    = 2e-4
    warmup_ratio: float     = 0.03
    lr_scheduler: str       = "cosine"
    weight_decay: float     = 0.01
    max_grad_norm: float    = 1.0
    fp16: bool              = False               # auto-set based on GPU
    bf16: bool              = False               # prefer bf16 on Ampere+

    # Evaluation
    eval_steps: int         = 50
    save_steps: int         = 100
    logging_steps: int      = 10
    early_stopping_patience: int = 3

    # Export
    push_to_hub: bool       = False
    hub_repo: str           = ""
    merge_weights: bool     = True               # merge LoRA into base model


# ── Data formatting ────────────────────────────────────────────────────────────
INSTRUCTION_TEMPLATE = """### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

CHAT_TEMPLATE = """<|user|>
{instruction}
<|assistant|>
{output}"""

def format_instruction(sample: dict, fmt: str) -> str:
    if fmt == "instruction":
        return INSTRUCTION_TEMPLATE.format(
            instruction=sample.get("instruction", ""),
            input=sample.get("input", ""),
            output=sample.get("output", ""),
        )
    elif fmt == "chat":
        return CHAT_TEMPLATE.format(
            instruction=sample.get("instruction", sample.get("input", "")),
            output=sample.get("output", ""),
        )
    else:  # raw text
        return sample.get("text", sample.get("content", ""))


def load_and_prepare_data(cfg: FinetuneConfig, tokenizer) -> tuple[Dataset, Dataset]:
    """Load JSONL/JSON/HuggingFace dataset and tokenize."""
    path = Path(cfg.dataset_path)

    # Load data
    if path.suffix == ".jsonl":
        with open(path) as f:
            records = [json.loads(l) for l in f if l.strip()]
    elif path.suffix == ".json":
        with open(path) as f:
            records = json.load(f)
            if isinstance(records, dict):
                records = records.get("data", list(records.values())[0])
    elif ":" in cfg.dataset_path:  # HuggingFace dataset id e.g. "tatsu-lab/alpaca"
        ds = load_dataset(cfg.dataset_path, split="train")
        records = list(ds)
    else:
        raise ValueError(f"Unsupported dataset format: {cfg.dataset_path}")

    if cfg.num_train_samples > 0:
        records = records[:cfg.num_train_samples]

    log.info(f"Loaded {len(records)} samples from {cfg.dataset_path}")

    # Format and tokenize
    def tokenize(sample):
        text = format_instruction(sample, cfg.data_format)
        tokens = tokenizer(
            text,
            truncation=True,
            max_length=cfg.max_seq_len,
            padding="max_length",
            return_tensors=None,
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    dataset = Dataset.from_list(records)
    tokenized = dataset.map(tokenize, remove_columns=dataset.column_names)

    # Train/val split
    split = tokenized.train_test_split(test_size=cfg.val_split, seed=42)
    return split["train"], split["test"]


# ── Model loading ──────────────────────────────────────────────────────────────
def load_model_and_tokenizer(cfg: FinetuneConfig):
    log.info(f"Loading base model: {cfg.base_model}")
    log.info(f"Technique: {cfg.technique.upper()}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Quantization config for QLoRA
    if cfg.technique == "qlora":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",        # NormalFloat4 — best quality
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,   # double quantization saves memory
        )
        model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        dtype = torch.float16 if cfg.fp16 else (torch.bfloat16 if cfg.bf16 else torch.float32)
        model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )

    # Apply LoRA
    if cfg.technique in ("lora", "qlora"):
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.lora_target_modules,
            bias="none",
            inference_mode=False,
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    return model, tokenizer


# ── Training ───────────────────────────────────────────────────────────────────
def build_training_args(cfg: FinetuneConfig) -> TrainingArguments:
    # Auto-detect best precision
    fp16 = cfg.fp16
    bf16 = cfg.bf16
    if not fp16 and not bf16 and torch.cuda.is_available():
        if torch.cuda.get_device_capability()[0] >= 8:  # Ampere+
            bf16 = True
        else:
            fp16 = True

    return TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accumulation,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        fp16=fp16,
        bf16=bf16,
        evaluation_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        logging_steps=cfg.logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=3,
        report_to="none",           # swap for "wandb" or "tensorboard"
        dataloader_num_workers=0,
        remove_unused_columns=False,
        push_to_hub=cfg.push_to_hub,
        hub_model_id=cfg.hub_repo if cfg.push_to_hub else None,
    )


def train(cfg: FinetuneConfig):
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Save config
    with open(f"{cfg.output_dir}/finetune_config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    model, tokenizer = load_model_and_tokenizer(cfg)
    train_ds, val_ds = load_and_prepare_data(cfg, tokenizer)

    log.info(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    trainer = Trainer(
        model=model,
        args=build_training_args(cfg),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)],
    )

    log.info("Starting training...")
    t0 = time.time()

    if cfg.resume_from:
        trainer.train(resume_from_checkpoint=cfg.resume_from)
    else:
        trainer.train()

    elapsed = (time.time() - t0) / 60
    log.info(f"Training complete in {elapsed:.1f} minutes")

    # Save final model
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    # Optionally merge LoRA weights into base model
    if cfg.technique in ("lora", "qlora") and cfg.merge_weights:
        log.info("Merging LoRA weights into base model...")
        base = AutoModelForCausalLM.from_pretrained(
            cfg.base_model,
            torch_dtype=torch.float16,
            device_map="cpu",
            trust_remote_code=True,
        )
        merged = PeftModel.from_pretrained(base, cfg.output_dir)
        merged = merged.merge_and_unload()
        merged_path = cfg.output_dir + "/merged"
        merged.save_pretrained(merged_path)
        tokenizer.save_pretrained(merged_path)
        log.info(f"Merged model saved to: {merged_path}")

    log.info(f"Model saved to: {cfg.output_dir}")
    return trainer


# ── Inference / evaluation ────────────────────────────────────────────────────
def generate(
    model_path: str,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    use_lora: bool = True,
) -> str:
    """Generate text from fine-tuned model."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if use_lora:
        base_cfg = json.load(open(f"{model_path}/finetune_config.json"))
        base = AutoModelForCausalLM.from_pretrained(
            base_cfg["base_model"],
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Fine-tuning Pipeline")
    parser.add_argument("--config",       type=str, help="Path to JSON config file")
    parser.add_argument("--model",        type=str, help="Base model ID")
    parser.add_argument("--data",         type=str, help="Dataset path or HF dataset ID")
    parser.add_argument("--technique",    type=str, default="qlora", choices=["lora", "qlora", "full"])
    parser.add_argument("--epochs",       type=int, default=3)
    parser.add_argument("--batch-size",   type=int, default=4)
    parser.add_argument("--lr",           type=float, default=2e-4)
    parser.add_argument("--lora-r",       type=int, default=16)
    parser.add_argument("--max-seq-len",  type=int, default=512)
    parser.add_argument("--output",       type=str, default="outputs/finetuned")
    parser.add_argument("--merge",        action="store_true", help="Merge LoRA weights after training")
    parser.add_argument("--generate",     type=str, help="Run inference with this prompt after training")
    args = parser.parse_args()

    if args.config:
        with open(args.config) as f:
            cfg = FinetuneConfig(**json.load(f))
    else:
        cfg = FinetuneConfig(
            base_model=args.model or "microsoft/phi-2",
            dataset_path=args.data or "data/train.jsonl",
            technique=args.technique,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            lora_r=args.lora_r,
            max_seq_len=args.max_seq_len,
            output_dir=args.output,
            merge_weights=args.merge,
        )

    trainer = train(cfg)

    if args.generate:
        response = generate(cfg.output_dir, args.generate)
        print(f"\nPrompt: {args.generate}")
        print(f"Response: {response}")
