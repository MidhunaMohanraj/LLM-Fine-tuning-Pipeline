# 🧬 LLM Fine-tuning Pipeline

<div align="center">

![Banner](https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=12,16,22&height=200&section=header&text=LLM%20Fine-tuning%20Pipeline&fontSize=44&fontColor=fff&animation=twinkling&fontAlignY=35&desc=LoRA%20%E2%80%A2%20QLoRA%20%E2%80%A2%20Instruction%20Tuning%20%E2%80%A2%20Domain%20Adaptation%20%E2%80%A2%20Training%20Dashboard&descAlignY=55&descSize=14)

<p>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black"/>
  <img src="https://img.shields.io/badge/PEFT-LoRA%20%2F%20QLoRA-7c3aed?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/Free%20Hardware-CPU%20%2F%208GB%20GPU-22C55E?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge"/>
</p>

<p>
  <b>A complete, production-grade LLM fine-tuning pipeline — data preparation, LoRA/QLoRA training, evaluation, inference, and a visual dashboard. Fine-tune any HuggingFace model on your own data with minimal hardware.</b>
</p>

</div>

---

## 🌟 What This Does

```
Your custom data (CSV / JSONL / TXT / HuggingFace dataset)
         │
         ▼
   Data Preparation  ─── filter, format, deduplicate, PII removal
         │
         ▼
   LoRA / QLoRA Training ─── parameter-efficient, runs on 8GB GPU
         │
         ▼
   Evaluation ─── loss curves, early stopping, validation metrics
         │
         ▼
   Weight Merging ─── merge LoRA adapters back into base model
         │
         ▼
   Inference Ready ─── use your model for generation
```

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔧 **QLoRA** | 4-bit quantization + LoRA — fine-tune 7B models on a single 8GB GPU |
| 🎯 **LoRA** | Standard Low-Rank Adaptation — train <1% of parameters |
| 📊 **Training Dashboard** | Streamlit UI with live loss curves, config builder, dataset studio |
| 📁 **Dataset Studio** | Generate samples, upload JSONL, visualize distributions, filter quality |
| 🧪 **Inference Playground** | Test prompts against fine-tuned models in the browser |
| 🔀 **Multiple Formats** | Instruction, chat, raw text — Alpaca/ChatML/plain |
| 🛑 **Early Stopping** | Automatically stops when validation loss plateaus |
| 🔗 **Weight Merging** | Merge LoRA adapters into base model for standalone deployment |
| ⚙️ **Config System** | JSON configs for reproducible experiments |
| 🧹 **Data Filtering** | PII removal, deduplication, length filtering |

---

## 🚀 Quick Start

### Install
```bash
git clone https://github.com/YOUR_USERNAME/llm-finetune-pipeline.git
cd llm-finetune-pipeline

python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# QLoRA requires CUDA:
pip install bitsandbytes>=0.43.0
```

### Generate sample data & train
```bash
# Step 1: Generate sample training data
python src/data_prep.py --generate-sample --task qa --n 200 --output data/train.jsonl

# Step 2: Train with QLoRA on TinyLlama (fits on 8GB GPU or CPU)
python src/finetune.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --data data/train.jsonl \
  --technique qlora \
  --epochs 3 \
  --lora-r 16 \
  --merge

# Step 3: Run inference
python src/finetune.py \
  --output outputs/finetuned \
  --generate "### Instruction:\nExplain gradient descent.\n\n### Response:"
```

### Launch the dashboard
```bash
streamlit run app.py
```

---

## 📊 Training Configurations

### Minimal (CPU / 4GB RAM) — for testing
```json
{
  "base_model": "facebook/opt-125m",
  "technique": "lora",
  "lora_r": 8,
  "batch_size": 2,
  "max_seq_len": 256,
  "epochs": 2
}
```

### Recommended (8GB GPU) — QLoRA
```json
{
  "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "technique": "qlora",
  "lora_r": 16,
  "batch_size": 4,
  "grad_accumulation": 4,
  "max_seq_len": 512,
  "epochs": 3
}
```

### Production (24GB GPU) — Larger model
```json
{
  "base_model": "mistralai/Mistral-7B-v0.1",
  "technique": "qlora",
  "lora_r": 32,
  "batch_size": 8,
  "grad_accumulation": 2,
  "max_seq_len": 2048,
  "epochs": 3
}
```

---

## 🧠 How LoRA Works

```
Standard fine-tuning:  Update W directly → billions of gradient updates
                       Needs 80GB+ VRAM for 7B model

LoRA:                  W stays frozen
                       Add A (d×r) and B (r×d) matrices
                       Only A and B are trained: r=16 → 0.03% of params!

QLoRA:                 W is 4-bit quantized (8x smaller)
                       A and B in BF16
                       Result: 7B model fits in 8GB!
```

### Memory comparison for 7B model

| Method | VRAM Needed | Trainable Params |
|---|---|---|
| Full FP32 | ~112 GB | 100% |
| Full FP16 | ~56 GB | 100% |
| LoRA FP16 | ~28 GB | ~0.5% |
| QLoRA INT4+LoRA | **~8 GB** | ~0.5% |

---

## 📁 Project Structure

```
llm-finetune-pipeline/
│
├── app.py                      # 🖥️ Streamlit dashboard
├── src/
│   ├── finetune.py             # 🧬 Core training pipeline
│   └── data_prep.py            # 📊 Dataset utilities
├── configs/
│   └── tinyllama_qlora.json    # ⚙️ Example config
├── data/
│   └── sample_train.jsonl      # 📦 10 sample examples
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 📋 Data Format

Your JSONL file — one JSON per line:

```jsonl
{"instruction": "Explain X", "input": "optional context", "output": "the answer"}
{"instruction": "Write code to...", "input": "", "output": "```python\n...```"}
```

**Supported formats:**
- `instruction` — Alpaca style (instruction + input + output)
- `chat` — ChatML style (user/assistant turns)
- `text` — Raw text for domain adaptation

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| [HuggingFace Transformers](https://huggingface.co/transformers) | Model loading, training, tokenization |
| [PEFT](https://github.com/huggingface/peft) | LoRA / QLoRA implementation |
| [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) | 4-bit quantization (QLoRA) |
| [Accelerate](https://huggingface.co/docs/accelerate) | Multi-GPU / mixed precision |
| [TRL](https://github.com/huggingface/trl) | Supervised fine-tuning helpers |
| [Streamlit](https://streamlit.io) | Visual dashboard |
| [Plotly](https://plotly.com) | Training curves |

---

## 🗺️ Roadmap

- [ ] W&B / TensorBoard integration for experiment tracking
- [ ] DPO (Direct Preference Optimization) support
- [ ] RLHF pipeline with reward model
- [ ] Multi-GPU training with FSDP / DeepSpeed
- [ ] Automated hyperparameter search (Optuna)
- [ ] One-click deploy to HuggingFace Hub
- [ ] Evaluation benchmarks (MMLU, HellaSwag, TruthfulQA)

---

## ⚠️ Notes

- QLoRA requires a CUDA GPU. For CPU-only, use `technique: lora` with a small model
- First run downloads model weights (~1-14GB depending on model)
- Use `TinyLlama/TinyLlama-1.1B-Chat-v1.0` for fastest testing

---

## 📄 License

MIT — see [LICENSE](LICENSE)

---

<div align="center">

**⭐ Star this repo if you find it useful!**

*From raw data to fine-tuned model — everything in one place.*

![Footer](https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=12,16,22&height=100&section=footer)

</div>
