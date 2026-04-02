"""
app.py — Streamlit Dashboard for LLM Fine-tuning Pipeline
Visual interface for configuring, launching, and monitoring fine-tuning runs.
Shows training curves, config builder, dataset stats, and inference playground.
"""

import streamlit as st
import json
import os
import sys
import subprocess
import random
import math
import time
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from data_prep import generate_sample_dataset, filter_dataset

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LLM Fine-tuning Pipeline",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .main { background: #07080d; }
  .hero {
    background: linear-gradient(135deg,#0a071a 0%,#07080d 55%,#071a0a 100%);
    border:1px solid #1a1a30; border-radius:16px;
    padding:32px 40px; text-align:center; margin-bottom:24px;
  }
  .hero h1 { font-size:38px; font-weight:700; color:#fff; margin:0 0 6px; }
  .hero p  { color:#64748b; font-size:14px; margin:0; }
  .stat-card {
    background:#0a0c14; border:1px solid #1a1a30;
    border-radius:10px; padding:16px; text-align:center;
  }
  .stat-val   { font-size:24px; font-weight:700; color:#a78bfa; font-family:'JetBrains Mono',monospace; }
  .stat-label { font-size:10px; color:#475569; text-transform:uppercase; letter-spacing:1.5px; margin-top:3px; }
  .config-box {
    background:#040507; border:1px solid #1a1a30; border-radius:8px;
    padding:14px 18px; font-family:'JetBrains Mono',monospace;
    font-size:12px; color:#7dd3fc; white-space:pre; overflow-x:auto; margin:8px 0;
  }
  .insight-box {
    background:#080c08; border-left:4px solid #22c55e;
    border-radius:0 10px 10px 0; padding:14px 18px;
    font-size:13px; color:#86efac; margin:8px 0; line-height:1.7;
  }
  div.stButton > button {
    background:linear-gradient(135deg,#4c1d95,#7c3aed);
    color:white; font-weight:700; border:none; border-radius:10px;
    padding:12px 28px; font-size:15px; width:100%;
  }
  div.stButton > button:hover { opacity:0.85; }
  .stSelectbox>div, .stTextInput input, .stTextArea textarea {
    background:#0a0c14!important; border-color:#1a1a30!important; color:#e2e8f0!important;
  }
</style>
""", unsafe_allow_html=True)

# ── Sidebar config builder ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧬 Fine-tuning Pipeline")
    st.markdown("---")

    st.markdown("### 🤖 Model")
    base_model = st.selectbox("Base model", [
        "microsoft/phi-2",
        "microsoft/Phi-3-mini-4k-instruct",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "facebook/opt-125m",
        "EleutherAI/pythia-410m",
        "google/gemma-2b",
        "mistralai/Mistral-7B-v0.1",
        "meta-llama/Llama-3.2-3B",
    ])

    st.markdown("### ⚙️ Technique")
    technique = st.selectbox("Fine-tuning method", ["qlora", "lora", "full"])
    lora_r    = st.slider("LoRA rank (r)", 4, 64, 16, step=4,
                          help="Higher = more capacity but more memory")
    lora_alpha = st.slider("LoRA alpha", 8, 128, 32, step=8,
                           help="Scaling factor. Usually 2x rank.")

    st.markdown("### 📦 Training")
    epochs        = st.slider("Epochs", 1, 10, 3)
    batch_size    = st.slider("Batch size per device", 1, 16, 4)
    grad_accum    = st.slider("Gradient accumulation", 1, 16, 4,
                               help=f"Effective batch = {batch_size}×{4} = {batch_size*4}")
    lr            = st.select_slider("Learning rate", [1e-5,2e-5,5e-5,1e-4,2e-4,5e-4], value=2e-4)
    max_seq_len   = st.select_slider("Max sequence length", [128,256,512,1024,2048], value=512)

    st.markdown("### 📊 Data")
    data_format   = st.selectbox("Data format", ["instruction", "chat", "text"])
    val_split     = st.slider("Validation split", 0.05, 0.3, 0.1)

    st.markdown("---")
    merge_weights = st.checkbox("Merge LoRA after training", value=True)

# ── Build config dict ──────────────────────────────────────────────────────────
config = {
    "base_model": base_model,
    "technique": technique,
    "lora_r": lora_r,
    "lora_alpha": lora_alpha,
    "epochs": epochs,
    "batch_size": batch_size,
    "grad_accumulation": grad_accum,
    "learning_rate": lr,
    "max_seq_len": max_seq_len,
    "data_format": data_format,
    "val_split": val_split,
    "merge_weights": merge_weights,
    "dataset_path": "data/train.jsonl",
    "output_dir": "outputs/finetuned",
}

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🧬 LLM Fine-tuning Pipeline</h1>
  <p>LoRA · QLoRA · Instruction Tuning · Domain Adaptation · Dataset Prep · Training Dashboard</p>
</div>
""", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Config & Setup", "📊 Dataset Studio", "🚀 Training Monitor",
    "🧪 Inference Playground", "📐 Architecture"
])

# ── Tab 1: Config ──────────────────────────────────────────────────────────────
with tab1:
    st.markdown("### ⚙️ Generated Configuration")

    # Memory estimation
    param_map = {
        "microsoft/phi-2": 2.7, "microsoft/Phi-3-mini-4k-instruct": 3.8,
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0": 1.1, "facebook/opt-125m": 0.125,
        "EleutherAI/pythia-410m": 0.41, "google/gemma-2b": 2.0,
        "mistralai/Mistral-7B-v0.1": 7.0, "meta-llama/Llama-3.2-3B": 3.0,
    }
    params_b = param_map.get(base_model, 3.0)
    bytes_per = {"qlora": 0.5, "lora": 2.0, "full": 4.0}[technique]
    model_mem = params_b * bytes_per
    lora_mem  = params_b * 0.01 * 4  # ~1% trainable params in FP32
    total_mem = model_mem + lora_mem + (batch_size * max_seq_len * 0.002)
    trainable_pct = 100 * (lora_r * 2) / (params_b * 1000) if technique != "full" else 100

    c1, c2, c3, c4 = st.columns(4)
    stats = [
        (f"{params_b}B",          "Model Parameters"),
        (f"~{total_mem:.1f} GB",  "Est. VRAM Needed"),
        (f"{trainable_pct:.2f}%", "Trainable Params"),
        (f"{batch_size*grad_accum}", "Effective Batch"),
    ]
    for col, (val, label) in zip([c1,c2,c3,c4], stats):
        with col:
            st.markdown(f'<div class="stat-card"><div class="stat-val">{val}</div><div class="stat-label">{label}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Config as JSON
    st.markdown("**Full config (save as `configs/my_run.json`):**")
    st.markdown(f'<div class="config-box">{json.dumps(config, indent=2)}</div>', unsafe_allow_html=True)

    col_dl, col_cmd = st.columns(2)
    with col_dl:
        st.download_button(
            "⬇️ Download Config JSON",
            data=json.dumps(config, indent=2),
            file_name="finetune_config.json",
            mime="application/json",
            use_container_width=True,
        )
    with col_cmd:
        cmd = (f"python src/finetune.py \\\n"
               f"  --model {base_model} \\\n"
               f"  --data data/train.jsonl \\\n"
               f"  --technique {technique} \\\n"
               f"  --epochs {epochs} \\\n"
               f"  --batch-size {batch_size} \\\n"
               f"  --lora-r {lora_r} \\\n"
               f"  --max-seq-len {max_seq_len}"
               + (" \\\n  --merge" if merge_weights else ""))
        st.download_button(
            "⬇️ Download Launch Script",
            data=f"#!/bin/bash\n{cmd}",
            file_name="run_finetune.sh",
            mime="text/plain",
            use_container_width=True,
        )

    st.markdown("**CLI launch command:**")
    st.code(cmd, language="bash")

    st.markdown("**Installation:**")
    st.code("""pip install -r requirements.txt

# For QLoRA (4-bit) — requires CUDA:
pip install bitsandbytes>=0.43.0""", language="bash")

    # Hardware recommendations
    st.markdown("### 💻 Hardware Recommendations")
    if total_mem <= 8:
        hw_rec = "✅ Fits on RTX 3080/4080 (8-10GB) or free Google Colab GPU"
        hw_col = "#22c55e"
    elif total_mem <= 16:
        hw_rec = "⚡ Needs RTX 3090/4090 (16-24GB) or A10G"
        hw_col = "#f59e0b"
    elif total_mem <= 40:
        hw_rec = "🖥️ Needs A100 40GB or H100 — use cloud (Lambda, RunPod, vast.ai)"
        hw_col = "#f97316"
    else:
        hw_rec = "🔴 Needs multi-GPU setup — consider reducing batch size or using more aggressive quantization"
        hw_col = "#ef4444"

    st.markdown(f'<div class="insight-box" style="border-left-color:{hw_col};">{hw_rec}</div>', unsafe_allow_html=True)

# ── Tab 2: Dataset Studio ──────────────────────────────────────────────────────
with tab2:
    st.markdown("### 📊 Dataset Studio")
    ds_col1, ds_col2 = st.columns([1, 1])

    with ds_col1:
        st.markdown("**Generate sample dataset for testing:**")
        task_type   = st.selectbox("Task type", ["qa", "code", "summarization"])
        n_samples   = st.slider("Number of samples", 10, 500, 100)
        gen_clicked = st.button("🔧 Generate Sample Dataset")

        if gen_clicked:
            os.makedirs("data", exist_ok=True)
            records = generate_sample_dataset(task_type, n_samples, "data/train.jsonl")
            records = filter_dataset(records)
            st.success(f"✅ Generated {len(records)} samples → data/train.jsonl")
            st.session_state["sample_records"] = records

        st.markdown("**Or upload your own JSONL:**")
        uploaded = st.file_uploader("Upload .jsonl file", type=["jsonl", "json"])
        if uploaded:
            content = uploaded.read().decode()
            records = [json.loads(l) for l in content.strip().split("\n") if l.strip()]
            st.success(f"✅ Loaded {len(records)} samples")
            st.session_state["sample_records"] = records

    with ds_col2:
        st.markdown("**Dataset format preview:**")
        st.markdown("Your JSONL file should look like this:")
        example = {
            "instruction": "Explain what a transformer model is.",
            "input": "",
            "output": "A transformer is a deep learning architecture based on self-attention..."
        }
        st.json(example)
        st.caption("Each line = one training example")

    records = st.session_state.get("sample_records", [])
    if records:
        st.markdown(f"### 📈 Dataset Statistics ({len(records)} samples)")
        output_lens = [len(r.get("output", r.get("text","")).split()) for r in records]
        input_lens  = [len(r.get("instruction","").split()) for r in records]

        s1, s2, s3, s4 = st.columns(4)
        ds_stats = [
            (len(records),                "Total Samples"),
            (f"{sum(output_lens)//len(output_lens)}", "Avg Output Words"),
            (max(output_lens),            "Max Output Words"),
            (sum(output_lens),            "Total Output Words"),
        ]
        for col, (val, label) in zip([s1,s2,s3,s4], ds_stats):
            with col:
                st.markdown(f'<div class="stat-card"><div class="stat-val">{val}</div><div class="stat-label">{label}</div></div>', unsafe_allow_html=True)

        fig = go.Figure()
        fig.add_trace(go.Histogram(x=output_lens, nbinsx=30, name="Output length",
                                   marker_color="#a78bfa", opacity=0.8))
        fig.update_layout(
            paper_bgcolor="#07080d", plot_bgcolor="#07080d", font_color="#94a3b8",
            xaxis=dict(title="Words", gridcolor="#1a2030"),
            yaxis=dict(title="Count", gridcolor="#1a2030"),
            height=260, margin=dict(t=20,b=20,l=10,r=10),
            title="Output Length Distribution",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Sample records:**")
        for r in records[:3]:
            st.json(r)

# ── Tab 3: Training Monitor ────────────────────────────────────────────────────
with tab3:
    st.markdown("### 🚀 Training Monitor")
    st.info("This dashboard shows a simulated training run. Connect to a real run by pointing to your trainer's log file or using Weights & Biases.")

    if st.button("▶️ Simulate Training Run"):
        # Simulate realistic training curves
        steps = list(range(0, 201, 10))
        base_loss = 2.5
        train_losses, val_losses, lrs = [], [], []

        for i, step in enumerate(steps):
            # Cosine decay with noise
            progress = step / max(steps)
            lr_val   = float(lr) * (1 + math.cos(math.pi * progress)) / 2
            t_loss   = base_loss * math.exp(-2.5 * progress) + 0.3 + random.gauss(0, 0.05)
            v_loss   = t_loss + 0.1 + random.gauss(0, 0.03)
            train_losses.append(max(0.2, t_loss))
            val_losses.append(max(0.25, v_loss))
            lrs.append(lr_val)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=steps, y=train_losses, name="Train Loss",
                                 line=dict(color="#a78bfa", width=2.5)))
        fig.add_trace(go.Scatter(x=steps, y=val_losses,   name="Val Loss",
                                 line=dict(color="#f59e0b", width=2.5)))
        fig.update_layout(
            paper_bgcolor="#07080d", plot_bgcolor="#07080d", font_color="#94a3b8",
            xaxis=dict(title="Step", gridcolor="#1a2030"),
            yaxis=dict(title="Loss", gridcolor="#1a2030"),
            height=360, margin=dict(t=20,b=20,l=10,r=10),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            title=f"Training Curves — {base_model.split('/')[-1]} | {technique.upper()} | lr={lr}",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Summary metrics
        m1,m2,m3,m4 = st.columns(4)
        run_stats = [
            (f"{min(train_losses):.3f}", "Best Train Loss"),
            (f"{min(val_losses):.3f}",  "Best Val Loss"),
            (f"{steps[-1]}",            "Total Steps"),
            (f"{epochs*10} min",        "Est. Train Time"),
        ]
        for col,(val,label) in zip([m1,m2,m3,m4], run_stats):
            with col:
                st.markdown(f'<div class="stat-card"><div class="stat-val">{val}</div><div class="stat-label">{label}</div></div>', unsafe_allow_html=True)

    st.markdown("### 📁 Connect to Real Training Logs")
    log_path = st.text_input("Path to trainer_state.json", placeholder="outputs/finetuned/trainer_state.json")
    if log_path and Path(log_path).exists():
        with open(log_path) as f:
            state = json.load(f)
        log_history = state.get("log_history", [])
        if log_history:
            df = pd.DataFrame(log_history)
            st.dataframe(df, use_container_width=True)

# ── Tab 4: Inference Playground ────────────────────────────────────────────────
with tab4:
    st.markdown("### 🧪 Inference Playground")
    st.markdown("Test prompts against your fine-tuned model. Paste a prompt to see what your model would generate.")

    col_prompt, col_settings = st.columns([2,1])
    with col_prompt:
        user_prompt = st.text_area(
            "Prompt",
            height=160,
            value="### Instruction:\nExplain the concept of gradient descent in simple terms.\n\n### Input:\n\n### Response:",
            help="Use the instruction template for instruction-tuned models",
        )
    with col_settings:
        temperature    = st.slider("Temperature", 0.0, 1.5, 0.7)
        top_p          = st.slider("Top-p", 0.1, 1.0, 0.9)
        max_new_tokens = st.slider("Max new tokens", 32, 512, 128)
        model_path     = st.text_input("Model path", "outputs/finetuned")

    run_inf = st.button("🧪 Generate Response")

    if run_inf:
        if Path(model_path).exists():
            with st.spinner("Generating..."):
                import subprocess
                result = subprocess.run(
                    ["python", "src/finetune.py",
                     "--generate", user_prompt,
                     "--output", model_path],
                    capture_output=True, text=True, timeout=120,
                )
            if result.returncode == 0:
                st.markdown("**Response:**")
                out_lines = result.stdout.strip().split("\n")
                response  = "\n".join(l for l in out_lines if l.startswith("Response:"))
                st.markdown(f'<div class="config-box">{response or result.stdout}</div>', unsafe_allow_html=True)
            else:
                st.error(f"Error: {result.stderr[:500]}")
        else:
            # Demo mode
            st.markdown("**Demo response (model not yet trained):**")
            demo = ("Gradient descent is an optimization algorithm used to minimize a function "
                    "by iteratively moving in the direction of steepest descent as defined by the "
                    "negative of the gradient. In machine learning, we use it to minimize the loss "
                    "function by updating model weights step by step.")
            st.markdown(f'<div class="config-box">{demo}</div>', unsafe_allow_html=True)
            st.caption("👆 This is a demo response. Train a model first to see real outputs.")

# ── Tab 5: Architecture ─────────────────────────────────────────────────────────
with tab5:
    st.markdown("### 📐 LoRA Architecture Explained")
    st.markdown("""
**How LoRA works:**

Instead of updating all model weights (billions of parameters), LoRA adds small trainable matrices:

```
Original:  W (frozen, e.g. 4096 × 4096 = 16.7M params)
LoRA adds: A (4096 × r) + B (r × 4096) where r = 16
           = 2 × 4096 × 16 = 131K params  ← only 0.78% of original!

Forward pass: output = W·x + (B·A)·x × (alpha/r)
                       ↑ frozen  ↑ trained
```

**QLoRA adds 4-bit quantization:**
```
W is stored in NF4 (Normal Float 4-bit) → 8x less memory
Gradients computed in BF16, then applied to LoRA adapters
Result: fine-tune a 7B model on a single 8GB GPU!
```
    """)

    st.markdown("### 🔢 Trainable Parameters by Config")
    rows = []
    for r in [4, 8, 16, 32, 64]:
        trainable = 2 * r * 4096 * 4  # 4 target modules, hidden=4096
        pct = trainable / (params_b * 1e9) * 100
        rows.append({"LoRA Rank": r, "Trainable Params": f"{trainable:,}", "% of Model": f"{pct:.3f}%"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("### ⚡ Training Tips")
    tips = [
        "🎯 Start with r=16, alpha=32 — these work for most tasks",
        "📉 If loss doesn't decrease: increase learning rate or reduce batch size",
        "📈 If loss oscillates: reduce learning rate or increase warmup ratio",
        "💾 QLoRA is 4-bit — always use it unless you have >40GB VRAM",
        "🔄 Gradient accumulation = fake larger batches without more memory",
        "⏱️ 1-3 epochs is usually enough for instruction tuning — more = overfitting",
        "📊 Watch validation loss — if it rises while train loss drops, you're overfitting",
        "🎲 Use temperature 0.1-0.3 for factual tasks, 0.7-1.0 for creative tasks",
    ]
    for tip in tips:
        st.markdown(f'<div class="insight-box" style="padding:10px 16px;margin:5px 0;">{tip}</div>', unsafe_allow_html=True)
