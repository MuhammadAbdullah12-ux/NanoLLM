# 🚀 NanoLLM: Building a Mini-GPT Transformer From Scratch (Days 1–21 Complete)

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

**NanoLLM** is a complete, educational implementation of an autoregressive Small Language Model (GPT-2 style Transformer architecture) built completely from scratch using raw PyTorch. 

Over 21 intensive days, this project progresses from fundamental PyTorch tensors to an $8.4\times$ scaled **211,777-parameter Mini-GPT** trained on Tiny Shakespeare, featuring **Cosine LR Scheduling**, **Gradient Clipping**, **AdamW Weight Decay**, **Inverted Dropout**, **Temperature / Top-K / Top-P (Nucleus) Sampling**, and an **Interactive Generation CLI Playground**.

---

## 🌟 Key Performance & Architecture Highlights

* **Architecture**: Scaled 4-Block Pre-LN Causal Transformer (`N_EMBD = 64`, `BLOCK_SIZE = 64`, `N_LAYER = 4`, `NUM_HEADS = 4`).
* **Trainable Parameters**: **211,777 parameters** ($8.42\times$ scaling expansion from baseline).
* **Validation Loss**: Driven down from **`4.3381` $\rightarrow$ `1.8942`**.
* **Validation Perplexity ($PPL = e^L$)**: Reduced from **`76.56` $\rightarrow$ `6.64`** (narrowing next-token confusion from 65 tokens to ~6 confident options).
* **Inference Sampling Pipeline**: Combined **Temperature Scaling** ($T=0.8$), **Top-K** ($K=20$), and **Top-P Nucleus Sampling** ($P=0.9$).
* **Interactive CLI**: Real-time prompt completion playground (`python generate.py`).

---

## 🛠️ Repository Layout

```text
NanoLLM/
├── generate.py                   # Interactive CLI Generation Playground
├── README.md                     # Capstone Documentation
├── checkpoints/
│   └── gpt_shakespeare.pt       # Saved Full Dictionary Checkpoint (Weights + Adam State + Config)
├── data/
│   ├── raw/                      # Tiny Shakespeare input.txt
│   └── prepared/                 # Saved Token Tensors (train.pt, val.pt)
└── src/
    ├── config.py                 # Centralized Hyperparameter Configuration (Days 1–21)
    ├── tokenizer.py              # Character-Level Tokenizer (encode / decode)
    ├── day9_data_loader.py       # Sequence Batching & Target Shifting (get_batch)
    ├── day10_embeddings.py       # Token & Positional Embeddings
    ├── day11_attention.py        # Causal Single Self-Attention Head (Q, K, V & tril mask)
    ├── day12_multihead_ffn.py    # Multi-Head Attention & 4x Feed-Forward Network
    ├── model.py                  # Full GPTLanguageModel Architecture & generate()
    ├── train.py                  # 2,000-Step Training Loop with LR Scheduler & Grad Clip
    ├── day15_lr_scheduler.py     # Cosine Annealing with Warmup Diagnostic
    ├── day16_grad_clip.py        # Gradient Clipping & AdamW Weight Decay Diagnostic
    ├── day17_param_count.py      # Architecture Scaling & Parameter Counter Diagnostic
    ├── day18_dropout.py          # Inverted Dropout Math & Train/Eval Mode Switching
    ├── day19_sampling.py        # Temperature Division & Top-K Masking Diagnostic
    ├── day20_nucleus_sampling.py # Top-P (Nucleus) Cumulative Probability Masking
    └── day21_eval_perplexity.py  # Final Loss, Perplexity & Throughput Evaluation Suite
```

---

## 📅 21-Day Complete Curriculum Roadmap

### 🧱 Week 1: PyTorch Foundations & Diagnostic Experiments (Days 1–7)
* **Day 1–3 (Training Mechanics)**: Built the 4 Sacred Loop Steps (`zero_grad()`, `forward()`, `backward()`, `step()`) and trained first linear classifier.
* **Day 4–5 (MNIST & Checkpointing)**: Built 2-layer MLP on MNIST dataset, implemented Train/Val loss monitoring and weight serialization.
* **Day 6–7 (Diagnostic Experiments)**: Simulated exploding learning rates (`lr = 5.0` $\rightarrow$ `NaN`), stochastic mini-batch noise (`batch_size = 1`), and overfitting.

### 🏗️ Week 2: Building the GPT Transformer Architecture (Days 8–14)
* **Day 8–9 (Tokenization & DataLoader)**: Tokenized 1.1M Shakespeare characters into $|V|=65$ tokens with `get_batch()` sequence target shifting ($y = x + 1$).
* **Day 10 (Embeddings)**: Combined Token Embeddings and Positional Embeddings ($x = E_{\text{tok}} + E_{\text{pos}}$).
* **Day 11 (Causal Self-Attention)**: Implemented Linear projections ($Q, K, V$), Scaled Dot-Product $\frac{QK^T}{\sqrt{d_k}}$, and Causal Masking (`tril`).
* **Day 12 (Multi-Head & FFN)**: Built 4 parallel attention heads and 2-layer FFN with $4\times$ expansion ($64 \rightarrow 256 \rightarrow 64$) using `GELU`.
* **Day 13–14 (Full GPT & Autoregressive Decoding)**: Assembled $N_{\text{layer}}=4$ Pre-LN Transformer Blocks (`x = x + sublayer(LN(x))`), LM Head projection, and `generate()` context cropping.

### 🚀 Week 3: Optimization, Regularization & Advanced Sampling (Days 15–21)
* **Day 15 (Learning Rate Scheduler)**: Implemented **Cosine Annealing with Warmup** (`WARMUP_ITERS=100`, `LR_DECAY_ITERS=2000`).
* **Day 16 (Training Stability)**: Added **Gradient Clipping** (`GRAD_CLIP=1.0`) and **AdamW Weight Decay** (`WEIGHT_DECAY=0.1`).
* **Day 17 (Model Scaling)**: Scaled architecture to **211,777 parameters** ($N_{\text{embd}}=64, Block_{\text{size}}=64, N_{\text{layer}}=4, N_{\text{head}}=4$).
* **Day 18 (Dropout Regularization)**: Integrated **Inverted Dropout** (`DROPOUT=0.1`) across Embedding, Attention, and Residual connections with `train()` vs `eval()` mode switching.
* **Day 19 (Temperature & Top-K)**: Integrated Temperature scaling ($T=0.8$) and Top-K filtering ($K=20$) into `generate()`, and upgraded to composite dictionary checkpointing.
* **Day 20 (Top-P Nucleus Sampling & CLI Playground)**: Implemented Top-P Nucleus Sampling ($P=0.9$) and built interactive prompt completion playground (`generate.py`).
* **Day 21 (Perplexity Benchmark & Grand Finale)**: Created `day21_eval_perplexity.py` computing Cross-Entropy Loss, Perplexity ($PPL = e^L$), generation throughput (tokens/sec), and multi-prompt benchmarks.

---

## 📊 Final Quantitative Benchmark Results

| Metric | Initial Un-Trained | Scaled Baseline (Day 17) | Final Trained GPT (Day 21) |
| :--- | :---: | :---: | :---: |
| **Train Loss** | `4.3138` | `2.4004` | **`1.8210`** |
| **Train Perplexity ($PPL$)** | `74.72` | `11.02` | **`6.17`** |
| **Validation Loss** | `4.3381` | `2.4270` | **`1.8942`** |
| **Validation Perplexity ($PPL$)** | `76.56` | `11.32` | **`6.64`** |
| **Trainable Parameters** | `25,121` | `211,777` | **`211,777`** |

---

## 💻 Quickstart & How to Run

### 1. Run Interactive LLM Playground:
```powershell
python generate.py
```
*Prompt your trained model interactively in terminal:*
```text
Enter prompt (or press Enter for default 'KING HENRY:'): ROMEO:
```

### 2. Train Model from Scratch:
```powershell
python src/train.py
```

### 3. Run Full Perplexity Benchmark Suite:
```powershell
python src/day21_eval_perplexity.py
```

---

## 📄 License
This project is open-source under the [MIT License](LICENSE).
