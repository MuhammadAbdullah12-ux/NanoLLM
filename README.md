# 🚀 NanoLLM: Building & Fine-Tuning a GPT Transformer (Days 1–26 Complete)

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

**NanoLLM** is a complete 4-week, 26-day educational codebase for building, training, evaluating, and fine-tuning an autoregressive Small Language Model (GPT Transformer architecture) from scratch in raw PyTorch and Hugging Face.

---

## 🌟 Major Highlights & Accomplishments

* **Architectures Built**:
  - **Custom NanoLLM Mini-GPT**: 4 Layers, 4 Heads, 64 Embed Dim $\implies$ **211,777 parameters** (From-Scratch).
  - **Hugging Face GPT-2 Small**: 12 Layers, 12 Heads, 768 Embed Dim $\implies$ **124,439,808 parameters** (Pretrained & Fine-Tuned).
* **Tokenizers**:
  - Character-level Tokenizer ($|V|=65$).
  - OpenAI Subword Byte-Pair Encoding (`tiktoken` `gpt2` with $|V|=50,257$ & $3.3\times$ token compression!).
* **Training Techniques**:
  - Cosine Learning Rate Annealing with Warmup (`WARMUP_ITERS=100`, `LR_DECAY_ITERS=2000`).
  - Gradient Clipping (`GRAD_CLIP=1.0`) & Decoupled Weight Decay (`WEIGHT_DECAY=0.1`).
  - Inverted Dropout Regularization (`DROPOUT=0.1`) across Embedding, Attention, and Residual Layers.
* **Inference Sampling Pipeline**:
  - Combined **Temperature Scaling** ($T=0.8$), **Top-K Filtering** ($K=20$), and **Top-P Nucleus Sampling** ($P=0.9$).
* **Interactive CLI & Fine-Tuning**:
  - Real-time prompt CLI playground (`python generate.py`).
  - Domain Fine-Tuning of 124M GPT-2 on Shakespeare (`python src/day26_finetune_gpt2.py`).

---

## 🛠️ Repository Layout

```text
NanoLLM/
├── generate.py                   # Interactive CLI Generation Playground
├── README.md                     # Portfolio Documentation
├── checkpoints/
│   ├── gpt_shakespeare.pt       # Trained 211k Character GPT Checkpoint
│   ├── gpt_bpe_shakespeare.pt   # Trained 211k BPE Subword GPT Checkpoint
│   └── gpt2_finetuned_shakespeare.pt # Fine-Tuned 124M Pretrained GPT-2 Checkpoint
├── data/
│   ├── raw/                      # Tiny Shakespeare input.txt
│   └── prepared/                 # Saved Tensors (train.pt, val.pt, bpe_train.pt, bpe_val.pt)
└── src/
    ├── config.py                 # Central Hyperparameter Configuration (Days 1–26)
    ├── tokenizer.py              # Character-Level Tokenizer (encode / decode)
    ├── bpe_tokenizer.py          # Subword BPE Tokenizer Wrapper (tiktoken)
    ├── day9_data_loader.py       # Character Sequence Data Loader
    ├── day23_bpe_data_loader.py   # Subword BPE Data Loader
    ├── day10_embeddings.py       # Token & Positional Embeddings
    ├── day11_attention.py        # Causal Self-Attention Head (Q, K, V & tril mask)
    ├── day12_multihead_ffn.py    # Multi-Head Attention & 4x Feed-Forward Network
    ├── model.py                  # Full GPTLanguageModel Architecture & generate()
    ├── train.py                  # 2,000-Step Character Model Training Loop
    ├── day22_bpe_tokenizer.py    # Diagnostic BPE Tokenizer & Compression Test
    ├── day23_prepare_bpe_dataset.py # BPE Subword Pre-processing Script
    ├── day24_train_bpe_model.py  # 1,000-Step Subword BPE Model Training Script
    ├── day25_huggingface_gpt2.py # Pretrained GPT-2 (124M) Loading & Demonstration
    └── day26_finetune_gpt2.py    # 100-Step Pretrained GPT-2 Fine-Tuning Script
```

---

## 📅 4-Week Complete Curriculum Roadmap

### 🧱 Week 1: PyTorch Foundations & Diagnostic Experiments (Days 1–7)
* PyTorch Tensors, Autograd, 4 Sacred Loop Steps, MNIST Classifier, Train/Val Loss Monitoring, Overfitting & Exploding LR Experiments.

### 🏗️ Week 2: Building the GPT Transformer Architecture (Days 8–14)
* Tokenization, Sequence Target Shifting ($y=x+1$), Token & Positional Embeddings, Causal Self-Attention, Multi-Head Attention, FFN, Pre-LN Residual Blocks, and Autoregressive Generation.

### 🚀 Week 3: Optimization, Regularization & Advanced Sampling (Days 15–21)
* Cosine LR Scheduler with Warmup, Gradient Clipping, AdamW Weight Decay, Architecture Scaling ($211k$ params), Inverted Dropout, Temperature Scaling, Top-K, Top-P Nucleus Sampling, Checkpointing, and Perplexity Benchmarks.

### 🎯 Week 4: Subword BPE & Transfer Learning Fine-Tuning (Days 22–26)
* **Day 22–23**: Subword Byte-Pair Encoding (`tiktoken` `gpt2`) with $3.3\times$ compression factor & BPE Data Loader.
* **Day 24**: Subword BPE GPT Model Training ($|V_{BPE}|=50,257$).
* **Day 25**: Loading Pretrained **124M GPT-2** from Hugging Face & Capacity Scaling Analysis ($587\times$ capacity).
* **Day 26**: Domain Fine-Tuning 124M GPT-2 on Shakespeare dialogue for 100 steps ($\eta = 5 \times 10^{-5}$) & Capstone Conclusion.

---

## 📊 Final Quantitative Benchmark Results

| Model / Approach | Parameters | Vocab Size | Validation Loss | Validation Perplexity ($PPL = e^L$) |
| :--- | :---: | :---: | :---: | :---: |
| **Un-Trained Random Initialized** | `211,777` | `65` | `4.3381` | `76.56` |
| **NanoLLM Mini-GPT (From Scratch)** | `211,777` | `65` | `1.8942` | `6.64` |
| **Pretrained GPT-2 Small (Hugging Face)** | `124,439,808` | `50,257` | `3.4210` | `30.60` |
| **Fine-Tuned GPT-2 Small (100 Steps)** | **`124,439,808`** | **`50,257`** | **`2.1580`** | **`8.65`** |

---

## 💻 Quickstart Commands

### 1. Interactive Prompt Playground:
```powershell
python generate.py
```

### 2. Fine-Tune Pretrained 124M GPT-2 on Shakespeare:
```powershell
python src/day26_finetune_gpt2.py
```

---

## 📄 License
This project is open-source under the [MIT License](LICENSE).
