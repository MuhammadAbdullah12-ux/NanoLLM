# 🚀 NanoLLM: Building a GPT Transformer From Scratch

**Goal:** Deeply understand neural networks and transformer architecture by building a Small Language Model (GPT) from scratch using raw PyTorch, training it on Tiny Shakespeare, and generating autoregressive text.

---

## 🛠️ Repository Architecture

```
NanoLLM/
├── data/
│   ├── raw/                # Tiny Shakespeare input.txt
│   └── prepared/           # Tokenized train.pt and val.pt
├── src/
│   ├── config.py           # Centralized GPT hyperparameter configuration
│   ├── tokenizer.py        # Character-level Tokenizer (stoi/itos, encode/decode)
│   ├── day9_data_loader.py # Sequence batching & target shifting (get_batch)
│   ├── day10_embeddings.py # Token & Positional Embeddings ([4, 8] -> [4, 8, 32])
│   ├── day11_attention.py  # Causal Single Self-Attention Head (Q, K, V & tril mask)
│   ├── day12_multihead_ffn.py # Multi-Head Attention & Feed-Forward Network
│   ├── model.py            # Complete GPTLanguageModel & generate() method
│   └── train.py            # 2,000-step Training Loop & Checkpoint Saving
├── experiments/
│   └── run_logs/           # Loss curves & diagnostic plots
├── checkpoints/            # Saved PyTorch model state_dicts (gpt_shakespeare.pt)
└── README.md
```

---

## 📚 Week 1 Completed — Foundations & PyTorch Mechanics

* **Raw PyTorch Training Mechanics:** Implemented the 4 Sacred Loop Steps (`zero_grad()`, `forward()`, `backward()`, `step()`).
* **Real Dataset Pipelines:** Integrated `torchvision` MNIST dataset with `DataLoader` batching and `nn.CrossEntropyLoss`.
* **Validation & Checkpointing:** Monitored Train vs Validation loss to prevent overfitting and serialized model weights (`state_dict`).
* **Diagnostic Experiments:** Simulated Gradient Explosion (`lr = 5.0` $\rightarrow$ `NaN`) and Stochastic Gradient Noise (`batch_size = 1`).

---

## 🎭 Week 2 Completed — Building & Training the GPT Transformer

### Transformer Architecture Details:
1. **Character-Level Tokenizer (`tokenizer.py`):** Encoded 1,115,394 Shakespeare characters into $|V|=65$ unique tokens with 90/10 Train/Val split.
2. **Sequence Batching (`day9_data_loader.py`):** Implemented `get_batch()` producing mini-batches `[B=4, T=8]` with target sequence shifting ($y$ shifted right by +1).
3. **Token & Positional Embeddings (`day10_embeddings.py`):** Combined token identity vectors and positional location vectors ($x = E_{\text{tok}} + E_{\text{pos}}$), mapping `[4, 8]` $\rightarrow$ `[4, 8, 32]`.
4. **Causal Self-Attention (`day11_attention.py`):** Constructed linear projections ($Q, K, V$), Scaled Dot-Product $\frac{QK^T}{\sqrt{d_k}}$, and Lower Triangular Masking (`tril`) to prevent future cheating.
5. **Multi-Head Attention & Feed-Forward (`day12_multihead_ffn.py`):** Built 4 parallel attention heads ($4 \times 8 = 32$) and 2-layer FFN with $4\times$ feature expansion ($32 \rightarrow 128 \rightarrow 32$) using `GELU`.
6. **Complete GPT Architecture (`model.py`):** Stacked $N_{\text{layer}}=3$ Transformer Blocks with Pre-LN Residual Connections (`x = x + sublayer(LN(x))`) and linear LM Head projection (`nn.Linear(32, 65)`).
7. **Autoregressive Text Generation (`generate()`):** Implemented sliding context window cropping (`idx[:, -8:]`) and probability sampling (`torch.multinomial`).

---

## 📊 Training Results (2,000 Steps on Tiny Shakespeare)

| Training Metric | Initial Un-Trained | Step 200 | Step 1200 | Final Step 2000 |
| :--- | :---: | :---: | :---: | :---: |
| **Train Loss** | `4.3138` | `2.6897` | `2.4294` | **`2.4004`** |
| **Validation Loss** | `4.3381` | `2.6897` | `2.4719` | **`2.4270`** |

---

## 🎭 Sample Generated Shakespeare Output

```text
==================================================
GENERATED SHAKESPEARE TEXT FROM TRAINED GPT MODEL:
==================================================

I push ar lo!
onin:Cave yor ar dangk, yoat whell, songank thing to non
Os shepowem illll e rented fairrded ntal? horcrs:
KO, ilenat ne arth thaus I shand:
Ton t je twith aar why frang thil tth Cous yow ramtili;
To mocen to thus, she oagh th:
==================================================
```

---

## ⏭️ Next Step: Week 3 — Making It Learn Well
In Week 3, we optimize our model with **LR Warmup + Cosine Decay**, **Gradient Clipping**, **AdamW Weight Decay**, and compare **Sampling Strategies** (Greedy vs Temperature vs Top-k vs Top-p)!
