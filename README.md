# 🚀 NanoLLM: Train a Language Model From Scratch

**Goal:** Deeply understand neural networks and transformer architecture by building a Small Language Model (GPT) from scratch using raw PyTorch.

---

## 🛠️ Project Structure

```
NanoLLM/
├── data/
│   └── raw/                # MNIST & raw datasets
├── src/
│   ├── config.py           # Centralized hyperparameter configuration
│   ├── day3_training_loop.py
│   ├── day4_mnist_classifier.py
│   ├── day5_validation_and_checkpoint.py
│   ├── day6_experiments.py
│   ├── day6_exp1_exploding_lr.py
│   └── day6_exp2_noisy_batch1.py
├── experiments/
│   └── run_logs/           # Loss curves & diagnostic plots
├── checkpoints/            # Saved PyTorch model state_dicts (.pt)
└── README.md
```

---

## 📚 Week 1 Completed — Foundations & PyTorch Training Mechanics

### Key Learnings & Implementations:
1. **Raw PyTorch Training Mechanics:** Implemented the 4 Sacred Loop Steps (`zero_grad()`, `forward()`, `backward()`, `step()`).
2. **Real Dataset Pipelines:** Integrated `torchvision` MNIST dataset with `DataLoader` batching and `nn.CrossEntropyLoss`.
3. **Validation & Checkpointing:** Monitored Train vs Validation loss to prevent overfitting and implemented model serialization via `torch.save(model.state_dict())`.
4. **Diagnostic Experiments:**
   - Visualized loss curves using `matplotlib`.
   - Simulated **Gradient Explosion** (`lr = 5.0` resulting in `NaN` loss).
   - Simulated **Stochastic Noise** (`batch_size = 1` causing step loss oscillation).

---

## 📊 Week 1 Diagnostic Experiment Summary

| Experiment | Setup | Observed Outcome | Cause |
| :--- | :--- | :--- | :--- |
| **Baseline** | `lr=0.001`, `batch_size=64` | Smooth convergence, Val Acc: 97.19% | Healthy learning trajectory |
| **Exp #1: Exploding LR** | `lr=5.0`, `batch_size=64` | `Train Loss: NaN`, `Val Loss: NaN` | Gradient overshoot & float overflow |
| **Exp #2: Noisy Batch** | `lr=0.001`, `batch_size=1` | High oscillation per step | Maximum stochastic gradient variance |

---

## ⏭️ Next Step: Week 2 — Building GPT from Scratch!
In Week 2, we build a working character-level Transformer GPT model on tiny Shakespeare!
