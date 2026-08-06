import os
import torch
import config
from tokenizer import decode

# Set random seed for reproducibility
torch.manual_seed(config.SEED)

# 1. Load tokenized datasets saved on Day 8
train_data = torch.load('./data/prepared/train.pt')
val_data   = torch.load('./data/prepared/val.pt')

# Hyperparameters for Data Loading
batch_size = 4  # Number of parallel sequences
block_size = 8  # Context window length

# 2. Function to generate mini-batches of inputs X and targets Y
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y

xb, yb = get_batch('train')

print("--- Task 9.2 & 9.3: Data Loader & Batch Generation ---")
print(f"Input Batch Shape  (X): {xb.shape} -> (batch_size={batch_size}, block_size={block_size})")
print(f"Target Batch Shape (Y): {yb.shape} -> (batch_size={batch_size}, block_size={block_size})\n")

# --- TASK 9.4: DECODING TENSORS TO READABLE SHAKESPEARE TEXT ---
print("--- Task 9.4: Decoded English Text & Step-by-Step Targets ---")

for b in range(batch_size):
    print(f"\n--- Batch Sequence #{b + 1} ---")
    print(f"Full Input  Text (X): '{decode(xb[b].tolist())}'")
    print(f"Full Target Text (Y): '{decode(yb[b].tolist())}'")
    print("\n  Step-by-Step Predictions inside this sequence:")
    
    for t in range(block_size):
        context = decode(xb[b][:t+1].tolist())
        target  = decode([yb[b][t].item()])
        print(f"  Step {t+1}: When input is {context:<10} -> Model must predict: '{target}'")
