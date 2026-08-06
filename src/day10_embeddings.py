import torch
import torch.nn as nn
import config
from day9_data_loader import get_batch

# Set random seed for reproducibility
torch.manual_seed(config.SEED)

# 1. Fetch a sample mini-batch from Day 9 data loader
# xb shape: (B=4, T=8) -> 4 parallel sequences of 8 integer token IDs
xb, yb = get_batch('train')

# 2. Define Token & Positional Embedding Layers
token_embedding_table = nn.Embedding(config.VOCAB_SIZE, config.N_EMBD)
position_embedding_table = nn.Embedding(config.BLOCK_SIZE, config.N_EMBD)

# 3. Step A: Token Embedding Lookup
# Transforms (B=4, T=8) integer IDs into (B=4, T=8, C=32) semantic feature vectors
tok_emb = token_embedding_table(xb)

# 4. Step B: Positional Embedding Lookup
# Create position indices [0, 1, 2, 3, 4, 5, 6, 7] of shape (T=8)
pos_indices = torch.arange(config.BLOCK_SIZE)
pos_emb = position_embedding_table(pos_indices)  # Shape: (T=8, C=32)

# 5. Step C: Combined Input Representation (Token + Position)
# PyTorch automatically broadcasts pos_emb (8, 32) across batch dimension B=4 to yield (4, 8, 32)
x_combined = tok_emb + pos_emb

# --- TASK 10.4: TENSOR SHAPE VERIFICATION ---
print("--- Task 10.3 & 10.4: Token + Positional Embedding Inspection ---")
print(f"Raw Input Token Batch Shape (xb) : {xb.shape} -> (Batch B={config.BATCH_SIZE}, Time T={config.BLOCK_SIZE})")
print(f"Token Embedding Tensor Shape     : {tok_emb.shape} -> (B=4, T=8, Channels C=32)")
print(f"Positional Embedding Tensor Shape: {pos_emb.shape} -> (T=8, Channels C=32)")
print(f"Combined Output Tensor Shape (x) : {x_combined.shape} -> (B=4, T=8, Channels C=32)\n")

print("SUCCESS: Input transformed from 2D integers [4, 8] to 3D continuous vectors [4, 8, 32]!")
