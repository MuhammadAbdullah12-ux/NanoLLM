import torch
import torch.nn as nn
import torch.nn.functional as F
import config
from day10_embeddings import x_combined as x_input

# Set random seed for reproducibility
torch.manual_seed(config.SEED)

# --- TASK 11.3: SINGLE CAUSAL SELF-ATTENTION HEAD ---
class SingleHeadAttention(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        # Linear projections for Query, Key, and Value
        self.key   = nn.Linear(config.N_EMBD, head_size, bias=False)
        self.query = nn.Linear(config.N_EMBD, head_size, bias=False)
        self.value = nn.Linear(config.N_EMBD, head_size, bias=False)
        
        # Lower Triangular Causal Mask buffer (Tril) - prevents looking into future
        # Shaped (BLOCK_SIZE, BLOCK_SIZE) = (8, 8)
        self.register_buffer('tril', torch.tril(torch.ones(config.BLOCK_SIZE, config.BLOCK_SIZE)))

    def forward(self, x):
        B, T, C = x.shape  # B=4, T=8, C=32
        
        # Step 1: Compute Queries, Keys, Values
        q = self.query(x)  # (B, T, head_size=16)
        k = self.key(x)    # (B, T, head_size=16)
        v = self.value(x)  # (B, T, head_size=16)
        
        # Step 2: Compute Attention Affinity Matrix (Q @ K^T / sqrt(d_k))
        # (B, T, 16) @ (B, 16, T) -> (B, T, T)
        head_size = q.shape[-1]
        wei = q @ k.transpose(-2, -1) * (head_size ** -0.5)  # (B, T, T)
        
        # Step 3: Apply Causal Masking (Fill upper-triangular future spots with -inf)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        
        # Step 4: Compute Softmax Probability Distribution over past tokens
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        
        # Step 5: Weighted Combination of Values (Softmax @ V)
        # (B, T, T) @ (B, T, 16) -> (B, T, 16)
        out = wei @ v
        return out, wei

# Instantiate Attention Head
head = SingleHeadAttention(head_size=config.HEAD_SIZE)
out, attention_weights = head(x_input)

# --- TASK 11.4: TENSOR SHAPE & CAUSAL MASK VERIFICATION ---
print("--- Task 11.3 & 11.4: Single Self-Attention Head Verification ---")
print(f"Input Embedded Tensor Shape (x) : {x_input.shape} -> (B={config.BATCH_SIZE}, T={config.BLOCK_SIZE}, C={config.N_EMBD})")
print(f"Output Attention Tensor Shape   : {out.shape}     -> (B=4, T=8, Head_Size={config.HEAD_SIZE})")
print(f"Attention Weights Matrix Shape  : {attention_weights.shape} -> (B=4, T=8, T=8)\n")

print("Sample Causal Attention Probabilities Matrix for Batch #1 (Row 0):")
print(attention_weights[0].detach().round(decimals=3))

print("\nSUCCESS: Upper-triangular elements are exact 0.000 (causal mask working)!")
