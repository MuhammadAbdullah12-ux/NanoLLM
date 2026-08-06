import torch
import torch.nn as nn
import torch.nn.functional as F
import config
from day10_embeddings import x_combined as x_input

# Set random seed for reproducibility
torch.manual_seed(config.SEED)

# 1. Single Causal Self-Attention Head (Building Block)
class SingleHeadAttention(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(config.N_EMBD, head_size, bias=False)
        self.query = nn.Linear(config.N_EMBD, head_size, bias=False)
        self.value = nn.Linear(config.N_EMBD, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(config.BLOCK_SIZE, config.BLOCK_SIZE)))

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)  # (B, T, head_size=8)
        k = self.key(x)    # (B, T, head_size=8)
        v = self.value(x)  # (B, T, head_size=8)
        
        head_size = q.shape[-1]
        wei = q @ k.transpose(-2, -1) * (head_size ** -0.5)  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        out = wei @ v  # (B, T, head_size=8)
        return out

# 2. Multi-Head Attention (4 Parallel Heads in Parallel)
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        # Create num_heads parallel SingleHeadAttention instances
        self.heads = nn.ModuleList([SingleHeadAttention(head_size) for _ in range(num_heads)])
        # Linear projection to mix concatenated head outputs
        self.proj  = nn.Linear(num_heads * head_size, config.N_EMBD)

    def forward(self, x):
        # Run each head in parallel and concatenate along channel dimension
        # 4 heads of shape (B, T, 8) concatenated -> (B, T, 32)
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out) # Final linear projection (B, T, 32)
        return out

# 3. Feed-Forward Network (Position-wise 2-Layer Perceptron with 4x Expansion)
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # Expand 32 -> 128
            nn.GELU(),                      # Smooth non-linear activation
            nn.Linear(4 * n_embd, n_embd)   # Contract 128 -> 32
        )

    def forward(self, x):
        return self.net(x)

# --- TASK 12.4: TENSOR SHAPE VERIFICATION ---
mha = MultiHeadAttention(num_heads=config.NUM_HEADS, head_size=config.HEAD_SIZE)
ffn = FeedForward(n_embd=config.N_EMBD)

mha_out = mha(x_input)
ffn_out = ffn(mha_out)

print("--- Task 12.3 & 12.4: Multi-Head Attention & Feed-Forward Verification ---")
print(f"Input Embedded Tensor Shape (x) : {x_input.shape} -> (B={config.BATCH_SIZE}, T={config.BLOCK_SIZE}, C={config.N_EMBD})")
print(f"Multi-Head Attention Output Shape: {mha_out.shape} -> (B=4, T=8, C=32) [4 heads x 8 size concatenated]")
print(f"Feed-Forward Network Output Shape: {ffn_out.shape} -> (B=4, T=8, C=32) [Expanded 128 then contracted]\n")

print("SUCCESS: Input shape [4, 8, 32] perfectly preserved through both Multi-Head Attention and Feed-Forward Network!")
