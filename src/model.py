import torch
import torch.nn as nn
import torch.nn.functional as F
import config
from day9_data_loader import get_batch

# Set random seed for reproducibility
torch.manual_seed(config.SEED)

# 1. Single Causal Self-Attention Head
class SingleHeadAttention(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(config.N_EMBD, head_size, bias=False)
        self.query = nn.Linear(config.N_EMBD, head_size, bias=False)
        self.value = nn.Linear(config.N_EMBD, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(config.BLOCK_SIZE, config.BLOCK_SIZE)))

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        head_size = q.shape[-1]
        wei = q @ k.transpose(-2, -1) * (head_size ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        out = wei @ v
        return out

# 2. Multi-Head Attention (4 Parallel Heads)
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([SingleHeadAttention(head_size) for _ in range(num_heads)])
        self.proj  = nn.Linear(num_heads * head_size, config.N_EMBD)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)

# 3. Feed-Forward Network
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd)
        )

    def forward(self, x):
        return self.net(x)

# 4. The Transformer Block (Attention + FFN + Pre-LN Residuals)
class Block(nn.Module):
    def __init__(self, n_embd, num_heads):
        super().__init__()
        head_size = n_embd // num_heads
        self.sa   = MultiHeadAttention(num_heads, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1  = nn.LayerNorm(n_embd)
        self.ln2  = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Pre-LayerNorm Residual Connections
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# 5. Complete GPT Language Model Architecture
class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Token & Positional Embedding tables
        self.token_embedding_table    = nn.Embedding(config.VOCAB_SIZE, config.N_EMBD)
        self.position_embedding_table = nn.Embedding(config.BLOCK_SIZE, config.N_EMBD)
        
        # Stack of N_LAYER (3) Transformer Blocks
        self.blocks = nn.Sequential(*[Block(config.N_EMBD, config.NUM_HEADS) for _ in range(config.N_LAYER)])
        
        # Final LayerNorm & Linear LM Head Projection
        self.ln_f    = nn.LayerNorm(config.N_EMBD)
        self.lm_head = nn.Linear(config.N_EMBD, config.VOCAB_SIZE)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # Step 1: Embeddings (Token + Position)
        tok_emb = self.token_embedding_table(idx)  # (B, T, C=32)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T, C=32)
        x = tok_emb + pos_emb  # (B, T, C=32)
        
        # Step 2: Pass through 3 Stacked Transformer Blocks
        x = self.blocks(x)  # (B, T, C=32)
        
        # Step 3: Final LayerNorm
        x = self.ln_f(x)    # (B, T, C=32)
        
        # Step 4: Final LM Head Projection to Vocab Size Logits
        logits = self.lm_head(x)  # (B, T, VOCAB_SIZE=65)

        # Calculate Loss if targets are provided
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

# --- TASK 13.4: VERIFICATION & INITIAL LOSS ---
xb, yb = get_batch('train')
model = GPTLanguageModel()
logits, loss = model(xb, yb)

print("--- Task 13.3 & 13.4: Complete GPT Architecture Verification ---")
print(f"Input Token Batch Shape (xb)   : {xb.shape} -> (B={config.BATCH_SIZE}, T={config.BLOCK_SIZE})")
print(f"Output Logits Tensor Shape     : {logits.shape} -> (B*T={xb.shape[0]*xb.shape[1]}, Vocab_Size={config.VOCAB_SIZE})")
print(f"Initial Un-trained CrossEntropy Loss: {loss.item():.4f}\n")

expected_initial_loss = -torch.log(torch.tensor(1.0 / config.VOCAB_SIZE)).item()
print(f"Theoretical Random Loss (-ln(1/65)): {expected_initial_loss:.4f}")
print("SUCCESS: Full GPT model architecture assembled and verified successfully!")
