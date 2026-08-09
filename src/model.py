import torch
import torch.nn as nn
import torch.nn.functional as F
import config
from day9_data_loader import get_batch
from tokenizer import decode

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
        self.dropout = nn.Dropout(config.DROPOUT)  # Task 18.4: Attention Dropout

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        head_size = q.shape[-1]
        wei = q @ k.transpose(-2, -1) * (head_size ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)  # Task 18.4: Apply dropout to attention weights
        out = wei @ v
        return out

# 2. Multi-Head Attention (4 Parallel Heads)
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([SingleHeadAttention(head_size) for _ in range(num_heads)])
        self.proj  = nn.Linear(num_heads * head_size, config.N_EMBD)
        self.dropout = nn.Dropout(config.DROPOUT)  # Task 18.4: Residual Projection Dropout

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))  # Task 18.4: Apply dropout to projection
        return out

# 3. Feed-Forward Network
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(config.DROPOUT)  # Task 18.4: Residual Projection Dropout
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
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# 5. Complete GPT Language Model Architecture
class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table    = nn.Embedding(config.VOCAB_SIZE, config.N_EMBD)
        self.position_embedding_table = nn.Embedding(config.BLOCK_SIZE, config.N_EMBD)
        self.embd_dropout = nn.Dropout(config.DROPOUT)  # Task 18.4: Embedding Dropout
        self.blocks  = nn.Sequential(*[Block(config.N_EMBD, config.NUM_HEADS) for _ in range(config.N_LAYER)])
        self.ln_f    = nn.LayerNorm(config.N_EMBD)
        self.lm_head = nn.Linear(config.N_EMBD, config.VOCAB_SIZE)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.embd_dropout(x)  # Task 18.4: Apply dropout to embeddings
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    # --- TASK 20.4 UPDATE: AUTOREGRESSIVE GENERATION WITH TEMP, TOP-K & TOP-P ---
    def generate(self, idx, max_new_tokens, temperature=config.TEMPERATURE, top_k=config.TOP_K, top_p=config.TOP_P):
        # idx is (B, T) array of indices in current context
        for _ in range(max_new_tokens):
            # Crop idx to the last block_size tokens so position embeddings stay in bounds
            idx_cond = idx[:, -config.BLOCK_SIZE:]
            
            # Forward pass to get logits for prediction
            logits, _ = self(idx_cond)
            
            # Focus only on the last time step logits: (B, T, C) -> (B, C)
            logits = logits[:, -1, :]
            
            # 1. Apply Temperature scaling
            if temperature > 0:
                logits = logits / temperature
                
            # 2. Apply Top-K filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # 3. Apply Top-P (Nucleus) filtering
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # Convert raw logits to probability distribution
            probs = F.softmax(logits, dim=-1)
            
            # Sample 1 new token from probability distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            
            # Append sampled index to running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
            
        return idx

# --- TASK 14.2 VERIFICATION ---
model = GPTLanguageModel()
context = torch.zeros((1, 1), dtype=torch.long)  # Start prompt: Token ID 0 ('\n')
generated_tokens = model.generate(context, max_new_tokens=100)[0].tolist()
un_trained_text = decode(generated_tokens)

print("--- Task 14.2: Autoregressive Text Generation Method Added ---")
print("Sample Text Generated by Un-trained Model (100 characters):")
print("-" * 50)
print(un_trained_text)
print("-" * 50)
print("\nSUCCESS: generate() method executed successfully!")
