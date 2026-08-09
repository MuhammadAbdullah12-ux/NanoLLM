import os
import sys
import torch
import torch.optim as optim
import config
from model import GPTLanguageModel
from bpe_tokenizer import encode_bpe, decode_bpe, VOCAB_SIZE_BPE
from day23_bpe_data_loader import get_batch_bpe
from day15_lr_scheduler import get_lr

# Reconfigure stdout for Windows console compatibility
sys.stdout.reconfigure(encoding='utf-8')

# Set seed for reproducibility
torch.manual_seed(config.SEED)

print("--- Task 24.3: Training Subword BPE GPT Model (|V|=50,257) ---\n")

# 1. Instantiate GPT Architecture adapted for BPE Vocabulary Size (50,257)
model = GPTLanguageModel(vocab_size=VOCAB_SIZE_BPE)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"BPE GPT Model Instantiated (|V| = {VOCAB_SIZE_BPE:,})")
print(f"Total Trainable Parameters : {total_params:,} parameters\n")

# 2. AdamW Optimizer with Decoupled Weight Decay
optimizer = optim.AdamW(
    model.parameters(),
    lr=config.LEARNING_RATE,
    weight_decay=config.WEIGHT_DECAY
)

# Evaluation Function to estimate Train vs Validation Loss
@torch.no_grad()
def estimate_bpe_loss(eval_iters=30):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch_bpe(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# 3. Training Loop over 1,000 Steps
max_iters = 1000
eval_interval = 200

print(f"Starting BPE Training Loop ({max_iters} steps)...\n")

for iter in range(1, max_iters + 1):
    # Dynamic Cosine LR Schedule Update
    lr = get_lr(iter)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Evaluate loss periodically
    if iter % eval_interval == 0 or iter == 1:
        losses = estimate_bpe_loss()
        print(f"Step [{iter:4d}/{max_iters}] | Current LR: {lr:.6f} | Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f}")

    # Fetch subword BPE mini-batch
    xb, yb = get_batch_bpe('train')

    # Forward pass & loss
    logits, loss = model(xb, yb)

    # 4 Sacred Loop Steps + Gradient Clipping
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.GRAD_CLIP)
    optimizer.step()

print("\nBPE Model Training complete!")

# 4. Save Comprehensive BPE Checkpoint
os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
checkpoint_path = os.path.join(config.CHECKPOINT_DIR, 'gpt_bpe_shakespeare.pt')
checkpoint = {
    'step': max_iters,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_loss': float(losses['train']),
    'val_loss': float(losses['val']),
    'vocab_size': VOCAB_SIZE_BPE,
    'config': {
        'n_embd': config.N_EMBD,
        'block_size': config.BLOCK_SIZE,
        'n_layer': config.N_LAYER,
        'num_heads': config.NUM_HEADS
    }
}
torch.save(checkpoint, checkpoint_path)
print(f"BPE Checkpoint saved to '{checkpoint_path}'")

# 5. Generate Text Prompt Completions using BPE Tokenizer!
print("\n" + "=" * 60)
print("🎭 BPE SUBWORD GENERATED SHAKESPEARE TEXT:")
print("=" * 60)

prompt = "KING HENRY:"
prompt_tokens = encode_bpe(prompt)
context = torch.tensor([prompt_tokens], dtype=torch.long)

model.eval()
with torch.no_grad():
    generated_tokens = model.generate(
        context, 
        max_new_tokens=100, 
        temperature=config.TEMPERATURE, 
        top_k=config.TOP_K, 
        top_p=config.TOP_P
    )[0].tolist()

generated_text = decode_bpe(generated_tokens)
print(f"--- Prompt: '{prompt}' ---")
print(generated_text)
print("=" * 60)
