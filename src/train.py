import os
import torch
import torch.optim as optim
import config
from day9_data_loader import get_batch
from model import GPTLanguageModel
from tokenizer import decode
from day15_lr_scheduler import get_lr

# Set seed for reproducibility
torch.manual_seed(config.SEED)

# 1. Instantiate Model & AdamW Optimizer
model = GPTLanguageModel()
optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

# Evaluation Function to estimate average Train vs Validation loss
@torch.no_grad()
def estimate_loss(eval_iters=50):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# 2. The Training Loop with Dynamic Cosine LR Scheduling (Task 15.4)
max_iters = 2000
eval_interval = 200

print(f"--- Task 15.4: Training GPT with Warmup + Cosine Decay ({max_iters} Steps) ---\n")

for iter in range(1, max_iters + 1):
    # --- TASK 15.4 ADDITION: DYNAMIC LEARNING RATE UPDATE ---
    lr = get_lr(iter)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Evaluate loss on train and val sets periodically
    if iter % eval_interval == 0 or iter == 1:
        losses = estimate_loss()
        print(f"Step [{iter:4d}/{max_iters}] | Current LR: {lr:.6f} | Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f}")

    # 1. Fetch mini-batch
    xb, yb = get_batch('train')

    # 2. Forward pass & compute loss
    logits, loss = model(xb, yb)

    # 3. 4 Sacred Loop Steps
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print("\nTraining complete!")

# 3. Save Model Checkpoint
os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
checkpoint_path = os.path.join(config.CHECKPOINT_DIR, 'gpt_shakespeare.pt')
torch.save(model.state_dict(), checkpoint_path)
print(f"Model saved to '{checkpoint_path}'")

# 4. Generate Shakespeare Text from Trained Model!
print("\n" + "=" * 50)
print("🎭 GENERATED SHAKESPEARE TEXT WITH COSINE DECAY TRAINED MODEL:")
print("=" * 50)

context = torch.zeros((1, 1), dtype=torch.long)  # Start prompt: '\n'
generated_indices = model.generate(context, max_new_tokens=400)[0].tolist()
generated_text = decode(generated_indices)

print(generated_text)
print("=" * 50)
