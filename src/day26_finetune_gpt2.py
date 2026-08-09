import os
import sys
import torch
import torch.optim as optim
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import config
from day23_bpe_data_loader import get_batch_bpe

# Reconfigure stdout for Windows console compatibility
sys.stdout.reconfigure(encoding='utf-8')

# Set seed for reproducibility
torch.manual_seed(config.SEED)

print("--- Task 26.2: Fine-Tuning Pretrained GPT-2 (124M) on Shakespeare ---\n")

# 1. Load Pretrained GPT-2 Tokenizer and Model
print("Loading pretrained 'gpt2' model (124M parameters) from Hugging Face...")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Evaluation function for pre-fine-tuning loss
@torch.no_grad()
def estimate_loss(eval_iters=20):
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_batch_bpe('val')
        # HuggingFace GPT2LMHeadModel returns loss directly when labels=Y is passed
        outputs = model(X, labels=Y)
        losses[k] = outputs.loss.item()
    model.train()
    return losses.mean().item()

initial_val_loss = estimate_loss()
print(f"Pre-Fine-Tuning Validation Loss : {initial_val_loss:.4f} (Perplexity: {torch.exp(torch.tensor(initial_val_loss)).item():.2f})\n")

# 2. Configure Fine-Tuning Optimizer (Small Learning Rate = 5e-5)
fine_tune_lr = 5e-5
optimizer = optim.AdamW(model.parameters(), lr=fine_tune_lr, weight_decay=0.01)

# 3. Fine-Tuning Loop over 100 Steps
max_steps = 100
print(f"Starting Fine-Tuning Loop on Shakespeare dataset ({max_steps} steps, LR = {fine_tune_lr})...\n")

model.train()
for step in range(1, max_steps + 1):
    xb, yb = get_batch_bpe('train')
    
    # Forward pass with labels computing CrossEntropyLoss
    outputs = model(xb, labels=yb)
    loss = outputs.loss
    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    if step % 20 == 0 or step == 1:
        print(f"Fine-Tuning Step [{step:3d}/{max_steps}] | Train Loss: {loss.item():.4f}")

print("\nFine-Tuning complete!")

final_val_loss = estimate_loss()
print(f"\n" + "=" * 65)
print("📊 FINE-TUNING LOSS & PERPLEXITY IMPROVEMENT:")
print("=" * 65)
print(f"  • Pre-Fine-Tuning Validation Loss  : {initial_val_loss:.4f} (PPL: {torch.exp(torch.tensor(initial_val_loss)).item():.2f})")
print(f"  • Post-Fine-Tuning Validation Loss : {final_val_loss:.4f} (PPL: {torch.exp(torch.tensor(final_val_loss)).item():.2f})")
print("=" * 65)

# 4. Save Fine-Tuned Checkpoint
os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
checkpoint_path = os.path.join(config.CHECKPOINT_DIR, 'gpt2_finetuned_shakespeare.pt')
torch.save({'model_state_dict': model.state_dict()}, checkpoint_path)
print(f"\nSaved Fine-Tuned Model Checkpoint to '{checkpoint_path}'")

# 5. Generate Shakespeare Text Completion After Fine-Tuning
print("\n" + "=" * 65)
print("🎭 FINE-TUNED GPT-2 SHAKESPEARE COMPLETION DEMONSTRATION:")
print("=" * 65)

prompt = "KING HENRY:\nShall I be bold to tell you"
inputs = tokenizer(prompt, return_tensors='pt')

model.eval()
with torch.no_grad():
    output_ids = model.generate(
        inputs['input_ids'],
        max_new_tokens=100,
        temperature=config.TEMPERATURE,
        top_k=config.TOP_K,
        top_p=config.TOP_P,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(f"--- Prompt: '{prompt}' ---")
print(generated_text)
print("=" * 65)
