import os
import sys
import time
import math
import torch
import config
from model import GPTLanguageModel
from tokenizer import encode, decode
from day9_data_loader import get_batch

# Reconfigure stdout for Windows console compatibility
sys.stdout.reconfigure(encoding='utf-8')

# Set seed for reproducible evaluation
torch.manual_seed(config.SEED)

print("=" * 70)
print("🎓 DAY 21: NANOLLM FINAL PERPLEXITY & THROUGHPUT EVALUATION BENCHMARK")
print("=" * 70 + "\n")

# 1. Instantiate Model & Load Trained Checkpoint
model = GPTLanguageModel()
checkpoint_path = os.path.join(config.CHECKPOINT_DIR, 'gpt_shakespeare.pt')

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        step_info = checkpoint.get('step', 'N/A')
        print(f"✅ Loaded trained checkpoint from '{checkpoint_path}' (Trained for {step_info} steps)")
    else:
        model.load_state_dict(checkpoint)
        print(f"✅ Loaded model weights from '{checkpoint_path}'")
else:
    print(f"⚠️ Warning: No trained checkpoint found at '{checkpoint_path}'. Evaluating un-trained model.")

model.eval()

# 2. Compute Total Trainable Parameter Count
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

# 3. Compute Loss & Perplexity (PPL = e^Loss) across Train and Validation Splits
@torch.no_grad()
def evaluate_metrics(eval_iters=50):
    metrics = {}
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        mean_loss = losses.mean().item()
        perplexity = math.exp(mean_loss)  # PPL = e^Loss
        metrics[split] = {'loss': mean_loss, 'ppl': perplexity}
    return metrics

print("\nComputing Cross-Entropy Loss & Perplexity metrics (50 iterations)...")
metrics = evaluate_metrics(eval_iters=50)

print("\n" + "=" * 70)
print("📊 NANOLLM QUANTITATIVE EVALUATION BENCHMARK METRICS:")
print("=" * 70)
print(f"  • Total Trainable Parameters : {total_params:,} parameters")
print(f"  • Embedding Dimension (N_EMBD): {config.N_EMBD}")
print(f"  • Context Window (BLOCK_SIZE): {config.BLOCK_SIZE} tokens")
print(f"  • Transformer Depth (N_LAYER): {config.N_LAYER} blocks")
print(f"  • Multi-Head Attention Heads : {config.NUM_HEADS} heads\n")
print(f"  • Train Cross-Entropy Loss   : {metrics['train']['loss']:.4f}")
print(f"  • Train Perplexity (PPL)     : {metrics['train']['ppl']:.4f}")
print(f"  • Validation Cross-Entropy Loss: {metrics['val']['loss']:.4f}")
print(f"  • Validation Perplexity (PPL)  : {metrics['val']['ppl']:.4f}")
print("=" * 70)

# 4. Measure Generation Throughput Speed (Tokens / Sec)
num_gen_tokens = 500
context = torch.zeros((1, 1), dtype=torch.long)

start_time = time.perf_counter()
with torch.no_grad():
    generated_indices = model.generate(
        context, 
        max_new_tokens=num_gen_tokens, 
        temperature=config.TEMPERATURE, 
        top_k=config.TOP_K, 
        top_p=config.TOP_P
    )[0].tolist()
end_time = time.perf_counter()

total_time = end_time - start_time
tokens_per_sec = num_gen_tokens / total_time

print(f"\n⚡ GENERATION THROUGHPUT BENCHMARK:")
print(f"  • Generated Tokens           : {num_gen_tokens} tokens")
print(f"  • Total Time Taken           : {total_time:.3f} seconds")
print(f"  • Throughput Speed           : {tokens_per_sec:.2f} tokens/second")

# 5. Multi-Character Prompt Completion Benchmarks
print("\n" + "=" * 70)
print("🎭 MULTI-CHARACTER PROMPT COMPLETION BENCHMARKS:")
print("=" * 70)

sample_prompts = ["KING HENRY:", "ROMEO:", "HAMLET:"]
for prompt in sample_prompts:
    prompt_tokens = encode(prompt)
    prompt_tensor = torch.tensor([prompt_tokens], dtype=torch.long)
    with torch.no_grad():
        comp_indices = model.generate(
            prompt_tensor, 
            max_new_tokens=150, 
            temperature=config.TEMPERATURE, 
            top_k=config.TOP_K, 
            top_p=config.TOP_P
        )[0].tolist()
    comp_text = decode(comp_indices)
    print(f"\n--- PROMPT: '{prompt}' ---")
    print(comp_text)
    print("-" * 50)

print("\n" + "=" * 70)
print("🏆 CONGRATULATIONS! NanoLLM Day 21 Benchmark Complete!")
print("=" * 70)
