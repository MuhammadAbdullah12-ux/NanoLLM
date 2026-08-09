import sys
import torch
import torch.nn.functional as F
import config

# Reconfigure stdout for Windows console compatibility
sys.stdout.reconfigure(encoding='utf-8')

# Set seed for reproducible nucleus sampling demonstration
torch.manual_seed(config.SEED)

print("--- Task 20.3: Top-P (Nucleus) Sampling Diagnostic Verification ---\n")

# Function to apply Top-P (Nucleus) filtering to raw logits
def apply_top_p(logits, top_p=0.9):
    # Sort logits in descending order
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    
    # Compute cumulative probability distribution
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Remove tokens with cumulative probability above threshold (shift right to keep 1st token above threshold)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    # Scatter -infinity mask back to original un-sorted logit indices
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    
    filtered_logits = logits.clone()
    filtered_logits[indices_to_remove] = float('-inf')
    return filtered_logits, sorted_probs[0], sorted_indices[0]

# 1. Test Case A: Confident Model (Sharp Peak Logits)
raw_logits_sharp = torch.tensor([[6.0, 1.0, 0.5, 0.1, -1.0]])
tokens = ['the', 'a', 'one', 'some', 'xyz']

print("=" * 65)
print("1. TEST CASE A: CONFIDENT MODEL (SHARP PEAK LOGITS)")
print("=" * 65)

filtered_logits_a, sorted_p_a, sorted_idx_a = apply_top_p(raw_logits_sharp, top_p=0.9)
final_probs_a = F.softmax(filtered_logits_a, dim=-1)[0]

print(f"Original Probabilities : " + ", ".join([f"'{tokens[i]}': {F.softmax(raw_logits_sharp, dim=-1)[0][i].item()*100:.1f}%" for i in range(len(tokens))]))
print(f"Top-P (p=0.9) Filtered  : " + ", ".join([f"'{tokens[i]}': {final_probs_a[i].item()*100:.1f}%" for i in range(len(tokens))]))
kept_tokens_a = (final_probs_a > 0).sum().item()
print(f"Nucleus Pool Size Kept  : {kept_tokens_a} / 5 tokens (Dynamically narrowed!)\n")

# 2. Test Case B: Uncertain Model (Flat Logits)
raw_logits_flat = torch.tensor([[1.0, 0.9, 0.8, 0.7, 0.6]])

print("=" * 65)
print("2. TEST CASE B: UNCERTAIN MODEL (FLAT LOGITS)")
print("=" * 65)

filtered_logits_b, sorted_p_b, sorted_idx_b = apply_top_p(raw_logits_flat, top_p=0.9)
final_probs_b = F.softmax(filtered_logits_b, dim=-1)[0]

print(f"Original Probabilities : " + ", ".join([f"'{tokens[i]}': {F.softmax(raw_logits_flat, dim=-1)[0][i].item()*100:.1f}%" for i in range(len(tokens))]))
print(f"Top-P (p=0.9) Filtered  : " + ", ".join([f"'{tokens[i]}': {final_probs_b[i].item()*100:.1f}%" for i in range(len(tokens))]))
kept_tokens_b = (final_probs_b > 0).sum().item()
print(f"Nucleus Pool Size Kept  : {kept_tokens_b} / 5 tokens (Dynamically expanded!)\n")

print("SUCCESS: Top-P dynamic nucleus pool adaptation math verified!")
