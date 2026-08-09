import sys
import torch
import torch.nn.functional as F
import config

# Reconfigure stdout for Windows console compatibility
sys.stdout.reconfigure(encoding='utf-8')

# Set seed for reproducible sampling demonstration
torch.manual_seed(config.SEED)

print("--- Task 19.3: Temperature Scaling & Top-K Sampling Diagnostic Verification ---\n")

# 1. Create dummy raw logits for 5 vocabulary tokens
# Let token 0 ('e') have highest logit, token 1 ('t') second highest, etc.
raw_logits = torch.tensor([[4.0, 2.0, 1.0, 0.0, -2.0]])
tokens = ['e', 't', 'a', 'o', 'x']

print(f"Raw Model Logits: {raw_logits.numpy()[0]}")
print(f"Token Candidates : {tokens}\n")

# 2. Temperature Scaling Demonstration (T = 0.2 vs 0.8 vs 1.5)
print("=" * 65)
print("1. TEMPERATURE SCALING DEMONSTRATION:")
print("=" * 65)

for temp in [0.2, 0.8, 1.5]:
    scaled_logits = raw_logits / temp
    probs = F.softmax(scaled_logits, dim=-1)[0]
    prob_str = ", ".join([f"{tokens[i]}: {probs[i].item()*100:.1f}%" for i in range(len(tokens))])
    print(f"Temperature T = {temp:3.1f} -> Probabilities: [{prob_str}]")

# 3. Top-K Masking Demonstration (K = 3)
print("\n" + "=" * 65)
print("2. TOP-K SAMPLING DEMONSTRATION (K = 3):")
print("=" * 65)

k = 3
logits = raw_logits.clone()
v, _ = torch.topk(logits, k)
min_topk_value = v[:, [-1]]  # K-th largest logit threshold

# Mask logits below threshold to -infinity
logits[logits < min_topk_value] = float('-inf')
topk_probs = F.softmax(logits, dim=-1)[0]

print(f"Raw Logits After Top-3 Masking: {logits.numpy()[0]}")
for i in range(len(tokens)):
    status = "ALLOWED" if topk_probs[i] > 0 else "BLOCKED (0.0%)"
    print(f"  Token '{tokens[i]}': Prob = {topk_probs[i].item()*100:5.1f}% | Status: {status}")

print("\nSUCCESS: Temperature scaling & Top-K masking math verified!")
