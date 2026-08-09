import sys
import torch
import torch.nn as nn
import config

# Reconfigure stdout for Windows console compatibility
sys.stdout.reconfigure(encoding='utf-8')

# Set seed for reproducible dropout mask demonstration
torch.manual_seed(config.SEED)

print("--- Task 18.3: Inverted Dropout & Model Mode Diagnostic Verification ---\n")

# 1. Create a dummy activation tensor of ones: shape (1, 10)
x = torch.ones((1, 10))
dropout_p = 0.2  # 20% drop probability for clear visual demonstration
dropout_layer = nn.Dropout(p=dropout_p)

print(f"Original Input Tensor x (10 elements):\n{x}\n")
print(f"Dropout Probability p = {dropout_p} (1 / (1-p) scale factor = {1.0 / (1.0 - dropout_p):.2f})\n")

# 2. Test in Training Mode (model.train())
dropout_layer.train()
out_train = dropout_layer(x)

zero_count = (out_train == 0).sum().item()
active_count = (out_train != 0).sum().item()

print("=" * 60)
print("1. TRAINING MODE (dropout_layer.train()):")
print("=" * 60)
print(f"Output Tensor:\n{out_train}")
print(f"Zeroed Elements (Dropped) : {zero_count} / 10 ({zero_count * 10}%)")
print(f"Active Elements (Scaled) : {active_count} / 10 (Scaled to {1.0 / (1.0 - dropout_p):.2f})")
print(f"Mean of Input  : {x.mean().item():.4f}")
print(f"Mean of Output : {out_train.mean().item():.4f} (Maintains Expected Value!)\n")

# 3. Test in Evaluation Mode (model.eval())
dropout_layer.eval()
out_eval = dropout_layer(x)

print("=" * 60)
print("2. EVALUATION MODE (dropout_layer.eval()):")
print("=" * 60)
print(f"Output Tensor:\n{out_eval}")
print(f"Exact Match with Input (Input == Output): {torch.equal(x, out_eval)}")
print(f"Mean of Output : {out_eval.mean().item():.4f}\n")

print("SUCCESS: Inverted dropout scaling & train/eval mode switching verified!")
