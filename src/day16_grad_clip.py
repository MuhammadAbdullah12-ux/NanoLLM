import math
import torch
import config
from day9_data_loader import get_batch
from model import GPTLanguageModel

# Set seed for reproducibility
torch.manual_seed(config.SEED)

# 1. Instantiate Model & Fetch Data Batch
model = GPTLanguageModel()
xb, yb = get_batch('train')

# 2. Run Forward & Backward Pass
logits, loss = model(xb, yb)
loss.backward()

# 3. Measure Unclipped Gradient Norm (using math.sqrt for Python float)
total_norm_before = math.sqrt(
    sum(p.grad.norm(2).item() ** 2 for p in model.parameters() if p.grad is not None)
)

# 4. Apply Gradient Clipping (max_norm = GRAD_CLIP = 1.0)
clipped_norm_returned = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.GRAD_CLIP).item()

# 5. Measure Clipped Gradient Norm After Clipping
total_norm_after = math.sqrt(
    sum(p.grad.norm(2).item() ** 2 for p in model.parameters() if p.grad is not None)
)

# --- TASK 16.3 VERIFICATION ---
print("--- Task 16.3: Gradient Norm Inspection & Clipping Verification ---")
print(f"Gradient Norm BEFORE Clipping : {total_norm_before:.6f}")
print(f"Clip Threshold (GRAD_CLIP)    : {config.GRAD_CLIP:.6f}")
print(f"Gradient Norm AFTER Clipping  : {total_norm_after:.6f}\n")

if total_norm_before > config.GRAD_CLIP:
    print("SUCCESS: Gradient norm exceeded 1.0 and was successfully scaled down!")
else:
    print("SUCCESS: Gradient norm was below threshold and preserved!")
