import torch
import config
from model import GPTLanguageModel

# Set seed for reproducibility
torch.manual_seed(config.SEED)

# 1. Instantiate Scaled Model
model = GPTLanguageModel()

# 2. Calculate Total Trainable Parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("--- Task 17.3: Scaled GPT Model Parameter Breakdown ---\n")

print(f"Token Embedding Parameters   : {config.VOCAB_SIZE} * {config.N_EMBD} = {config.VOCAB_SIZE * config.N_EMBD:,}")
print(f"Positional Embedding Params  : {config.BLOCK_SIZE} * {config.N_EMBD} = {config.BLOCK_SIZE * config.N_EMBD:,}")
print(f"Number of Transformer Blocks : {config.N_LAYER} Blocks")
print(f"Embedding Feature Dimension  : d_model = {config.N_EMBD}")
print(f"Context Window Length       : T = {config.BLOCK_SIZE}\n")

print("-" * 50)
print(f"TOTAL TRAINABLE PARAMETERS: {total_params:,} parameters")
print("-" * 50)

# Print comparison with Baseline Model
baseline_params = 25153  # Approximate parameters of Day 13 baseline
growth_factor = total_params / baseline_params
print(f"\nModel Capacity Scaling:")
print(f"  Baseline Model (Day 13): ~{baseline_params:,} parameters")
print(f"  Scaled Model (Day 17)  : ~{total_params:,} parameters")
print(f"  Capacity Expansion     : {growth_factor:.2f}x Larger Brain Size!\n")
