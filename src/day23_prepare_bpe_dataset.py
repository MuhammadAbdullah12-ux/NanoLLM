import os
import sys
import torch
import config
from bpe_tokenizer import encode_bpe

# Reconfigure stdout for Windows console compatibility
sys.stdout.reconfigure(encoding='utf-8')

print("--- Task 23.2: BPE Subword Dataset Pre-processing & Tensor Generation ---\n")

# 1. Verify and Load Raw Dataset File
raw_data_path = os.path.join(config.DATA_DIR, 'input.txt')

if not os.path.exists(raw_data_path):
    print(f"❌ Error: Raw dataset file not found at '{raw_data_path}'. Please run previous dataset downloads first.")
    sys.exit(1)

with open(raw_data_path, 'r', encoding='utf-8') as f:
    text = f.read()

total_chars = len(text)
print(f"Loaded raw dataset from '{raw_data_path}' ({total_chars:,} characters)")

# 2. Encode Entire Dataset using Subword BPE Tokenizer (tiktoken gpt2)
print("Encoding dataset with Subword BPE Tokenizer (tiktoken)...")
bpe_tokens = encode_bpe(text)
total_bpe_tokens = len(bpe_tokens)

compression_ratio = total_chars / total_bpe_tokens
print(f"Total Subword BPE Tokens : {total_bpe_tokens:,} tokens")
print(f"Subword Compression Ratio : {compression_ratio:.2f} chars / token ({total_chars:,} chars -> {total_bpe_tokens:,} BPE tokens!)\n")

# 3. 90% / 10% Train & Validation Split
n_train = int(0.9 * total_bpe_tokens)
train_bpe_tokens = bpe_tokens[:n_train]
val_bpe_tokens = bpe_tokens[n_train:]

print(f"Train Dataset BPE Tokens (90%) : {len(train_bpe_tokens):,} tokens")
print(f"Val Dataset BPE Tokens   (10%) : {len(val_bpe_tokens):,} tokens\n")

# 4. Convert to PyTorch LongTensors and Save to File
train_tensor = torch.tensor(train_bpe_tokens, dtype=torch.long)
val_tensor = torch.tensor(val_bpe_tokens, dtype=torch.long)

os.makedirs(config.PREPARED_DATA_DIR, exist_ok=True)
train_save_path = os.path.join(config.PREPARED_DATA_DIR, 'bpe_train.pt')
val_save_path = os.path.join(config.PREPARED_DATA_DIR, 'bpe_val.pt')

torch.save(train_tensor, train_save_path)
torch.save(val_tensor, val_save_path)

print("=" * 65)
print("SUCCESS: Subword BPE dataset tensors saved successfully!")
print(f"  • Train Tensor Path : '{train_save_path}' (Shape: {train_tensor.shape})")
print(f"  • Val Tensor Path   : '{val_save_path}' (Shape: {val_tensor.shape})")
print("=" * 65)
