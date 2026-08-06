import os
import torch

# Path to raw dataset
DATA_PATH = './data/raw/input.txt'

# 1. Read the dataset text
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    text = f.read()

# 2. Extract unique characters and vocabulary size
chars = sorted(list(set(text)))
vocab_size = len(chars)

# 3. Dictionaries mapping character <-> integer ID
stoi = { ch: i for i, ch in enumerate(chars) }
itos = { i: ch for i, ch in enumerate(chars) }

def encode(s: str) -> list[int]:
    return [stoi[c] for c in s]

def decode(l: list[int]) -> str:
    return ''.join([itos[i] for i in l])

print("--- Task 8.3: Testing Tokenizer Encoder & Decoder ---")
test_text = "hii there"
encoded = encode(test_text)
decoded = decode(encoded)

print(f"Original Text: '{test_text}'")
print(f"Encoded Token IDs: {encoded}")
print(f"Decoded Text:  '{decoded}'\n")

# --- TASK 8.4 & 8.5: FULL TENSOR ENCODING & 90/10 TRAIN/VAL SPLIT ---
print("--- Task 8.4 & 8.5: Encoding Full Dataset & 90/10 Train/Val Split ---")

# Convert all 1.1 million characters into a single PyTorch Tensor
data = torch.tensor(encode(text), dtype=torch.long)

# Split into 90% Training Data and 10% Validation Data
n = int(0.9 * len(data))
train_data = data[:n]
val_data   = data[n:]

print(f"Full Dataset Tensor Shape: {data.shape}, Type: {data.dtype}")
print(f"Train Dataset Token Count: {len(train_data):,} tokens (90%)")
print(f"Val Dataset Token Count:   {len(val_data):,} tokens (10%)")

# Save tokenized datasets to data/prepared/ directory for Week 2 training
os.makedirs('./data/prepared', exist_ok=True)
torch.save(train_data, './data/prepared/train.pt')
torch.save(val_data, './data/prepared/val.pt')
print("\nSUCCESS: Tokenized datasets saved to 'data/prepared/train.pt' and 'data/prepared/val.pt'!")
