import os
import sys
import torch
import config

# Reconfigure stdout for Windows console compatibility
sys.stdout.reconfigure(encoding='utf-8')

# 1. Load Pre-processed Subword BPE Tensors
train_data_path = os.path.join(config.PREPARED_DATA_DIR, 'bpe_train.pt')
val_data_path = os.path.join(config.PREPARED_DATA_DIR, 'bpe_val.pt')

if os.path.exists(train_data_path) and os.path.exists(val_data_path):
    train_data_bpe = torch.load(train_data_path, weights_only=False)
    val_data_bpe = torch.load(val_data_path, weights_only=False)
else:
    train_data_bpe = None
    val_data_bpe = None

def get_batch_bpe(split='train', batch_size=config.BATCH_SIZE, block_size=config.BLOCK_SIZE):
    """
    Generates a mini-batch of BPE subword inputs (X) and targets (Y).
    X shape: [batch_size, block_size]
    Y shape: [batch_size, block_size] (shifted right by 1 token)
    """
    data = train_data_bpe if split == 'train' else val_data_bpe
    if data is None:
        raise FileNotFoundError("BPE subword dataset tensors not found! Run 'python src/day23_prepare_bpe_dataset.py' first.")
        
    # Generate random starting indices in the dataset
    ix = torch.randint(len(data) - block_size, (batch_size,))
    
    # Extract input context X and shifted target sequence Y
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    
    return x, y

# Verification & Demonstration
if __name__ == '__main__':
    print("--- Task 23.3: BPE Data Loader Verification ---\n")
    if train_data_bpe is not None:
        xb, yb = get_batch_bpe('train', batch_size=4, block_size=8)
        print(f"Train Dataset Total BPE Tokens: {len(train_data_bpe):,}")
        print(f"BPE Batch Input  Shape (X)    : {xb.shape}")
        print(f"BPE Batch Target Shape (Y)    : {yb.shape}\n")
        print("Sample BPE Token IDs Input (X[0])  :", xb[0].tolist())
        print("Sample BPE Token IDs Target (Y[0]) :", yb[0].tolist())
        print("\nSUCCESS: BPE Data Loader verified!")
    else:
        print("⚠️ Warning: Run 'python src/day23_prepare_bpe_dataset.py' first to generate BPE dataset tensors.")
