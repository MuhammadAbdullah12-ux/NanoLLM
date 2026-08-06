import os

# Central Configuration for NanoLLM & GPT Model Experiments

# 1. System & Reproducibility Settings
SEED = 42

# 2. File & Directory Paths
DATA_DIR = './data/raw'
PREPARED_DATA_DIR = './data/prepared'
CHECKPOINT_DIR = 'checkpoints'
RUN_LOGS_DIR = 'experiments/run_logs'

# 3. GPT Model Architecture Parameters
VOCAB_SIZE = 65   # Unique characters in Tiny Shakespeare
N_EMBD = 32       # Embedding vector dimension (d_model)
BLOCK_SIZE = 8    # Context window length (T)
HEAD_SIZE = 16    # Dimension of single self-attention head projection (d_k)

# 4. Training Hyperparameters
BATCH_SIZE = 4    # Mini-batch size (B)
LEARNING_RATE = 0.001
EPOCHS = 5

# Ensure output directories exist automatically upon import
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PREPARED_DATA_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RUN_LOGS_DIR, exist_ok=True)
