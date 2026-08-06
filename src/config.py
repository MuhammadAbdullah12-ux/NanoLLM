import os

# Central Configuration for NanoLLM Experiments

# 1. System & Reproducibility Settings
SEED = 42

# 2. File & Directory Paths
DATA_DIR = './data/raw'
CHECKPOINT_DIR = 'checkpoints'
RUN_LOGS_DIR = 'experiments/run_logs'

# 3. Model Architecture Parameters (MNIST Baseline)
INPUT_DIM = 784   # 28x28 pixels
HIDDEN_DIM1 = 128
HIDDEN_DIM2 = 64
OUTPUT_DIM = 10   # Digits 0-9

# 4. Training Hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 5

# Ensure output directories exist automatically upon import
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RUN_LOGS_DIR, exist_ok=True)
