import os

# Central Configuration for NanoLLM & GPT Model Experiments

# 1. System & Reproducibility Settings
SEED = 42

# 2. File & Directory Paths
DATA_DIR = './data/raw'
PREPARED_DATA_DIR = './data/prepared'
CHECKPOINT_DIR = 'checkpoints'
RUN_LOGS_DIR = 'experiments/run_logs'

# 3. Scaled GPT Model Architecture Parameters (Task 17.2)
VOCAB_SIZE = 65   # Unique characters in Tiny Shakespeare
N_EMBD = 64       # Scaled embedding vector dimension (was 32)
BLOCK_SIZE = 64   # Scaled context window length (was 8)
NUM_HEADS = 4     # Number of parallel multi-head attention heads
HEAD_SIZE = N_EMBD // NUM_HEADS  # Dimension per head (64 // 4 = 16)
N_LAYER = 4       # Scaled number of Transformer Blocks (was 3)

# 4. Training Hyperparameters & Scheduler Parameters
BATCH_SIZE = 16       # Scaled mini-batch size (was 4)
LEARNING_RATE = 0.001 # Maximum learning rate (lr_max)
EPOCHS = 5

WARMUP_ITERS = 100     # Number of steps to ramp up from 0 to LEARNING_RATE
LR_DECAY_ITERS = 2000  # Number of steps to decay down to MIN_LR
MIN_LR = 1e-4          # Minimum learning rate floor (0.0001)

GRAD_CLIP = 1.0        # Maximum allowed gradient norm threshold
WEIGHT_DECAY = 0.1     # Decoupled L2 regularization coefficient for AdamW
DROPOUT = 0.1          # Probability of zeroing activations during training (10%)

TEMPERATURE = 0.8      # Logit scaling factor (lower = conservative, higher = creative)
TOP_K = 20             # Truncate sampling pool to top K highest probability tokens
TOP_P = 0.9            # Nucleus sampling threshold (keep top tokens up to cumulative prob 90%)

# Ensure output directories exist automatically upon import
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PREPARED_DATA_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RUN_LOGS_DIR, exist_ok=True)
