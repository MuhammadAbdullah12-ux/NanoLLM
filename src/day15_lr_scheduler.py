import os
import math
import matplotlib.pyplot as plt
import config

# --- TASK 15.3: COSINE LEARNING RATE SCHEDULER FUNCTION ---
def get_lr(it):
    # 1. Linear Warmup Phase (if iteration < WARMUP_ITERS)
    if it < config.WARMUP_ITERS:
        return config.LEARNING_RATE * (it / config.WARMUP_ITERS)
    
    # 2. Post-Decay Floor Phase (if iteration > LR_DECAY_ITERS)
    if it > config.LR_DECAY_ITERS:
        return config.MIN_LR
    
    # 3. Cosine Decay Phase (between WARMUP_ITERS and LR_DECAY_ITERS)
    decay_ratio = (it - config.WARMUP_ITERS) / (config.LR_DECAY_ITERS - config.WARMUP_ITERS)
    assert 0.0 <= decay_ratio <= 1.0
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # Coeff ranges from 1.0 down to 0.0
    return config.MIN_LR + coeff * (config.LEARNING_RATE - config.MIN_LR)

# --- EXECUTION & PLOTTING FOR VERIFICATION ---
if __name__ == '__main__':
    total_steps = 2500
    lr_list = [get_lr(step) for step in range(total_steps)]

    print("--- Task 15.3: Cosine Learning Rate Schedule Verification ---")
    print(f"Step 0     (Start)          | LR: {get_lr(0):.6f}")
    print(f"Step 50    (Mid Warmup)     | LR: {get_lr(50):.6f}")
    print(f"Step 100   (End Warmup/Max) | LR: {get_lr(100):.6f}")
    print(f"Step 1000  (Mid Cosine)     | LR: {get_lr(1000):.6f}")
    print(f"Step 2000  (End Cosine/Min) | LR: {get_lr(2000):.6f}")
    print(f"Step 2500  (Post Floor)     | LR: {get_lr(2500):.6f}\n")

    # Plot schedule using Matplotlib
    os.makedirs(config.RUN_LOGS_DIR, exist_ok=True)
    plot_path = os.path.join(config.RUN_LOGS_DIR, 'cosine_lr_schedule.png')

    plt.figure(figsize=(9, 5))
    plt.plot(range(total_steps), lr_list, label='Learning Rate (get_lr)', color='teal', linewidth=2)
    plt.axvline(x=config.WARMUP_ITERS, color='orange', linestyle='--', label='End Warmup (100)')
    plt.axvline(x=config.LR_DECAY_ITERS, color='red', linestyle='--', label='End Decay (2000)')
    plt.title('Task 15.3: Cosine Learning Rate Schedule with Linear Warmup')
    plt.xlabel('Training Iterations (Step)')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(plot_path)
    print(f"SUCCESS: Learning rate schedule plot saved to '{plot_path}'!")
