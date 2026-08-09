import os
import sys
import torch
import src.config as config
from src.model import GPTLanguageModel
from src.tokenizer import encode, decode

# Reconfigure stdout for Windows console compatibility
sys.stdout.reconfigure(encoding='utf-8')

def main():
    print("=" * 60)
    print("🎭 NanoLLM Interactive Generation Playground (Day 20)")
    print("=" * 60)
    
    # 1. Instantiate Model Architecture
    model = GPTLanguageModel()
    
    # 2. Load Trained Checkpoint
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, 'gpt_shakespeare.pt')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded trained checkpoint from '{checkpoint_path}' (Step {checkpoint.get('step', 'N/A')})")
        else:
            model.load_state_dict(checkpoint)
            print(f"Loaded model weights from '{checkpoint_path}'")
    else:
        print(f"⚠️ Warning: No trained checkpoint found at '{checkpoint_path}'. Using un-trained model.")

    # 3. Set Model to Evaluation Mode
    model.eval()
    
    print("\nCurrent Sampling Controls:")
    print(f"  • Temperature (T) : {config.TEMPERATURE}")
    print(f"  • Top-K (K)       : {config.TOP_K}")
    print(f"  • Top-P (Nucleus) : {config.TOP_P}\n")
    print("-" * 60)
    
    # 4. Interactive Prompt Loop
    prompt_text = input("Enter prompt (or press Enter for default 'KING HENRY:'): ").strip()
    if not prompt_text:
        prompt_text = "KING HENRY:"
        
    print(f"\nGenerating 300 characters starting from prompt: '{prompt_text}'...\n")
    print("=" * 60)
    
    # 5. Tokenize Prompt
    prompt_tokens = encode(prompt_text)
    context = torch.tensor([prompt_tokens], dtype=torch.long)
    
    # 6. Generate Text
    with torch.no_grad():
        generated_indices = model.generate(
            context, 
            max_new_tokens=300, 
            temperature=config.TEMPERATURE, 
            top_k=config.TOP_K, 
            top_p=config.TOP_P
        )[0].tolist()
        
    generated_text = decode(generated_indices)
    
    print(generated_text)
    print("=" * 60)

if __name__ == '__main__':
    main()
