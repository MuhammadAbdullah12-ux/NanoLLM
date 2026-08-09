import sys
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import config

# Reconfigure stdout for Windows console compatibility
sys.stdout.reconfigure(encoding='utf-8')

# Set seed for reproducible generation
torch.manual_seed(config.SEED)

print("--- Task 25.3: Loading & Evaluating Pretrained GPT-2 Small (Hugging Face) ---\n")

# 1. Load Pretrained GPT-2 Tokenizer and Model Weights from Hugging Face
print("Loading pretrained 'gpt2' model & tokenizer from Hugging Face hub...")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

model.eval()

# 2. Compute and Compare Parameter Counts
gpt2_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("\n" + "=" * 65)
print("📊 MODEL CAPACITY & PARAMETER COUNT COMPARISON:")
print("=" * 65)
print(f"  • NanoLLM Mini-GPT (Our From-Scratch Model) :      211,777 parameters")
print(f"  • OpenAI GPT-2 Small (Pretrained Model)   : 124,439,808 parameters")
print(f"  • Capacity Scale Factor                  : {gpt2_params / 211777:.1f}x larger capacity!\n")

# 3. Test Generation Quality across Prompts
prompts = [
    "KING HENRY:\nShall I be bold to tell you",
    "The secret to building artificial intelligence is",
    "In a distant star system, human civilization"
]

print("=" * 65)
print("🎭 PRETRAINED GPT-2 (124M) TEXT GENERATION DEMONSTRATION:")
print("=" * 65)

for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors='pt')
    
    with torch.no_grad():
        output_ids = model.generate(
            inputs['input_ids'],
            max_new_tokens=60,
            temperature=config.TEMPERATURE,
            top_k=config.TOP_K,
            top_p=config.TOP_P,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"\n--- PROMPT: '{prompt}' ---")
    print(generated_text)
    print("-" * 50)

print("\nSUCCESS: Pretrained GPT-2 Small (124M) loaded and evaluated successfully!")
