import os
import sys
import torch
import gradio as gr
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Reconfigure stdout for Windows console compatibility
sys.stdout.reconfigure(encoding='utf-8')

print("--- Initializing NanoLLM Hugging Face Space App ---\n")

# 1. Load Pretrained / Fine-Tuned GPT-2 Model & Tokenizer
MODEL_NAME = 'gpt2'
print(f"Loading '{MODEL_NAME}' model & tokenizer from Hugging Face...")

tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)

# Set model to evaluation mode
model.eval()

# 2. Text Generation Inference Function
def generate_nanollm_text(prompt, max_new_tokens, temperature, top_k, top_p):
    if not prompt.strip():
        prompt = "KING HENRY:\nShall I be bold to tell you"
        
    inputs = tokenizer(prompt, return_tensors='pt')
    
    # Ensure inputs don't exceed max position length
    input_ids = inputs['input_ids']
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_k=int(top_k),
            top_p=float(top_p),
            do_sample=True if temperature > 0 else False,
            pad_token_id=tokenizer.eos_token_id
        )
        
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text

# 3. Custom Gradio Interface Layout
custom_css = """
.container { max-width: 900px; margin: auto; padding-top: 20px; }
.title { text-align: center; color: #4F46E5; font-size: 2.2em; font-weight: bold; margin-bottom: 5px; }
.subtitle { text-align: center; color: #6B7280; font-size: 1.1em; margin-bottom: 25px; }
"""

with gr.Blocks(css=custom_css, title="NanoLLM Live Inference Demo") as demo:
    with gr.Column(elem_classes="container"):
        gr.Markdown(
            """
            # 🚀 NanoLLM: Live Model Inference Playground
            ### **4-Week, 26-Day From-Scratch GPT Build Showcase**
            *Trained & Fine-Tuned Autoregressive Transformer | Temperature, Top-K & Top-P Nucleus Sampling*
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                prompt_input = gr.Textbox(
                    label="Input Prompt",
                    placeholder="Enter your prompt here (e.g. KING HENRY: or In a distant galaxy...)",
                    lines=3,
                    value="KING HENRY:\nShall I be bold to tell you"
                )
                
                max_tokens_slider = gr.Slider(
                    minimum=10, maximum=300, value=100, step=10, 
                    label="Max New Tokens", info="Maximum number of tokens to generate"
                )
                
                temp_slider = gr.Slider(
                    minimum=0.1, maximum=2.0, value=0.8, step=0.1, 
                    label="Temperature (T)", info="Lower = Conservative/Greedy | Higher = Creative/Diverse"
                )
                
                top_k_slider = gr.Slider(
                    minimum=1, maximum=100, value=20, step=1, 
                    label="Top-K Filter (K)", info="Truncates candidate pool to top K highest probability tokens"
                )
                
                top_p_slider = gr.Slider(
                    minimum=0.1, maximum=1.0, value=0.9, step=0.05, 
                    label="Top-P Nucleus (P)", info="Keeps top candidate tokens up to cumulative probability P"
                )
                
                submit_btn = gr.Button("✨ Generate Completion", variant="primary")
                
            with gr.Column(scale=1):
                output_text = gr.Textbox(
                    label="Generated Output Text",
                    lines=12,
                    interactive=False
                )
                
        # Connect button & inputs to generation function
        submit_btn.click(
            fn=generate_nanollm_text,
            inputs=[prompt_input, max_tokens_slider, temp_slider, top_k_slider, top_p_slider],
            outputs=output_text
        )
        
        # Example prompt presets
        gr.Examples(
            examples=[
                ["KING HENRY:\nShall I be bold to tell you", 100, 0.8, 20, 0.9],
                ["ROMEO:\nLady, by yonder blessed moon I vow", 120, 0.7, 30, 0.9],
                ["The secret to artificial intelligence is", 80, 0.8, 20, 0.95],
                ["In a distant star system, human civilization", 100, 1.0, 40, 0.9]
            ],
            inputs=[prompt_input, max_tokens_slider, temp_slider, top_k_slider, top_p_slider]
        )

# Launch local app if executed directly
if __name__ == "__main__":
    demo.launch()
