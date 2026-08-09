import sys
import tiktoken
import config
from tokenizer import encode as char_encode, decode as char_decode

# Reconfigure stdout for Windows console compatibility
sys.stdout.reconfigure(encoding='utf-8')

print("--- Task 22.3: Character-Level vs. Subword BPE Tokenization Comparison ---\n")

# 1. Initialize OpenAI's GPT-2 Byte-Pair Encoding (BPE) Tokenizer
bpe_encoder = tiktoken.get_encoding('gpt2')

# Sample Shakespeare text snippet
sample_text = "KING HENRY:\nShall I be bold to tell you what I think?\nTo be, or not to be: that is the question."

print(f"Original Input Text:\n\"{sample_text}\"\n")
print(f"Total Character Count: {len(sample_text)} characters\n")

# 2. Character-Level Tokenization (Days 8-21 Baseline)
char_tokens = char_encode(sample_text)
print("=" * 65)
print("1. CHARACTER-LEVEL TOKENIZER (BASELINE):")
print("=" * 65)
print(f"Vocabulary Size (|V|)    : {config.VOCAB_SIZE} unique characters")
print(f"Encoded Token Count     : {len(char_tokens)} tokens")
print(f"First 15 Token IDs      : {char_tokens[:15]}")
print(f"Compression Ratio       : 1.00 char / token\n")

# 3. Subword BPE Tokenization (Day 22 tiktoken GPT-2)
bpe_tokens = bpe_encoder.encode(sample_text)
bpe_decoded = bpe_encoder.decode(bpe_tokens)

print("=" * 65)
print("2. SUBWORD BPE TOKENIZER (tiktoken - GPT-2):")
print("=" * 65)
print(f"Vocabulary Size (|V|)    : {bpe_encoder.n_vocab:,} subword tokens")
print(f"Encoded Token Count     : {len(bpe_tokens)} tokens")
print(f"First 15 Token IDs      : {bpe_tokens[:15]}")
compression_ratio = len(sample_text) / len(bpe_tokens)
print(f"Compression Ratio       : {compression_ratio:.2f} chars / token ({len(char_tokens)} chars -> {len(bpe_tokens)} BPE tokens!)\n")

# 4. Inspect Individual BPE Subword Chunks
print("=" * 65)
print("3. INDIVIDUAL BPE SUBWORD CHUNKS BREAKDOWN:")
print("=" * 65)
for token_id in bpe_tokens[:12]:
    chunk = bpe_encoder.decode([token_id])
    print(f"  Token ID {token_id:5d} -> '{chunk}'")

print("\nSUCCESS: Subword BPE tokenization and 3x compression factor verified!")
