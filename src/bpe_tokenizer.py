import tiktoken

# Initialize GPT-2 Byte-Pair Encoding (BPE) Subword Tokenizer
_bpe_encoder = tiktoken.get_encoding('gpt2')

# Total vocabulary size for GPT-2 BPE tokenizer
VOCAB_SIZE_BPE = _bpe_encoder.n_vocab  # 50,257

def encode_bpe(text: str) -> list[int]:
    """Encodes a string into a list of BPE subword token IDs."""
    return _bpe_encoder.encode(text)

def decode_bpe(tokens: list[int]) -> str:
    """Decodes a list of BPE subword token IDs back into a string."""
    return _bpe_encoder.decode(tokens)
