import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from scl.compressors.arithmetic_coding import AECParams, ArithmeticEncoder, ArithmeticDecoder  
from scl.core.data_block import DataBlock
from scl.core.prob_dist import Frequencies
from scl.utils.bitarray_utils import BitArray, bitarray_to_uint, uint_to_bitarray
from scl.compressors.probability_models import FreqModelBase
import math
import numpy as np
import copy
from typing import List, Dict
import time

# -------------------------
# 1. Load TinyLlama 
# -------------------------
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print("Device:", "GPU" if torch.cuda.is_available() else "CPU")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

print("Loading model")

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32, low_cpu_mem_usage=True)
model.to(device)
model.eval()
torch.set_grad_enabled(False)

# -------------------------
# 2. LLM-based Frequency Model
# -------------------------
class LLMFreqModel(FreqModelBase):
    """
    Frequency model using a language model (TinyLlama) to provide symbol frequencies.
    Maintains context tokens and uses model's KV cache for efficiency.
    1. On each symbol update, advance the model by one token.
    2. On frequency query, use cached logits if available; otherwise, run model on full context.
    3. Convert logits → softmax → probabilities → integer frequencies for arithmetic coding.
    4. Context length is limited to MAX_CONTEXT_FEED for efficiency.
    5. Frequencies are scaled to fit within MAX_ALLOWED_TOTAL_FREQ.
    6. Designed for use with SCL arithmetic coder.
    """

    def __init__(self, tokenizer, model, params: AECParams,
                 context_max_tokens=None, max_context_feed=512, device_override=None):

        self.tokenizer = tokenizer
        self.model = model
        self.params = params
        self.device = device_override or device

        # Context storage
        self.context_tokens = []

        # Limits
        self.context_max = context_max_tokens or getattr(tokenizer, "model_max_length", 2048)
        self.MAX_CONTEXT_FEED = min(max_context_feed, self.context_max)
        self.max_total_freq = max(1, params.MAX_ALLOWED_TOTAL_FREQ - 1)

        # KV cache
        self.past_key_values = None
        self._cached_next_logits_gpu = None  

        # vocab size
        self.vocab_size = tokenizer.vocab_size

    def update_model(self, symbol: int):
        """
        Advance the model by one token with full GPU KV cache.
        """
        self.context_tokens.append(symbol)

        # sliding window of previously encoded tokens to add context
        if len(self.context_tokens) > self.MAX_CONTEXT_FEED:
            self.context_tokens = self.context_tokens[-self.MAX_CONTEXT_FEED:]

        if self.past_key_values is not None:
            input_ids = torch.tensor([[symbol]], dtype=torch.long, device=self.device)

            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    past_key_values=self.past_key_values,
                    use_cache=True
                )

            self.past_key_values = outputs.past_key_values
            self._cached_next_logits_gpu = outputs.logits[:, -1, :]  

        else:
            self._cached_next_logits_gpu = None  

    @property
    def freqs_current(self) -> Frequencies:
        """
        Get current frequencies as Frequencies object.
        Called by arithmetic coder to get symbol frequencies.
        """
        # uniform distribution for empty context
        if len(self.context_tokens) == 0:
            return Frequencies({i: 1 for i in range(self.vocab_size)})

        # FAST PATH – GPU cached logits
        if self._cached_next_logits_gpu is not None and self.past_key_values is not None:
            probs_gpu = torch.softmax(self._cached_next_logits_gpu, dim=-1)
            probs = probs_gpu.squeeze().cpu().numpy()  # convert ONCE
            return self._probs_to_freqs(probs)

        # SLOW PATH – initial warmup
        context = self.context_tokens[-self.MAX_CONTEXT_FEED:]
        input_ids = torch.tensor([context], dtype=torch.long, device=self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, use_cache=True)

        self.past_key_values = outputs.past_key_values
        self._cached_next_logits_gpu = outputs.logits[:, -1, :]

        probs_gpu = torch.softmax(self._cached_next_logits_gpu, dim=-1)
        probs = probs_gpu.squeeze().cpu().numpy()
        return self._probs_to_freqs(probs)

    def _probs_to_freqs(self, probs):
        """
        Convert probabilities to integer frequencies within max_total_freq.
        Ensures no zero frequencies and total frequency does not exceed max_total_freq.
        This ensures arithmetic coder gets valid frequency tables.
        """
        scaled = probs * self.max_total_freq
        int_freqs = scaled.astype(np.int32)
        int_freqs[int_freqs < 1] = 1

        total = int_freqs.sum()
        if total > self.max_total_freq:
            factor = self.max_total_freq / total
            int_freqs = np.maximum(1, (int_freqs * factor).astype(np.int32))

        return Frequencies({i: int(int_freqs[i]) for i in range(len(int_freqs))})


        return Frequencies({i: int(int_freqs[i]) for i in range(len(int_freqs))})

def tokens_from_text_in_chunks(text: str, max_chunk_tokens: int = 512):
    """
    Yield token chunks from a long text, never exceeding max_chunk_tokens.
    This prevents context overflow in the LLM. (Max 2048 tokens for TinyLlama.)
    """
    vocab_size = tokenizer.vocab_size if hasattr(tokenizer, "vocab_size") else tokenizer.model_max_length
    start = 0
    text_len = len(text)
    while start < text_len:
        chunk_text = text[start:start + max_chunk_tokens * 4]  # heuristic: 4 chars ~ 1 token
        tokens = tokenizer.encode(chunk_text, add_special_tokens=False)
        if len(tokens) > max_chunk_tokens:
            tokens = tokens[:max_chunk_tokens]
        yield tokens
        start += len(chunk_text)

def text_from_tokens(token_ids: List[int]) -> str:
    # 1. Remove special tokens safely (BOS, EOS, UNK, PAD, system tokens)
    cleaned = [tid for tid in token_ids if tid not in tokenizer.all_special_ids]

    # 2. Decode while skipping any remaining special tokens
    return tokenizer.decode(cleaned, skip_special_tokens=True)

# -------------------------
# 4. Compression / Decompression functions
# -------------------------
def compress_text_to_bitarray(text: str, params: AECParams, freq_model_factory, chunk_size=512):
    """
    Compress text using ArithmeticEncoder and freq_model_factory to create a fresh freq_model
    For each token chunk: text → tokens → arithmetic encoder → bitarray
    Returns BitArray which uses SCL's BitArray for bit storage.
    """
    full_bits = BitArray()

    # Tokenize and yield chunks
    freq_model_enc = freq_model_factory(params)
    encoder = ArithmeticEncoder(params, freq_model_enc)

    for token_chunk in tokens_from_text_in_chunks(text, max_chunk_tokens=chunk_size):
        data_block = DataBlock(token_chunk)
        encoded_chunk = encoder.encode_block(data_block)
        full_bits.extend(encoded_chunk)

    return full_bits

def decompress_bitarray_to_text(encoded_bitarray: BitArray, params: AECParams, freq_model_factory, chunk_size=512):
    """
    Decode using ArithmeticDecoder and freq_model_factory to create a fresh freq_model (decoder uses its own copy)
    For each token chunk: bitarray → arithmetic decoder → tokens → text
    Returns recovered text string.
    """
    pos = 0
    decoded_tokens = []

    while pos < len(encoded_bitarray):
        freq_model_dec = freq_model_factory(params)
        decoder = ArithmeticDecoder(params, freq_model_dec)
        decoded_block, used = decoder.decode_block(encoded_bitarray[pos:])
        decoded_tokens.extend(decoded_block.data_list)
        pos += used

    return text_from_tokens(decoded_tokens)

# -------------------------
# 5. CLI / demo
# -------------------------
def main_demo():
    """
    Runs compression/decompression pipeline demo.
    Compresses input text file (or demo text) and decompresses to verify correctness.
    """
    import argparse
    parser = argparse.ArgumentParser(description="LLM-based compression demo using SCL arithmetic coder (TinyLlama).")
    parser.add_argument("--infile", type=str, default=None, help="Path to input text file. If omitted, a short demo sentence is used.")
    parser.add_argument("--outfile", type=str, default="out.sclbit", help="Path to write compressed bits (not used for decompression here).")
    parser.add_argument("--precision", type=int, default=32, help="Arithmetic coder precision (bits). Default 32.")
    args = parser.parse_args()

    if args.infile:
        with open(args.infile, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        text = "The weather today is sunny and pleasant. I expect light winds and warm sunshine."

    params = AECParams(DATA_BLOCK_SIZE_BITS=32, PRECISION=args.precision)

    # Frequency model factory to create encoder/decoder models (fresh instances)
    def freq_model_factory(p):
        return LLMFreqModel(tokenizer, model, p, context_max_tokens=tokenizer.model_max_length)

    print("=== Compressing ===")
    start_time = time.time()

    encoded_bitarray = compress_text_to_bitarray(text, params, freq_model_factory)
    end_time = time.time()
    compression_time = end_time - start_time

    compressed_bits = len(encoded_bitarray)
    compressed_bytes = compressed_bits / 8

    original_bytes = len(text.encode("utf-8"))

    compression_ratio = original_bytes / compressed_bytes if compressed_bytes > 0 else float("inf")
    print(f"Compressed length (bits): {compressed_bits}")
    print(f"Original size (bytes): {original_bytes}")
    print(f"Compressed size (bytes): {compressed_bytes:.2f}")
    print(f"Compression ratio (orig / comp): {compression_ratio:.3f}x")
    print(f"Compression time: {compression_time:.3f} sec")

    # Save to file (raw bitarray serialization - here we store bitarray.bin)
    bit_length = len(encoded_bitarray)
    with open(args.outfile, "wb") as fout:
        # Write the length as 4 bytes, big endian
        fout.write(bit_length.to_bytes(4, "big"))
        # Then write actual bits
        fout.write(encoded_bitarray.tobytes())

    print("Saved compressed file to", args.outfile)

    print("\n=== Decompressing and verifying ===")
    # load back
    with open(args.outfile, "rb") as fin:
        raw = fin.read()

    # First 4 bytes = bit length
    bit_length = int.from_bytes(raw[:4], "big")
    payload = raw[4:]

    # Reconstruct bitarray
    loaded = BitArray()
    loaded.frombytes(payload)

    # Trim padding bits
    loaded = loaded[:bit_length]

    start_dec = time.time()
    recovered_text = decompress_bitarray_to_text(loaded, params, freq_model_factory)
    end_dec = time.time()
    decompression_time = end_dec - start_dec

    print("\nRecovered text snippet:", recovered_text[:200])
    print("\nMatch exact:", recovered_text == text)
    print(f"\nDecompression time: {decompression_time:.3f} sec")



if __name__ == "__main__":
    main_demo()