import os
import sys
try:
    # Try to get the path if running as a standard script (.py)
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Fallback: Get the current working directory if running in Jupyter/Interactive
    # (Assumes notebook/terminal is currently inside the 'Project' folder)
    current_dir = os.getcwd()

# 1. Go up one level (from 'Project' -> 'scl')
parent_dir = os.path.dirname(current_dir)

# 2. Go up one more level (from 'scl' -> 'EE274-LLM-Compressor' root)
root_dir = os.path.dirname(parent_dir)

# 3. Add the root to the path so scl can be imported
if root_dir not in sys.path:
    sys.path.append(root_dir)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, MambaConfig, MambaForCausalLM, logging as hf_logging
from scl.compressors.arithmetic_coding import AECParams, ArithmeticEncoder, ArithmeticDecoder  
from scl.core.data_block import DataBlock
from scl.core.prob_dist import Frequencies, ProbabilityDist
from scl.utils.bitarray_utils import BitArray, bitarray_to_uint, uint_to_bitarray
from scl.compressors.probability_models import FreqModelBase
import math
import numpy as np
import copy
from typing import List, Dict, Optional
import time

# -------------------------
# Utilities: model loader
# -------------------------
def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    
    # MPS on macOS
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_tokenizer_and_model(
    model_choice: str,
    dtype: torch.dtype,
    hf_token: Optional[str] = None,
):
    """
    Load tokenizer and model according to user's choice.
    model_choice: 'tinyllama', 'llama3', 'mamba'
    dtype: torch.float16 or torch.float32
    Returns: (tokenizer, model, device)
    """
    device = choose_device()
    print(f"Selected device: {device}")

    use_auth = hf_token if hf_token else None

    if model_choice == "tinyllama":
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        # tinyllama is small; load normally
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, use_auth_token=use_auth)
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, low_cpu_mem_usage=True, use_auth_token=use_auth)
        model.to(device)  # Move to device after loading
    
    elif model_choice == "llama3":
        # Large gated model (70B). This will likely fail locally without proper hardware/auth.
        model_name = "meta-llama/Llama-3.3-70B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, use_auth_token=use_auth)
        # Attempt to use device_map='auto' if CUDA is available for large model
        try:
            # prefer device_map auto when CUDA is present
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=dtype,
                low_cpu_mem_usage=True,
                device_map="auto" if torch.cuda.is_available() else None,
                use_auth_token=use_auth,
            )
        except Exception as e:
            print("Warning: automatic device_map loading failed for LLaMA-3. Attempting CPU/memory-conservative load.")
            model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, low_cpu_mem_usage=True, use_auth_token=use_auth)
            model.to(device)  # Move to device after loading
    
    elif model_choice == "mamba":
        # use state-spaces / Mamba small model
        model_name = "state-spaces/mamba-130m-hf"
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b", use_fast=True, use_auth_token=use_auth)
        config = MambaConfig.from_pretrained(model_name)
        model = MambaForCausalLM.from_pretrained(
          model_name,
          config=config,
          dtype=dtype,
          device_map=None
        )
        model.to(device)
    
    elif model_choice == "mamba1.4b":
        # use state-spaces / Mamba small model
        model_name = "state-spaces/mamba-1.4b-hf"
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b", use_fast=True, use_auth_token=use_auth)
        config = MambaConfig.from_pretrained(model_name)
        model = MambaForCausalLM.from_pretrained(
          model_name,
          config=config,
          dtype=dtype,
          device_map=None
        )
        model.to(device)
    else:
        raise ValueError(f"Unknown model choice: {model_choice}")

    model.eval()
    torch.set_grad_enabled(False)

    # Print a short summary
    try:
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Loaded {model_choice} ({model_name}) model with ~{num_params:,} parameters (dtype={dtype}).")
    except Exception:
        print(f"Loaded {model_choice} model: {model_name} (dtype={dtype}).")

    return tokenizer, model, device

# ---------------------------------
# Simple Probability Distribution
# ---------------------------------

class SimpleProbDist:
    """
    A lightweight wrapper for probability distributions.
    Bypasses SCL's strict '1e-6' validation check.
    LLM probabilities can be very small, so we allow smaller values here.
    """
    def __init__(self, probs_array):
        self.probs = probs_array
        
    @property
    def entropy(self):
        # Calculate entropy efficiently on the numpy array
        # Filter out 0s to avoid log(0)
        p = self.probs
        p = p[p > 1e-20] # Ignore practically zero values
        return -np.sum(p * np.log2(p))
    
    def probability(self, symbol):
        if 0 <= symbol < len(self.probs):
            return self.probs[symbol]
        return 0.0

# ---------------------
# Metrics Tracker Class 
# ---------------------

class MetricsLog:
    """
    Class to track compression metrics during encoding.
    """
    def __init__(self):
        self.theoretical_bits = 0.0
        self.token_count = 0
        self.model_entropies = [] 

    def log(self, dist, actual_token):
        # We need to access the probability distribution from the model

        p = dist.probability(actual_token)
        p = max(p, 1e-12) # Avoid log(0)
        self.theoretical_bits += -math.log2(p)
        self.model_entropies.append(dist.entropy)
        self.token_count += 1

# --- HELPER: Report Generator ---
def generate_final_report(text, compressed_bits, logs, duration):
    print("\n" + "="*60)
    print(f"{'FINAL COMPRESSION REPORT':^60}")
    print("="*60)
    
    # Sizes
    original_bytes = len(text.encode('utf-8'))
    original_bits = original_bytes * 8
    comp_bits = len(compressed_bits)
    ratio = original_bits / comp_bits if comp_bits > 0 else 0
    bpb = comp_bits / original_bytes if original_bytes > 0 else 0
    
    # Entropy Metrics
    avg_theoretical = logs.theoretical_bits / max(1, logs.token_count)
    avg_actual = comp_bits / max(1, logs.token_count)
    avg_uncertainty = sum(logs.model_entropies) / max(1, len(logs.model_entropies))
    
    print(f"1. EFFICIENCY")
    print(f"   Original Size:    {original_bytes:.0f} bytes")
    print(f"   Compressed Size:  {comp_bits/8:.2f} bytes")
    print(f"   Compression Ratio: {ratio:.3f}x")
    print(f"   Bits Per Byte:    {bpb:.3f} bpb")
    print(f"   Speed:            {logs.token_count/duration:.1f} tokens/sec")

    
    print(f"\n2. ENTROPY")
    print(f"   LLM Theoretical:  {avg_theoretical:.4f} bits/token (Cross-Entropy)")
    print(f"   Actual SCL Code:  {avg_actual:.4f} bits/token")
    print(f"   Overhead:         {avg_actual - avg_theoretical:.4f} bits/token")
    print(f"   Model Uncertainty: {avg_uncertainty:.4f} bits (Avg Entropy)")
    print("="*60 + "\n")

# -------------------------
# LLM-based Frequency Model
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

    def __init__(
        self,
        tokenizer,
        model,
        device: torch.device,
        params: AECParams,
        context_tokens: Optional[List[int]] = None,
        max_context_feed: int = 512,
        top_k: int = 1024,
        metrics_log=None
    ):

        self.tokenizer = tokenizer
        self.model = model
        self.params = params
        self.device = device
        self.metrics_log = metrics_log


        # Context storage
        self.context_tokens = []

        self.is_mamba = "mamba" in model.config.model_type.lower()

        # Limits
        self.context_max = getattr(tokenizer, "model_max_length", 2048)
        self.MAX_CONTEXT_FEED = min(max_context_feed, self.context_max)
        self.max_total_freq = max(1, params.MAX_ALLOWED_TOTAL_FREQ - 1)

        # KV cache
        self.past_key_values = None
        self._cached_next_logits_gpu = None
        self._cached_next_logits = None
  

        # vocab size
        self.vocab_size = tokenizer.vocab_size
        
        # Cache for Mamba
        self.cache_params = None
        self.seq_len = 0        

    def update_model(self, symbol: int):
        """
        Advance the model by one token with full GPU KV cache.
        """
        if self.metrics_log:
            dist = self.get_current_scl_dist()
            self.metrics_log.log(dist, symbol)

        self.context_tokens.append(symbol)

        # sliding window of previously encoded tokens to add context
        if len(self.context_tokens) > self.MAX_CONTEXT_FEED:
            self.context_tokens = self.context_tokens[-self.MAX_CONTEXT_FEED:]

        input_ids = torch.tensor([[symbol]], dtype=torch.long, device=self.device)
        
        with torch.no_grad():
            if self.is_mamba:
              # Mamba Path
              if self.cache_params is None:

                # Initialize or Recovery
                  outputs = self.model(input_ids=input_ids, use_cache=True)
                  self.seq_len = 1 
              else:
                  # Incremental Step: Must pass cache_position
                  cache_position = torch.tensor([self.seq_len], device=self.device)
                  outputs = self.model(
                      input_ids=input_ids, 
                      cache_params=self.cache_params, 
                      use_cache=True, 
                      cache_position=cache_position
                  )
                  self.seq_len += 1
              
              self.cache_params = outputs.cache_params
              self._cached_next_logits_gpu = outputs.logits[:, -1, :]

            else:
              # Transformer Path
                if self.past_key_values is not None:
                  outputs = self.model(
                      input_ids=input_ids,
                      past_key_values=self.past_key_values,
                      use_cache=True
                  )
                  self.past_key_values = outputs.past_key_values
                  self._cached_next_logits_gpu = outputs.logits[:, -1, :]
                else:
                  outputs = self.model(input_ids=input_ids, use_cache=True)
                
                self.past_key_values = outputs.past_key_values
                self._cached_next_logits_gpu = outputs.logits[:, -1, :]
    
    def get_current_scl_dist(self):
        """
        Returns a probability distribution for the UPCOMING token.
        Uses SimpleProbDist to avoid SCL AssertionError on small probabilities.
        """
        if self._cached_next_logits_gpu is None:
            # Initial state: Uniform distribution
            vocab = self.tokenizer.vocab_size
            probs = np.ones(vocab) / vocab
            return SimpleProbDist(probs)
            
        # Get raw probabilities (no thresholding)
        probs = torch.softmax(self._cached_next_logits_gpu, dim=-1).squeeze().cpu().numpy()
        return SimpleProbDist(probs)             

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
        has_cache = (self.is_mamba and self.cache_params is not None) or \
                    (not self.is_mamba and self.past_key_values is not None)

        if self._cached_next_logits_gpu is not None and has_cache:
            probs_gpu = torch.softmax(self._cached_next_logits_gpu.float(), dim=-1)
            probs = probs_gpu.squeeze().cpu().numpy()  # convert ONCE
            return self._probs_to_freqs(probs)

        # SLOW PATH – initial warmup to feed full context
        context = self.context_tokens[-self.MAX_CONTEXT_FEED:]
        input_ids = torch.tensor([context], device=self.device)
        
        
        with torch.no_grad():
          if self.is_mamba:
              # Reset Mamba State
              outputs = self.model(input_ids=input_ids, use_cache=True)
              self.cache_params = outputs.cache_params
              self.seq_len = len(context) # Sync sequence length
          else:
              # Reset Transformer State
              outputs = self.model(input_ids=input_ids, use_cache=True)
              self.past_key_values = outputs.past_key_values

          self._cached_next_logits_gpu = outputs.logits[:, -1, :]

        probs_gpu = torch.softmax(self._cached_next_logits_gpu.float(), dim=-1)
        probs = probs_gpu.squeeze().cpu().numpy()
        return self._probs_to_freqs(probs)

    def _probs_to_freqs(self, probs):
        """
        Convert probabilities to integer frequencies within max_total_freq.
        Ensures no zero frequencies and total frequency does not exceed max_total_freq.
        This ensures arithmetic coder gets valid frequency tables.
        """
        # Scale to integer frequencies 
        scaled = probs * self.max_total_freq
        int_freqs = scaled.astype(np.int32)
        np.maximum(int_freqs, 1, out=int_freqs)

        # Normalize sum 
        total = int_freqs.sum()
        if total > self.max_total_freq:
            factor = self.max_total_freq / total
            int_freqs = (int_freqs * factor).astype(np.int32)
            np.maximum(int_freqs, 1, out=int_freqs)

        # Critical Optimization: Convert to list before dict creation
        return Frequencies(dict(enumerate(int_freqs.tolist())))

def tokens_from_text_in_chunks(tokenizer, text: str, max_chunk_tokens: int = 512):
    """
    Yield token chunks from a long text, never exceeding max_chunk_tokens.
    This prevents context overflow in the LLM. (Max 2048 tokens for TinyLlama.)
    """
    # vocab_size = tokenizer.vocab_size if hasattr(tokenizer, "vocab_size") else tokenizer.model_max_length
    start = 0
    text_len = len(text)
    while start < text_len:
        chunk_text = text[start:start + max_chunk_tokens * 4]  # heuristic: 4 chars ~ 1 token
        tokens = tokenizer.encode(chunk_text, add_special_tokens=False)

        max_vocab_id = tokenizer.vocab_size - 1
        unk_token_id = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else 0
        cleaned_tokens = []
        for token_id in tokens:
            if token_id >= max_vocab_id:
                # If the token is out-of-bounds (like 50266), replace it with UNK (0)
                cleaned_tokens.append(unk_token_id)
            else:
                cleaned_tokens.append(token_id)
        tokens = cleaned_tokens

        if len(tokens) > max_chunk_tokens:
            tokens = tokens[:max_chunk_tokens]
        yield tokens
        start += len(chunk_text)

def text_from_tokens(token_ids: List[int], tokenizer) -> str:
    # 1. Remove special tokens safely (BOS, EOS, UNK, PAD, system tokens)
    cleaned = [tid for tid in token_ids if tid not in tokenizer.all_special_ids]

    # 2. Decode while skipping any remaining special tokens
    return tokenizer.decode(cleaned, skip_special_tokens=True)

def compress_text_to_bitarray(
    tokenizer,
    model,
    device: torch.device,
    text: str,
    params: AECParams,
    freq_model_factory,
    chunk_size: int,
    metrics_log = None
    ) -> BitArray:
    
    """
    Compress text using ArithmeticEncoder and freq_model_factory to create a fresh freq_model
    For each token chunk: text → tokens → arithmetic encoder → bitarray
    Returns BitArray which uses SCL's BitArray for bit storage.
    """
    full_bits = BitArray()

    # Tokenize and yield chunks
    for token_chunk in tokens_from_text_in_chunks(tokenizer, text, max_chunk_tokens=chunk_size):
        freq_model_enc = freq_model_factory(params, metrics_log)
        encoder = ArithmeticEncoder(params, freq_model_enc)
        block = DataBlock(token_chunk)
        encoded_bits = encoder.encode_block(block)
        full_bits.extend(encoded_bits)

    return full_bits

def decompress_bitarray_to_text(
    tokenizer, 
    model, 
    device: torch.device, 
    encoded_bitarray: BitArray, 
    params: AECParams, 
    freq_model_factory,
    ) -> str:
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

    return text_from_tokens(decoded_tokens, tokenizer)

# -------------------------
# CLI / demo
# -------------------------
def main_demo():
    """
    Runs compression/decompression pipeline demo.
    Compresses input text file (or demo text) and decompresses to verify correctness.
    """
    # Parse command line arguments
    import argparse
    p = argparse.ArgumentParser(description="LLM-based compression demo using SCL arithmetic coder (refactored).")
    p.add_argument("--model", type=str, default="tinyllama", choices=["tinyllama", "llama3", "mamba", "mamba1.4b"], help="Model backend")
    p.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32"], help="Numeric precision")
    p.add_argument("--hf-token", type=str, default=None, help="HuggingFace token (optional)")
    p.add_argument("--infile", type=str, default=None, help="Input text file (if omitted, demo sentence used)")
    p.add_argument("--outfile", type=str, default="out.sclbit", help="Output compressed file")
    p.add_argument("--chunk-size", type=int, default=512, help="Token chunk size")
    p.add_argument("--context-size", type=int, default=512, help="Context size fed to LLM (<= model max)")
    p.add_argument("--top-k", type=int, default=1024, help="Top-K tokens to convert to frequency table (performance)")
    p.add_argument("--precision", type=int, default=32, help="Arithmetic coder precision (bits)")

    args = p.parse_args()

    # Load model and tokenizer
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    print(f"\n=== Loading Model: {args.model} (dtype={args.dtype}) ===")
    tokenizer, model, device = load_tokenizer_and_model(
        model_choice=args.model,
        dtype=dtype,
        hf_token=args.hf_token
    )
    
    # Load input text
    if args.infile:
        with open(args.infile, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        text = "The weather today is sunny and pleasant. I expect light winds and warm sunshine."


    params = AECParams(DATA_BLOCK_SIZE_BITS=32, PRECISION=args.precision)

    # Frequency model factory to create encoder/decoder models (fresh instances)
    def freq_model_factory(p, log=None):
        return LLMFreqModel(
            tokenizer=tokenizer,
            model=model,
            device=device,
            params=p,
            max_context_feed=args.context_size,
            top_k=args.top_k,
            metrics_log=log
        )

    print("=== Compressing ===")
    start_time = time.time()
    logs = MetricsLog()

    encoded_bitarray = compress_text_to_bitarray(
        tokenizer,
        model,
        device,
        text,
        params,
        freq_model_factory,
        chunk_size=args.chunk_size,
        metrics_log=logs
    )
    end_time = time.time()
    compression_time = end_time - start_time
    print(f"Compression time: {compression_time:.3f} sec")
    
    generate_final_report(text, encoded_bitarray, logs, compression_time)

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
    recovered_text = decompress_bitarray_to_text(
        tokenizer,
        model,
        device,
        loaded,
        params,
        freq_model_factory,
    )
    end_dec = time.time()
    decompression_time = end_dec - start_dec

    print("\nRecovered text snippet:", recovered_text[:200])
    print("\nMatch exact:", recovered_text == text)
    print(f"\nDecompression time: {decompression_time:.3f} sec")

if __name__ == "__main__":
    main_demo()