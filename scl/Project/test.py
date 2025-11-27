import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from scl.compressors.arithmetic_coding import AECParams, ArithmeticEncoder, ArithmeticDecoder  # adjust path if different
from scl.core.data_block import DataBlock
from scl.core.prob_dist import Frequencies
from scl.utils.bitarray_utils import BitArray, bitarray_to_uint, uint_to_bitarray
from scl.compressors.probability_models import FreqModelBase
import math
import copy
from typing import List, Dict

def main_decode():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", type=str, default="compressed.bin")
    args = parser.parse_args()

    # Load TinyLlama tokenizer + model
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device_map="auto",
    )

    params = AECParams(DATA_BLOCK_SIZE_BITS=32, PRECISION=32)

    def freq_model_factory(p):
        return LLMFreqModel(tokenizer, model, p)

    # === Read compressed file ===
    with open(args.infile, "rb") as fin:
        raw = fin.read()

    bit_length = int.from_bytes(raw[:4], "big")
    payload = raw[4:]

    loaded = BitArray()
    loaded.frombytes(payload)
    loaded = loaded[:bit_length]

    recovered_text = decompress_bitarray_to_text(loaded, params, freq_model_factory)

    print("\n=== DECODED TEXT ===\n")
    print(recovered_text)
    print("\n====================\n")


# >>> THIS MUST BE HERE <<<
if __name__ == "__main__":
    main_decode()