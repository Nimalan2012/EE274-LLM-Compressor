from transformers import AutoTokenizer, AutoModelForCausalLM
from scl.utils.bitarray_utils import BitArray
from scl.compressors.arithmetic_coding import AECParams
from llm_compressor import LLMFreqModel, decompress_bitarray_to_text

# load model
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device_map="auto"
)

params = AECParams(DATA_BLOCK_SIZE_BITS=32, PRECISION=32)

def freq_model_factory(p):
    return LLMFreqModel(tokenizer, model, p)

# load file
with open("compressed.bin", "rb") as fin:
    raw = fin.read()

bit_length = int.from_bytes(raw[:4], "big")
payload = raw[4:]

bits = BitArray()
bits.frombytes(payload)
bits = bits[:bit_length]

print("bits:", bits)
print("type(bits):", type(bits))
print("bit_length:", bit_length)

# decompress
text = decompress_bitarray_to_text(bits, params, freq_model_factory)

print("\n=== DECODED TEXT ===\n")
print(text)
print("\n====================\n")