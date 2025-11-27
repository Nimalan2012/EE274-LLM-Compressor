import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# -----------------------
# Device selection (macOS)
# -----------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Silicon MPS backend")
else:
    device = torch.device("cpu")
    print("Using CPU")

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float32,     # macOS CPU cannot use float16
    low_cpu_mem_usage=True
).to(device)

model.eval()
torch.set_grad_enabled(False)

def get_next_token_probs(text):
    # Tokenize
    ids = tokenizer(text, return_tensors="pt").input_ids.to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(ids)
        logits = outputs.logits

    # Take last-token logits
    next_logits = logits[0, -1]

    # Convert to probabilities
    probs = torch.softmax(next_logits, dim=-1)

    return probs.cpu()  # return to CPU for safety

if __name__ == "__main__":
    test_text = "The weather today is"
    probs = get_next_token_probs(test_text)

    top_k = 10
    values, indices = probs.topk(top_k)

    print("\nTop next-token predictions:")
    for v, idx in zip(values, indices):
        print(f"{tokenizer.decode(idx.item())!r}: {v.item():.5f}")