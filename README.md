# EE274-LLM-Compressor
This project demonstrates the implementation of LLM-based compression integrated with SCL's (Stanford Compression Library) arithmetic coder, for lossless data compression using language models. The primary script, llm_compressor.py, facilitates the compression and decompression of data. 

## Installation & Setup
For the best performance (and access to GPUs), we recommend running this project on **Google Colab**. Other supported runtimes are **Apple MPS** (macOS) and **CPU-only** for small tests.

- Google Colab (recommended)
    - Best for GPU access and quick setup.
- Apple MPS (macOS)
    - Supported via PyTorch’s MPS backend; performance and compatibility may vary by PyTorch version and macOS release.
- CPU-only
    - Works for demos and small models, but will be significantly slower for real workloads.

Note: Gated models (e.g., Llama-3) may require a Hugging Face token and substantial GPU memory.

### 1. Clone the Repository
Run the following commands in a Colab cell to clone the repo and navigate to the project folder:

```bash
!git clone https://github.com/Nimalan2012/EE274-LLM-Compressor.git
%cd EE274-LLM-Compressor/scl/Project
```
### 2. Install Dependencies
Once inside the project folder, install the required libraries. You only need to install bitarray if Colab is being used.

```bash
!pip install torch transformers numpy bitarray sentencepiece protobuf
```

### 3. Quick Start 
(Demo) To verify everything is working, run the script without arguments. This will compress a default short sentence using TinyLlama:

```bash
!python llm_compressor.py
```
Output: This will compress a sample sentence, save it to `out.sclbit`, and immediately decompress it to verify integrity.

Alternatively, compress-only (without decompression):
```bash
!python llm_compressor.py --mode compress
```

Or decompress-only (requires a previously compressed file):
```bash
!python llm_compressor.py --mode decompress
```

### 4. Compressing Your Own File

To compress a specific text file using the Mamba model:

```bash
python llm_compressor.py --model mamba --infile "news_transcript.txt" --outfile "news_compressed.sclbit" --dtype fp16
```

This reads `news_transcript.txt`, compresses it with the selected model, and writes the compressed binary to `news_compressed.sclbit`.

## Arguments & Configuration

You can tune the compressor with these command-line arguments:

| Argument | Default | Description |
| --- | ---: | --- |
| `--model` | `tinyllama` | Which model to use for compression. Options: `tinyllama` (1.1B transformer), `mamba` (130M state-space model), `mamba1.4b` (1.4B state-space model), `llama3` (gated Llama 3; requires HF token and large GPU memory). |
| `--infile` | `None` | Path to the input text file to compress. If omitted, the script uses a default demo sentence. |
| `--outfile` | `out.sclbit` | Output filename for the compressed binary. |
| `--dtype` | `fp16` | Model precision: `fp16` (recommended—faster, lower VRAM) or `fp32` (full precision). |
| `--context-size` | `512` | How many previous tokens the model can attend to (context window size). |
| `--precision` | `32` | Integer precision (bits) used by the SCL Arithmetic Encoder. |
| `--mode` | `all` | Execution mode: `compress` (compress only), `decompress` (decompress only), or `all` (both in sequence). |
| `--hf-token` | `None` | Hugging Face access token required for gated models (e.g., `llama3`). |
