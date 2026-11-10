# 17 · NVIDIA Squeezed a 74B AI Into 2 Bits. So I Tested It.

The test from the video: one model family (Qwen3 4B, 8B and 14B) at different bit levels, run with
llama.cpp on a MacBook, to see whether a **bigger model squashed to ~2.7 bits** beats a **smaller model
at 4 or 8 bits** in the same amount of memory. Each model gets a speed test, a perplexity test and
400 HellaSwag questions.

## Run it

Needs macOS or Linux, [llama.cpp](https://github.com/ggml-org/llama.cpp) (tested with 0.9.0 from
Homebrew), the Hugging Face CLI (`hf`), about 35 GB of free disk and 16 GB of RAM for the 14B 4-bit model.

```bash
brew install llama.cpp
pip install -U huggingface_hub          # gives you the hf command
mkdir -p data && cd data
curl -LO https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip && unzip wikitext-2-raw-v1.zip
curl -LO https://raw.githubusercontent.com/klosax/hellaswag_text_data/main/hellaswag_val_full.txt
cd ..
./run.sh                                 # all 7 models, about 3 hours on an M5
./run.sh Qwen_Qwen3-8B-Q4_K_M.gguf       # just one
```

It downloads each model into `models/` and writes `logs/<model>.{bench,ppl,hs}.txt`. Models and
finished logs are skipped on a re-run. Delete `models/` afterwards: it's about 35 GB.

## My results (MacBook, Apple M5, 16 GB, macOS 26.5.2, llama.cpp 0.9.0, Metal)

| Model | Bits/weight (format) | File | Perplexity ↓ | HellaSwag | Generate tok/s |
|---|---|---|---|---|---|
| 4B | ~4.8 (Q4_K_M) | 2.32 GiB | 14.85 | 65.25% | 41.2 |
| 8B | ~2.7 (IQ2_M) | 2.84 GiB | 11.59 | 66.75% | 31.2 |
| 4B | 8.5 (Q8_0) | 3.98 GiB | 14.40 | 66.00% | 27.4 |
| 8B | ~4.8 (Q4_K_M) | 4.68 GiB | 10.44 | 73.25% | 23.6 |
| 14B | ~2.7 (IQ2_M) | 4.95 GiB | 10.48 | 73.50% | 16.7 |
| 14B | ~3.4 (Q2_K) | 5.35 GiB | 10.61 | 72.75% | 18.8 |
| 14B | ~4.8 (Q4_K_M) | 8.38 GiB | 9.39 | 76.50% | 12.5 |

Perplexity: wikitext-2 test, 100 chunks of 512 tokens. HellaSwag: the first 400 validation tasks.
Speed: `llama-bench`, 128 generated tokens, mean of 3 runs. Raw output is in `logs/`.

At ~5 GB the 14B at ~2.7 bits **ties** the 8B at 4 bits on quality and is 29% slower. At ~3–4 GB the
8B at ~2.7 bits clearly beats the 4B at 8 bits.

## Caveats

- These are post-training quantizations (imatrix GGUFs). NVIDIA's Nemotron 3.5 Bolt was trained to
  work at 2 bits (quantization-aware distillation) and uses IQ2_XXS (2.06 bits); I didn't test Bolt.
- 400 HellaSwag questions give about ±4.5 points of uncertainty, so small gaps aren't real differences.
- One machine, one model family, no power measurements.

## Files

- `run.sh`: downloads each model, runs `llama-bench`, `llama-perplexity` and the HellaSwag test, logs everything.
- `logs/`: my raw output for all 7 models.

Third-party: Qwen3 models (Apache 2.0) as GGUFs by [bartowski](https://huggingface.co/bartowski);
[llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT); WikiText-2 (CC BY-SA 3.0);
[HellaSwag](https://rowanzellers.com/hellaswag/) (MIT) via klosax/hellaswag_text_data.
