# 22 · Qwen-Drive: the car brain that never stops talking

The timing test from the video: Alibaba's **Qwen-Drive-1.0** and NVIDIA's **Alpamayo 2 Edge**, both
self-driving vision-language models, run on the same rented GPU (a Google Colab A100) with the code
their teams released. One plan per run, one warm-up, then timed runs, CUDA-synchronised.

## Run it

Needs a Google Colab plan with A100 access and the official [Colab CLI](https://github.com/googlecolab/google-colab-cli)
(`uv tool install google-colab-cli`; `colab usage` signs you in once). Alpamayo 2 Edge's weights and the
NVIDIA PhysicalAI-AV dataset are gated: accept both on Hugging Face first.

```bash
colab new -s qd --gpu A100
for f in setup_qwen.sh time_qwen.py profile_qwen.py setup_alpamayo.sh time_alpamayo.py; do
  colab upload -s qd $f /content/$f; done

# Qwen-Drive (public weights, ~13.7 GB)
colab ssh -s qd -- bash /content/setup_qwen.sh
colab ssh -s qd -- /content/venv-qwen/bin/python /content/time_qwen.py --planner planner-sft --modes direct reasoning
colab ssh -s qd -- /content/venv-qwen/bin/python /content/time_qwen.py --planner planner-rl --modes reasoning
colab ssh -s qd -- /content/venv-qwen/bin/python /content/profile_qwen.py

# Alpamayo 2 Edge (gated: log the VM in yourself first)
colab ssh -s qd -- hf auth login
colab ssh -s qd -- bash /content/setup_alpamayo.sh
colab ssh -s qd -- /content/venv-a2e/bin/python /content/time_alpamayo.py
colab ssh -s qd -- /content/venv-a2e/bin/python /content/time_alpamayo.py --attn flash_attention_2 --accel tuned --repeats 10

colab download -s qd /content/out/qwen/planner-sft.json planner-sft.json   # results are in /content/out
colab stop -s qd                                                             # an idle A100 still costs units
```

Each model gets its own virtual environment (they pin different `transformers` versions). The two
sessions in the video used about 3 Colab compute units in total.

## My results (Colab, NVIDIA A100-SXM4-40GB, torch 2.8.0+cu128, BF16)

| Model, code path | Median per plan | Reasoning tokens | Peak GPU memory |
|---|---|---|---|
| Qwen-Drive SFT planner, no reasoning, as released | 1.24 s | 0 | 12.3 GiB |
| Qwen-Drive SFT or RL planner, with reasoning, as released | 2.2–2.4 s | 10–13 | 11–12 GiB |
| Alpamayo 2 Edge, plain PyTorch (SDPA) | 1.25 s | 14–24 | 5.8 GiB |
| Alpamayo 2 Edge, `--accel tuned` (torch.compile + CUDA graphs, 114 s to compile once) | 0.275 s (0.198 s when it writes `<NOMINAL>`) | 5–31 | 5.8 GiB |

Profile of one Qwen-Drive direct plan (1.35 s): about 0.37 s resizing images on the CPU, 0.30 s reading
the scene (prefill), 0.66 s for the planner's 10 steps; about 29,000 kernel launches, and the GPU is
busy for only about 0.45 s. Raw numbers are in `results/`.

## Caveats

- Neither number is in-car latency. NVIDIA's 160–190 ms is on DRIVE AGX Thor with TensorRT and FP8;
  this is BF16 PyTorch on a data-centre GPU.
- The inputs differ: Qwen-Drive sees 3 cameras × 4 frames of its four bundled Waymo demo scenes,
  Alpamayo sees 6 cameras / 22 frames of NVIDIA's PhysicalAI-AV example clip.
- Qwen-Drive ships no accelerated path, and I didn't try compiling it, so its numbers are the code as
  released, not what the model could do.
- Four demo scenes and one example clip: timings only, not an accuracy test.

## Files

- `setup_qwen.sh`, `setup_alpamayo.sh`: install each model on the Colab VM in its own venv, pinned to the commits I used.
- `time_qwen.py`, `time_alpamayo.py`: the timing runs; they write JSON to `/content/out/`.
- `profile_qwen.py`: where one Qwen-Drive plan spends its time (cProfile and the torch profiler).
- `results/*.json`: my runs.

Third-party: [Qwen-Drive-1.0](https://github.com/QwenLM/Qwen-Drive-1.0) (code and weights, Apache 2.0),
[Alpamayo 2 Edge](https://github.com/NVlabs/alpamayo2-edge) (code Apache 2.0; weights
[OpenMDW-1.1](https://huggingface.co/nvidia/Alpamayo2-Edge)), NVIDIA
[PhysicalAI-Autonomous-Vehicles](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)
(gated, NVIDIA's terms). None of them are included here.
