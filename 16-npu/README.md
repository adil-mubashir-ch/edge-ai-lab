# 16 · Why Every Chip Suddenly Has an NPU

The benchmark from the video: one Core ML model, run on the **CPU**, the **GPU** and the **Neural
Engine** of an Apple Silicon Mac, plus which part of the chip Core ML actually put each layer on.

## Run it

Needs a Mac with Apple Silicon and Python 3 (tested with Python 3.12 and coremltools 9.0).

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python ane_bench.py              # all five workloads, about 2 minutes
.venv/bin/python ane_bench.py conv matvec  # just some
```

Close other apps first: a busy machine (a video render, a browser full of tabs) slows the CPU numbers
the most. Each time is the median of 30 runs after 5 warm-ups (`--runs` changes that).

| Workload | What it stands for |
|---|---|
| `conv` | 12 × 3×3 convolutions, 128 channels, 112×112: a vision model (lots of maths, small weights) |
| `matvec` | 8 × 4096×4096 linear layers, one token: an LLM writing one word (huge weights, used once each) |
| `prefill` | the same weights, 256 tokens at once: an LLM reading your prompt |
| `cumsum` | `conv` with one cumsum layer in the middle: a layer the Neural Engine can't run |
| `erf` | `conv` with one erf layer in the middle: the control (supported, stays on the Neural Engine) |

## My results (MacBook, Apple M5, 16 GB, macOS 26.5.2, coremltools 9.0)

| Workload | CPU | GPU | Neural Engine |
|---|---|---|---|
| conv | 14.9 ms | 4.9 ms | **2.06 ms** |
| matvec | 4.2 ms | **2.8–3.1 ms** | 13.3 ms |
| prefill | 19.8 ms | **6.4 ms** | 11.8 ms |
| cumsum | 16.0 ms | 5.0 ms | 5.03 ms (the cumsum ran on the CPU) |
| erf | 16.3 ms | 5.0 ms | 2.14 ms |

On `matvec` the weights moved at about 64 GB/s (CPU), 86–97 GB/s (GPU) and 20 GB/s (Neural Engine),
out of the M5's 153 GB/s.

**Post yours** in the comments under the video (chip, macOS version, the table): I'd love to see an
M1, an M4 Pro or an M5 Max next to this.

## Caveats

- Synthetic layers with random weights, float16 only. Real models (and int8) will behave differently.
- Core ML decides how to split and schedule the model. "Neural Engine" means `CPU_AND_NE`: Core ML may
  still put some layers on the CPU, which is exactly what the plan line shows.
- Apple doesn't document the Neural Engine's internals, so *why* `matvec` is slow there is a guess.
- `MLComputePlan` needs macOS 14.4 or later. On older systems the plan line just says it's unavailable.

## Files

- `ane_bench.py`: builds each workload with coremltools' MIL builder (no PyTorch needed), compiles
  it once into `models/` (git-ignored), and times it on each compute unit.
- `requirements.txt`: coremltools and numpy.
