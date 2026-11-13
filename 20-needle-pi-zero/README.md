# 20 · I Told My AI "Don't". It Did It Anyway (Needle 3 on a $15 Pi)

The benchmark from the video: Cactus Compute's tiny tool-calling models, **Needle 2** and **Needle 3**
(full 20 layers, and a 4-layer build), on a **Raspberry Pi Zero 2 W**. Each model gets the same 23 test
requests, 5 times each, with 5, 10, 25 and 50 tools to choose from: 460 requests per model. A request
only counts if the model picks the right function *and* fills in every value correctly. Some requests
are traps: missing information, impossible values, not a request at all, and "don't".

## Run it

On the Pi (Raspberry Pi OS / Debian 13, Python 3.13), for Needle 3:

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
cactus fetch                                   # downloads the engine (4.1.0)
# get the weights: needle3.cact from https://huggingface.co/Cactus-Compute/needle3
pip install "cactus-compute[train]==4.2.0"     # only for the next line, on a bigger machine (needs JAX)
cactus build --layers 4                        # writes the 4-layer needle3-4l.cact
./run.sh needle3.cact needle3
./run.sh needle3-4l.cact needle3-4l
```

For Needle 2, use a separate venv with `cactus-needle==2.0.12` instead of `cactus-compute` and its
`needle2.cact`. The adapter (`src/needle_bench/needle_adapter.py`) uses whichever package is installed.
Results land in `results/` (a summary CSV, metadata with the weights' sha256, and one JSON line per request).

The "don't" demo uses `run()`, the call an app uses to actually execute a function, with fake tools
that print instead of switching anything:

```bash
PYTHONPATH=src NEEDLE_WEIGHTS=needle3.cact python scripts/dont_demo.py
```

`scripts/dont_variants.py` tries 18 ways of saying "don't" at 5 and 50 tools. My Pi dropped off the
network halfway through it, so it's not in the video.

## My results (Pi Zero 2 W, 416 MB usable RAM, no cooling, 13 Nov 2026)

| Model | File | Exact calls | 5 / 10 / 25 / 50 tools | Median time | Engine RAM |
|---|---|---|---|---|---|
| Needle 2 (cactus-needle 2.0.12) | 13.7 MB | 64.1% | 65 / 70 / 65 / 57% | 9.1 s | 70 MB |
| Needle 3, 20 layers (cactus-compute 4.2.0) | 35.3 MB | 89.1% | 91 / 91 / 91 / 83% | 3.3 s | 114–130 MB |
| Needle 3, 4 layers | 15.4 MB | 63.0% | 70 / 74 / 61 / 48% | 0.86 s | 82–83 MB |

- Every model gave the same answer on all 5 repeats. The Needle 2 re-run matched my September run exactly.
- "Do not turn on the bedroom lights." → `control_lights(on, bedroom)`, 20 out of 20, on all three models.
  With `run()` it executes (`results/dont_demo_20261113.txt`).
- "Don't start the fan in the study." → Needle 2 turns the fan off; Needle 3 makes no call.
- The negation flag: on the Pi, the lights call comes back with no flag. On a Mac with the same weights
  and versions it comes back `negation: true` and the call is still returned (`results/dont_mac_20261113.txt`).
  In cactus-compute 4.2.0, `run()` only blocks values you never said; the negation check is in `extract()`.
- Confidence (full model): right calls median 0.77, wrong 0.49. Blocking under 0.5 keeps 90% of right
  calls and stops 60% of wrong ones. On the 4-layer model it's 0.26 vs 0.22: no signal.
- The 4-layer file is 15.4 MB rather than ~8 MB because each engram lookup table is 14.2M parameters
  (110,592 × 128); the full model has five, the 4-layer build keeps one.

## Caveats

- One Pi, one run per model (deterministic, so repeats don't add information), no cooling: the CPU
  reached 80–82 °C and the firmware reported frequency capping at some point (no under-voltage).
- 23 hand-written test cases in English; a different set of tools or requests will give different numbers.
- I haven't checked why Needle 2 slows down from 1.9 s to 10.8 s above five tools.

## Files

- `src/needle_bench/`: the benchmark (tools, test cases, scoring, hardware readings, the Needle adapter).
- `scripts/run_benchmark.py`, `run.sh`: the full run. `scripts/make_report.py`: charts from a results file.
- `scripts/dont_demo.py`, `scripts/dont_variants.py`: the "don't" tests with `run()`.
- `data/test_cases.json`, `config/benchmark.json`: the 23 cases and the run settings.
- `results/`: my raw results for all three models, plus the September Needle 2 run.

Third-party: Needle 2 and Needle 3 and the cactus packages are by [Cactus Compute](https://github.com/cactus-compute/cactus);
check their licence before redistributing the weights (not included here).
