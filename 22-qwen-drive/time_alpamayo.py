"""Time Alpamayo 2 Edge on its published navigation example (run on the Colab VM, in venv-a2e).

    /content/venv-a2e/bin/python /content/time_alpamayo.py
    /content/venv-a2e/bin/python /content/time_alpamayo.py --attn flash_attention_2

Same protocol as time_qwen.py: one warm-up, then --repeats timed runs of one reasoning + one
trajectory (CUDA synchronised), plus the reasoning text and its token count. Plain PyTorch BF16
on whatever GPU this is: not NVIDIA's Thor recipe (FP8, TensorRT), so not their in-car latency.
Writes /content/out/alpamayo/<attn>.json.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from alpamayo2_edge.checkpoint import load_model
from alpamayo2_edge.helper import prepare_model_inputs, to_device
from alpamayo2_edge.load_physical_aiavdataset import load_physical_aiavdataset

WEIGHTS = "/content/weights/Alpamayo2-Edge"
OUT = Path("/content/out/alpamayo")
CLIP, T0_US, NAV = "ea7bbd31-b7a5-4972-8dbd-7089e6b53de4", 4_000_000, "Turn right in 30m"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attn", default="sdpa", choices=["sdpa", "flash_attention_2"])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--accel", choices=["compile", "tuned"], default=None,
                        help="NVIDIA's torch.compile + CUDA graphs path (needs flash-attn)")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    sample = load_physical_aiavdataset(CLIP, t0_us=T0_US, nav_text=NAV)
    load_start = time.perf_counter()
    model, _ = load_model(WEIGHTS, device="cuda", attn_implementation=args.attn)
    load_s = time.perf_counter() - load_start
    inputs = to_device(prepare_model_inputs(sample, model.config, model.tokenizer), device="cuda")
    accel = None
    if args.accel:
        from alpamayo2_edge.accel import AccelConfig, enable_acceleration

        accel = enable_acceleration(model, AccelConfig(mode=args.accel))
        compile_start = time.perf_counter()
        accel.warmup(inputs)
        print(f"compiled in {time.perf_counter() - compile_start:.0f} s", flush=True)

    def run(seed):
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            return model.sample_trajectories_from_data(
                inputs, num_traj_sets=1, num_traj_samples=1, max_generation_length=256,
                diffusion_kwargs={"inference_step": 10,
                                  "generator": torch.Generator(device="cuda").manual_seed(seed)},
                return_extra=True,
            )

    torch.cuda.reset_peak_memory_stats()
    run(42)  # warm-up
    times, runs = [], []
    for seed in range(43, 43 + args.repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        xyz, _, _, extra = run(seed)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
        cot = str(extra["cot"].reshape(-1)[0])
        runs.append({
            "seed": seed,
            "ms": round(times[-1], 1),
            "reasoning": cot,
            "reasoning_tokens": len(model.tokenizer(cot).input_ids) if cot else 0,
            "endpoint": [round(float(v), 2) for v in xyz.reshape(-1, 3)[-1]],
        })
        print(f"seed {seed}  {times[-1]:7.1f} ms  tokens {runs[-1]['reasoning_tokens']:3d}  {cot}", flush=True)

    report = {
        "model": "nvidia/Alpamayo2-Edge",
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "attn": args.attn,
        "accel": args.accel,
        "accel_describe": str(accel.describe()) if accel else None,
        "clip": CLIP, "t0_us": T0_US, "nav_text": NAV,
        "load_s": round(load_s, 1),
        "median_ms": round(statistics.median(times), 1),
        "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "runs": runs,
    }
    (OUT / f"{args.attn}{'_' + args.accel if args.accel else ''}.json").write_text(json.dumps(report, indent=2))
    print(f"median {report['median_ms']} ms, load {report['load_s']} s, peak {report['peak_gib']} GiB")


if __name__ == "__main__":
    main()
