"""Time Qwen-Drive-1.0 on the four bundled demo scenes (run on the Colab VM, in venv-qwen).

    /content/venv-qwen/bin/python /content/time_qwen.py --planner planner-sft --modes direct reasoning
    /content/venv-qwen/bin/python /content/time_qwen.py --planner planner-rl --modes reasoning

For each scene and mode: one warm-up run, then --repeats timed runs (CUDA synchronised), the
reasoning text and its token count, ADE/FDE against the logged future, and a summary plot of the
last run. One sample per run is what a car would use; --num-samples 6 times the best-of-6 setup.
Writes /content/out/qwen/<planner>.json and <planner>_<scene>_<mode>.png.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path("/content/qwen-drive")
WEIGHTS = Path("/content/weights/Qwen-Drive-1.0-4B")
OUT = Path("/content/out/qwen")
sys.path.insert(0, str(REPO / "src"))

from qwen_drive import InferenceMode, QwenDriveForPlanning  # noqa: E402
from qwen_drive.benchmarks import read_scene_file  # noqa: E402
from qwen_drive.images import ImageArchive  # noqa: E402
from qwen_drive.visualize import plot_scene_summary  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

MODES = {"direct": InferenceMode.DIRECT_PLANNING, "reasoning": InferenceMode.REASONING_PLANNING}


def timed(fn):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    return result, (time.perf_counter() - start) * 1000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--planner", default="planner-sft", choices=["planner-sft", "planner-rl"])
    parser.add_argument("--modes", nargs="+", default=["direct", "reasoning"], choices=list(MODES))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--num-samples", type=int, default=1)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    load_start = time.perf_counter()
    model = QwenDriveForPlanning.from_pretrained(
        str(WEIGHTS),
        planner=str(WEIGHTS / args.planner),
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to("cuda").eval()
    load_s = time.perf_counter() - load_start
    tokenizer = AutoTokenizer.from_pretrained(str(WEIGHTS))

    archive = ImageArchive.open(str(REPO / "data/demo/frames.parquet"))
    samples = list(read_scene_file(
        str(REPO / "data/demo/planning_scenes.jsonl"),
        image_archive=archive,
        num_history_points=model.config.num_history_points,
    ))

    report = {
        "planner": args.planner,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "num_samples": args.num_samples,
        "repeats": args.repeats,
        "load_s": round(load_s, 1),
        "runs": [],
    }
    torch.cuda.reset_peak_memory_stats()
    for index, sample in enumerate(samples):
        for mode_name in args.modes:
            run = lambda: model.run(MODES[mode_name], scene=sample.scene, num_samples=args.num_samples)
            run()  # warm-up: kernels, allocator, caches
            times, result = [], None
            for _ in range(args.repeats):
                result, ms = timed(run)
                times.append(ms)
            entry = {
                "scene": index,
                "token": sample.token,
                "mode": mode_name,
                "ms": [round(t, 1) for t in times],
                "median_ms": round(statistics.median(times), 1),
                "reasoning": result.reasoning,
                "reasoning_tokens": len(tokenizer(result.reasoning).input_ids) if result.reasoning else 0,
                "endpoint": np.round(result.trajectory[-1], 2).tolist(),
            }
            if sample.future_trajectory is not None:
                error = np.linalg.norm(
                    result.trajectory[:, :2] - sample.future_trajectory[: len(result.trajectory), :2], axis=-1
                )
                entry["ade_m"], entry["fde_m"] = round(float(error.mean()), 3), round(float(error[-1]), 3)
            report["runs"].append(entry)
            print(f"scene {index} {mode_name:9s} median {entry['median_ms']:7.1f} ms  "
                  f"tokens {entry['reasoning_tokens']:3d}  {entry.get('ade_m', '')}  {result.reasoning or ''}",
                  flush=True)
            plot_scene_summary(
                sample.scene, result.trajectories, history=sample.scene.history,
                ground_truth=sample.future_trajectory, reasoning=result.reasoning,
                title=f"Qwen-Drive {args.planner} · {mode_name} · scene {index}",
                output=OUT / f"{args.planner}_{index}_{mode_name}.png",
            )
    report["peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    (OUT / f"{args.planner}.json").write_text(json.dumps(report, indent=2))
    print(f"load {report['load_s']} s, peak {report['peak_gib']} GiB -> {OUT / (args.planner + '.json')}")


if __name__ == "__main__":
    main()
