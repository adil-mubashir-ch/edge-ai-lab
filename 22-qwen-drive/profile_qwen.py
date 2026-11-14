"""Where does one Qwen-Drive plan spend its time? (run on the Colab VM, in venv-qwen)

    /content/venv-qwen/bin/python /content/profile_qwen.py

Prints the top functions by cumulative time (cProfile) for one direct and one reasoning plan on demo
scene 0, after a warm-up, plus the torch profiler's top CUDA and CPU ops for the direct plan.
"""

import cProfile
import pstats
import sys

import torch

sys.argv = [sys.argv[0]]
sys.path.insert(0, "/content")
import time_qwen as t  # noqa: E402

model = t.QwenDriveForPlanning.from_pretrained(
    str(t.WEIGHTS), planner=str(t.WEIGHTS / "planner-sft"), dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
).to("cuda").eval()
archive = t.ImageArchive.open(str(t.REPO / "data/demo/frames.parquet"))
scene = next(iter(t.read_scene_file(
    str(t.REPO / "data/demo/planning_scenes.jsonl"), image_archive=archive,
    num_history_points=model.config.num_history_points,
))).scene

for name, mode in t.MODES.items():
    model.run(mode, scene=scene)  # warm-up
    profiler = cProfile.Profile()
    profiler.enable()
    model.run(mode, scene=scene)
    torch.cuda.synchronize()
    profiler.disable()
    print(f"\n===== {name}: cProfile, top 25 by cumulative time =====")
    pstats.Stats(profiler).sort_stats("cumulative").print_stats(25)

with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                        torch.profiler.ProfilerActivity.CUDA]) as prof:
    model.run(t.MODES["direct"], scene=scene)
    torch.cuda.synchronize()
print("\n===== direct: torch profiler, top 20 by CUDA time =====")
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
print("\n===== direct: torch profiler, top 15 by CPU time =====")
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))
