"""CPU vs GPU vs Neural Engine: the same Core ML model on each part of an Apple Silicon chip.

From the Edge AI Lab video "Why Every Chip Suddenly Has an NPU" (episode 16).

    python ane_bench.py                  # every workload
    python ane_bench.py conv matvec      # just these
    python ane_bench.py --runs 50        # more runs per measurement (median is reported)

Workloads (float16, random weights, built directly with coremltools' MIL builder, no PyTorch):
  conv      12 x 3x3 convolutions, 128 channels, 112x112 image: compute-bound, like a vision model
  matvec    8 x 4096x4096 linear layers, one token: memory-bound, like an LLM writing one word
  prefill   the same weights, 256 tokens at once: like an LLM reading a prompt
  cumsum    the conv model with one cumsum layer in the middle (not supported on the Neural Engine)
  erf       the conv model with one erf layer in the middle (supported: the control for cumsum)

For each one it prints the median time on CPU only, CPU + GPU and CPU + Neural Engine, the
throughput that implies, and where Core ML actually planned each layer (MLComputePlan).
"""

import argparse
import os
import shutil
import statistics
import time
from pathlib import Path

import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types

HERE = Path(__file__).resolve().parent
MODELS = HERE / "models"          # built models are cached here (git-ignored)
UNITS = {"CPU": ct.ComputeUnit.CPU_ONLY,
         "GPU": ct.ComputeUnit.CPU_AND_GPU,
         "NeuralEngine": ct.ComputeUnit.CPU_AND_NE}
TARGET = ct.target.macOS15


def conv_stack(C=128, H=112, L=12, middle=None, seed=0):
    rng = np.random.default_rng(seed)

    @mb.program(input_specs=[mb.TensorSpec((1, C, H, H), types.fp16)], opset_version=TARGET)
    def prog(x):
        for i in range(L):
            w = (rng.standard_normal((C, C, 3, 3)) * 0.02).astype(np.float16)
            x = mb.relu(x=mb.conv(x=x, weight=w, pad_type="same"))
            if middle and i == L // 2 - 1:
                x = mb.cumsum(x=x, axis=3) if middle == "cumsum" else mb.erf(x=x)
        return x

    inp = {"x": rng.standard_normal((1, C, H, H)).astype(np.float16)}
    return prog, inp, L * C * C * 9 * H * H, L * C * C * 9 * 2


def linear_stack(D=4096, L=8, T=1, seed=0):
    rng = np.random.default_rng(seed)

    @mb.program(input_specs=[mb.TensorSpec((T, D), types.fp16)], opset_version=TARGET)
    def prog(x):
        for _ in range(L):
            w = (rng.standard_normal((D, D)) * 0.01).astype(np.float16)
            x = mb.relu(x=mb.linear(x=x, weight=w))
        return x

    inp = {"x": rng.standard_normal((T, D)).astype(np.float16)}
    return prog, inp, L * D * D * T, L * D * D * 2


WORKLOADS = {
    "conv": lambda: conv_stack(),
    "matvec": lambda: linear_stack(T=1),
    "prefill": lambda: linear_stack(T=256),
    "cumsum": lambda: conv_stack(middle="cumsum"),
    "erf": lambda: conv_stack(middle="erf"),
}


def build(name):
    """Convert and compile a workload once. Returns (package path, compiled path, inputs, MACs, weight bytes)."""
    prog, inp, macs, wbytes = WORKLOADS[name]()
    MODELS.mkdir(exist_ok=True)
    pkg, compiled = MODELS / f"{name}.mlpackage", MODELS / f"{name}.mlmodelc"
    model = ct.convert(prog, convert_to="mlprogram", compute_precision=ct.precision.FLOAT16,
                       minimum_deployment_target=TARGET)
    shutil.rmtree(pkg, ignore_errors=True)
    shutil.rmtree(compiled, ignore_errors=True)
    model.save(str(pkg))
    ct.models.utils.compile_model(str(pkg), str(compiled))
    return pkg, compiled, inp, macs, wbytes


def plan(compiled, unit):
    """{"op->device": count} from MLComputePlan (macOS 14.4+), or a short note if it isn't available."""
    try:
        from coremltools.models.compute_plan import MLComputePlan
        cp = MLComputePlan.load_from_path(path=str(compiled), compute_units=unit)
        counts = {}
        for op in cp.model_structure.program.functions["main"].block.operations:
            usage = cp.get_compute_device_usage_for_mlprogram_operation(op)
            if usage is None:
                continue
            device = type(usage.preferred_compute_device).__name__.replace("ML", "").replace("ComputeDevice", "")
            key = f"{op.operator_name.split('.')[-1]}->{device}"
            counts[key] = counts.get(key, 0) + 1
        return ", ".join(f"{k} x{v}" for k, v in counts.items())
    except Exception as e:  # older macOS / coremltools
        return f"(plan unavailable: {type(e).__name__})"


def bench(name, runs):
    pkg, compiled, inp, macs, wbytes = build(name)
    print(f"\n== {name}: {macs / 1e9:.2f} G multiply-adds, {wbytes / 1e6:.1f} MB of weights")
    for label, unit in UNITS.items():
        model = ct.models.MLModel(str(pkg), compute_units=unit)
        for _ in range(5):
            model.predict(inp)                       # warm-up (first runs include loading)
        times = []
        for _ in range(runs):
            t = time.perf_counter()
            model.predict(inp)
            times.append((time.perf_counter() - t) * 1e3)
        ms = statistics.median(times)
        print(f"  {label:13s} {ms:8.2f} ms   {2 * macs / ms / 1e9:7.2f} TOPS-equivalent   "
              f"{wbytes / ms / 1e6:6.1f} GB/s of weights")
        print(f"     plan: {plan(compiled, unit)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("workloads", nargs="*", metavar="workload",
                    help=f"any of: {', '.join(WORKLOADS)} (default: all)")
    ap.add_argument("--runs", type=int, default=30, help="timed runs per measurement (default 30)")
    args = ap.parse_args()
    unknown = [w for w in args.workloads if w not in WORKLOADS]
    if unknown:
        ap.error(f"unknown workload {', '.join(unknown)} (choose from {', '.join(WORKLOADS)})")
    os.environ.setdefault("TMPDIR", str(MODELS))     # Core ML's compiler writes temp files here
    for name in args.workloads or WORKLOADS:
        bench(name, args.runs)


if __name__ == "__main__":
    main()
