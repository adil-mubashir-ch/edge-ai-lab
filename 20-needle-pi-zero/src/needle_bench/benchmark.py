from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .evaluator import evaluate_response
from .hardware import hardware_snapshot
from .needle_adapter import build_agent, complete, needle_version, weights_info
from .schemas import TestCase
from .tools import tools_for_count


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data" / "test_cases.json"

RESULTS_DIR = ROOT / "results"
RAW_DIR = RESULTS_DIR / "raw"

FIGURES_DIR = ROOT / "reports" / "figures"


# ---------------------------------------------------------------------------
# Test-case loading
# ---------------------------------------------------------------------------

def load_test_cases(path: Path = DATA_PATH) -> list[TestCase]:
    """
    Load benchmark test cases from JSON.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    return [TestCase(**item) for item in data]


# ---------------------------------------------------------------------------
# Hardware / process metrics
# ---------------------------------------------------------------------------

def rss_mb() -> float | None:
    """
    Return current Python process RSS in MB.

    This is separate from Needle's own peak_ram_mb metric.
    """
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1024**2

    except Exception:
        return None


def timed_complete(
    agent: Any,
    prompt: str,
    max_new_tokens: int,
) -> tuple[dict[str, Any], float, float | None, float | None]:
    """
    Execute Needle.complete() while measuring wall-clock latency
    and process RSS.

    Returns:
        response
        elapsed_seconds
        rss_before_mb
        rss_after_mb
    """
    before = rss_mb()

    t0 = time.perf_counter()

    response = complete(
        agent,
        prompt,
        max_new_tokens=max_new_tokens,
    )

    elapsed = time.perf_counter() - t0

    after = rss_mb()

    return response, elapsed, before, after


# ---------------------------------------------------------------------------
# Agent lifecycle helpers
# ---------------------------------------------------------------------------

def reset_agent(agent: Any) -> None:
    """
    Reset Needle state between benchmark samples when supported.
    """
    try:
        agent.reset()
    except Exception:
        pass


def close_agent(agent: Any) -> None:
    """
    Close the Needle agent/native resources when supported.
    """
    try:
        agent.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def run_benchmark(
    tool_counts: list[int],
    repetitions: int,
    warmups: int,
    seed: int,
    max_new_tokens: int,
    smoke: bool = False,
):
    """
    Run the controlled Needle tool-selection benchmark.

    The main experimental variable is tool catalogue size.

    For each tool count:
        - construct a fresh Needle agent
        - perform warmups
        - evaluate every test case
        - repeat each case the requested number of times
        - collect model, latency, memory, and hardware metrics
    """

    os.environ.setdefault("NEEDLE_TELEMETRY", "0")

    cases = load_test_cases()

    # ---------------------------------------------------------------
    # Smoke-test configuration
    # ---------------------------------------------------------------

    if smoke:
        smoke_ids = {
            "positive_lights",
            "positive_thermostat",
            "missing_lights",
            "irrelevant",
            "invalid_thermostat",
            "parallel",
        }

        cases = [
            case
            for case in cases
            if case.id in smoke_ids
        ]

        tool_counts = [5, 10]
        repetitions = 1
        warmups = 1

    # ---------------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------------

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),

        "needle_version": needle_version(),
        "weights": weights_info(),

        "seed": seed,

        # Needle.complete() does not expose a seed parameter.
        # The seed is therefore recorded for experiment bookkeeping,
        # but does not directly control model inference.
        "seed_applied_to_needle": False,

        "tool_counts": tool_counts,
        "repetitions": repetitions,
        "warmups": warmups,
        "max_new_tokens": max_new_tokens,

        "smoke": smoke,

        "num_test_cases": len(cases),
        "test_case_ids": [case.id for case in cases],

        "hardware": hardware_snapshot(),

        "notes": (
            "Benchmark uses Needle.complete() with explicit model weights. "
            "Only tool catalogue size is varied across experimental groups. "
            "Tool calls are evaluated without executing real-world actions."
        ),
    }

    rows: list[dict[str, Any]] = []

    # ---------------------------------------------------------------
    # Main experiment
    # ---------------------------------------------------------------

    for count in tool_counts:

        tools = tools_for_count(count)

        print()
        print("=" * 72)
        print(f"Tool catalogue: {count} tools")
        print(f"Cases:           {len(cases)}")
        print(f"Repetitions:     {repetitions}")
        print(f"Warmups:         {warmups}")
        print("=" * 72)

        agent = build_agent(tools)

        try:

            # -------------------------------------------------------
            # Warmup
            # -------------------------------------------------------

            for warmup_index in range(warmups):

                print(
                    f"Warmup {warmup_index + 1}/{warmups}...",
                    flush=True,
                )

                try:
                    complete(
                        agent,
                        "Dim the study lights to 30 percent.",
                        max_new_tokens=max_new_tokens,
                    )

                finally:
                    reset_agent(agent)

            # -------------------------------------------------------
            # Measured runs
            # -------------------------------------------------------

            total_runs = len(cases) * repetitions
            completed_runs = 0

            for case in cases:

                for rep in range(repetitions):

                    reset_agent(agent)

                    start_hw = hardware_snapshot()

                    response = None
                    latency = None
                    rss_before = None
                    rss_after = None
                    error = None

                    try:

                        (
                            response,
                            latency,
                            rss_before,
                            rss_after,
                        ) = timed_complete(
                            agent,
                            case.prompt,
                            max_new_tokens,
                        )

                    except Exception as exc:

                        error = repr(exc)

                        # Preserve a response-like structure so the
                        # benchmark can still record the failure.
                        response = {
                            "type": "error",
                            "success": False,
                            "error": str(exc),
                            "function_calls": [],
                        }

                        latency = None

                    end_hw = hardware_snapshot()

                    metrics = evaluate_response(
                        response,
                        case.expected_calls,
                    )

                    completed_runs += 1

                    # ------------------------------------------------
                    # Determine process RSS peak for this sample
                    # ------------------------------------------------

                    rss_values = [
                        value
                        for value in (
                            rss_before,
                            rss_after,
                        )
                        if value is not None
                    ]

                    rss_peak = (
                        max(rss_values)
                        if rss_values
                        else None
                    )

                    # ------------------------------------------------
                    # Store complete result row
                    # ------------------------------------------------

                    row = {
                        "timestamp_utc": (
                            datetime.now(timezone.utc).isoformat()
                        ),

                        # Experimental variables
                        "tool_count": count,
                        "num_tools_supplied": len(tools),

                        "case_id": case.id,
                        "category": case.category,
                        "critical": case.critical,
                        "description": case.description,
                        "prompt": case.prompt,

                        "repetition": rep,

                        # Timing
                        "latency_ms": (
                            latency * 1000
                            if latency is not None
                            else None
                        ),

                        # Python process memory
                        "rss_mb_before": rss_before,
                        "rss_mb_after": rss_after,
                        "rss_mb_peak": rss_peak,

                        # Hardware state
                        "temperature_start_c": (
                            start_hw.get("cpu_temperature_c")
                        ),
                        "temperature_end_c": (
                            end_hw.get("cpu_temperature_c")
                        ),

                        # Benchmark metrics
                        **metrics,

                        # Python/runtime error, if any
                        "benchmark_error": error,
                    }

                    rows.append(row)

                    print(
                        f"[{completed_runs:>3}/{total_runs}] "
                        f"{case.id:<28} "
                        f"rep={rep + 1} "
                        f"exact={metrics['exact_call_correct']} "
                        f"tool={metrics['tool_correct']} "
                        f"args={metrics['arguments_correct']} "
                        f"latency="
                        f"{row['latency_ms']:.1f}ms"
                        if row["latency_ms"] is not None
                        else
                        f"[{completed_runs:>3}/{total_runs}] "
                        f"{case.id:<28} "
                        f"rep={rep + 1} "
                        f"ERROR"
                    )

        finally:

            # Always release the native Needle resources before
            # constructing the next tool-count configuration.
            close_agent(agent)

    return rows, metadata


# ---------------------------------------------------------------------------
# Result saving
# ---------------------------------------------------------------------------

def save_results(
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
    label: str | None = None,
):
    """
    Save:
        - raw JSONL
        - metadata JSON
        - summary CSV
    """

    RESULTS_DIR.mkdir(
        exist_ok=True,
        parents=True,
    )

    RAW_DIR.mkdir(
        exist_ok=True,
        parents=True,
    )

    stamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    if label:
        stamp = f"{label}_{stamp}"

    raw_path = (
        RAW_DIR /
        f"results_{stamp}.jsonl"
    )

    with raw_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for row in rows:

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    metadata_path = (
        RESULTS_DIR /
        f"metadata_{stamp}.json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    csv_path = (
        RESULTS_DIR /
        f"summary_{stamp}.csv"
    )

    pd.DataFrame(rows).to_csv(
        csv_path,
        index=False,
    )

    return raw_path, csv_path, metadata_path


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Needle 2 tool-calling reliability "
            "as the tool catalogue grows."
        )
    )

    parser.add_argument(
        "--tool-counts",
        nargs="+",
        type=int,
        default=[5, 10, 25, 50],
        help="Tool catalogue sizes to benchmark.",
    )

    parser.add_argument(
        "--repetitions",
        type=int,
        default=3,
        help="Measured repetitions per test case.",
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Warmup runs per tool-count configuration.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Experiment bookkeeping seed.",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum Needle output tokens.",
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run the small smoke-test configuration.",
    )

    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Model label (e.g. needle3-20l) added to every row and result file name.",
    )

    args = parser.parse_args()

    rows, metadata = run_benchmark(
        tool_counts=args.tool_counts,
        repetitions=args.repetitions,
        warmups=args.warmup,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        smoke=args.smoke,
    )

    metadata["label"] = args.label
    for row in rows:
        row["model_label"] = args.label

    raw_path, csv_path, metadata_path = save_results(
        rows,
        metadata,
        args.label,
    )

    # ---------------------------------------------------------------
    # Charts
    # ---------------------------------------------------------------

    try:

        from .charts import generate_all_charts

        output_dir = (
            FIGURES_DIR /
            csv_path.stem
        )

        generate_all_charts(
            csv_path,
            output_dir,
        )

    except Exception as exc:

        output_dir = None

        print()
        print(
            f"WARNING: chart generation failed: {exc}"
        )

    # ---------------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------------

    print()
    print("=" * 72)
    print("BENCHMARK COMPLETE")
    print("=" * 72)

    print(
        f"Measured calls: {len(rows)}"
    )

    print(
        f"Raw JSONL:      {raw_path}"
    )

    print(
        f"CSV:             {csv_path}"
    )

    print(
        f"Metadata:        {metadata_path}"
    )

    if output_dir is not None:
        print(
            f"Charts:          {output_dir}"
        )

    print("=" * 72)


if __name__ == "__main__":
    main()