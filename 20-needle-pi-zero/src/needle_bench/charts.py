from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save(fig, path: Path) -> None:
    """Save and close a matplotlib figure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def generate_all_charts(
    csv_path: Path,
    out_dir: Path,
) -> None:
    """
    Generate all benchmark figures from the summary CSV.
    """

    df = pd.read_csv(csv_path)

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ================================================================
    # 01 — Accuracy vs tool count
    # ================================================================

    g = (
        df.groupby("tool_count")
        .agg(
            exact=("exact_call_correct", "mean"),
            tool=("tool_correct", "mean"),
            args=("arguments_correct", "mean"),
        )
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        g.tool_count,
        g.exact * 100,
        marker="o",
        label="Exact call",
    )

    ax.plot(
        g.tool_count,
        g.tool * 100,
        marker="o",
        label="Tool selection",
    )

    ax.plot(
        g.tool_count,
        g.args * 100,
        marker="o",
        label="Arguments",
    )

    ax.set(
        xlabel="Available tools",
        ylabel="Accuracy (%)",
        title="Needle 2 tool-calling accuracy vs tool count",
    )

    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.25)
    ax.legend()

    save(
        fig,
        out_dir / "01_accuracy_vs_tools.png",
    )

    # ================================================================
    # 02 — Mean / median latency
    # ================================================================

    lat = (
        df.groupby("tool_count")["latency_ms"]
        .agg(["mean", "median"])
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        lat.tool_count,
        lat["mean"],
        marker="o",
        label="Mean",
    )

    ax.plot(
        lat.tool_count,
        lat["median"],
        marker="o",
        label="Median",
    )

    ax.set(
        xlabel="Available tools",
        ylabel="Latency (ms)",
        title="Inference latency vs tool count",
    )

    ax.grid(True, alpha=0.25)
    ax.legend()

    save(
        fig,
        out_dir / "02_latency_vs_tools.png",
    )

    # ================================================================
    # 03 — Latency percentiles
    # ================================================================

    q = (
        df.groupby("tool_count")["latency_ms"]
        .quantile([0.50, 0.95])
        .unstack()
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    if 0.50 in q.columns:
        ax.plot(
            q.index,
            q[0.50],
            marker="o",
            label="p50",
        )

    if 0.95 in q.columns:
        ax.plot(
            q.index,
            q[0.95],
            marker="o",
            label="p95",
        )

    ax.set(
        xlabel="Available tools",
        ylabel="Latency (ms)",
        title="Latency percentiles",
    )

    ax.grid(True, alpha=0.25)
    ax.legend()

    save(
        fig,
        out_dir / "03_latency_percentiles.png",
    )

    # ================================================================
    # 04 — Process memory vs tool count
    # ================================================================

    if "rss_mb_peak" in df.columns:

        mem = (
            df.groupby("tool_count")["rss_mb_peak"]
            .mean()
            .dropna()
        )

        if not mem.empty:

            fig, ax = plt.subplots(figsize=(8, 5))

            ax.plot(
                mem.index,
                mem.values,
                marker="o",
            )

            ax.set(
                xlabel="Available tools",
                ylabel="Process RSS (MB)",
                title="Peak process RSS vs tool count",
            )

            ax.grid(True, alpha=0.25)

            save(
                fig,
                out_dir / "04_memory_vs_tools.png",
            )

    # ================================================================
    # 05 — Accuracy by category
    # ================================================================

    cat = (
        df.groupby("category")["exact_call_correct"]
        .mean()
        .sort_values(ascending=False)
        * 100
    )

    if not cat.empty:

        fig, ax = plt.subplots(figsize=(9, 5))

        cat.plot(
            kind="bar",
            ax=ax,
        )

        ax.set(
            xlabel="",
            ylabel="Exact-call accuracy (%)",
            title="Accuracy by test category",
        )

        ax.set_ylim(0, 105)
        ax.tick_params(
            axis="x",
            rotation=35,
        )

        save(
            fig,
            out_dir / "05_accuracy_by_category.png",
        )

    # ================================================================
    # 06 — Confidence distribution
    # ================================================================

    if "confidence" in df.columns:

        confidence = pd.to_numeric(
            df["confidence"],
            errors="coerce",
        ).dropna()

        if not confidence.empty:

            fig, ax = plt.subplots(figsize=(8, 5))

            confidence.plot(
                kind="hist",
                bins=15,
                ax=ax,
            )

            ax.set(
                xlabel="Needle confidence",
                title="Confidence distribution",
            )

            save(
                fig,
                out_dir / "06_confidence_distribution.png",
            )

    # ================================================================
    # 07 — Latency distribution
    # ================================================================

    fig, ax = plt.subplots(figsize=(9, 5))

    df.boxplot(
        column="latency_ms",
        by="tool_count",
        ax=ax,
    )

    ax.set(
        xlabel="Available tools",
        ylabel="Latency (ms)",
        title="Latency distribution",
    )

    fig.suptitle("")

    save(
        fig,
        out_dir / "07_latency_boxplot.png",
    )

    # ================================================================
    # 08 — Failure breakdown
    # ================================================================

    failures = df[
        ~df["exact_call_correct"].astype(bool)
    ].copy()

    if not failures.empty:

        def classify_failure(row):
            if not bool(row["tool_correct"]):
                return "Wrong tool"

            if not bool(row["arguments_correct"]):
                return "Wrong arguments"

            return "Other"

        failures["error_type"] = failures.apply(
            classify_failure,
            axis=1,
        )

        fig, ax = plt.subplots(figsize=(7, 5))

        (
            failures["error_type"]
            .value_counts()
            .plot(
                kind="bar",
                ax=ax,
            )
        )

        ax.set(
            xlabel="",
            ylabel="Failed calls",
            title="Failure breakdown",
        )

        ax.tick_params(
            axis="x",
            rotation=20,
        )

        save(
            fig,
            out_dir / "08_failure_breakdown.png",
        )

    # ================================================================
    # 09 — Extra tool calls vs tool count
    # ================================================================

    if "num_extra_tools" in df.columns:

        extra = (
            df.groupby("tool_count")["num_extra_tools"]
            .mean()
            .reset_index()
        )

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(
            extra["tool_count"],
            extra["num_extra_tools"],
            marker="o",
        )

        ax.set(
            xlabel="Available tools",
            ylabel="Mean extra tool calls",
            title="Spurious tool calls vs tool count",
        )

        ax.grid(True, alpha=0.25)

        save(
            fig,
            out_dir / "09_extra_tools_vs_count.png",
        )

    # ================================================================
    # 10 — Missing tool calls vs tool count
    # ================================================================

    if "num_missing_tools" in df.columns:

        missing = (
            df.groupby("tool_count")["num_missing_tools"]
            .mean()
            .reset_index()
        )

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(
            missing["tool_count"],
            missing["num_missing_tools"],
            marker="o",
        )

        ax.set(
            xlabel="Available tools",
            ylabel="Mean missing tool calls",
            title="Missing tool calls vs tool count",
        )

        ax.grid(True, alpha=0.25)

        save(
            fig,
            out_dir / "10_missing_tools_vs_count.png",
        )