import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

DENSE_COLORS = {
    "baseline_ijk": "#d95f02",
    "reordered_ikj": "#1b9e77",
    "tiled_b32": "#7570b3"
}

DENSE_LABELS = {
    "baseline_ijk": "Baseline (i-j-k)",
    "reordered_ikj": "Reordered (i-k-j)",
    "tiled_b32": "Tiled Blocked (B_s=32)"
}


def analyze_dense():
    dense_path = os.path.join(RESULTS_DIR, "dense_results.csv")
    if not os.path.exists(dense_path):
        print(f"File not found: {dense_path}")
        return

    df = pd.read_csv(dense_path)
    summary = df.groupby(["method", "n"]).agg(
        median_time=("time_s", "median"),
        q1_time=("time_s", lambda x: np.percentile(x, 25)),
        q3_time=("time_s", lambda x: np.percentile(x, 75)),
        median_mem=("memory_mb", "median"),
        q1_mem=("memory_mb", lambda x: np.percentile(x, 25)),
        q3_mem=("memory_mb", lambda x: np.percentile(x, 75)),
        reps=("repetition", "count")
    ).reset_index()

    summary["iqr_time"] = summary["q3_time"] - summary["q1_time"]
    summary["iqr_mem"] = summary["q3_mem"] - summary["q1_mem"]

    # Calculate speedup relative to baseline for each n
    baseline_medians = summary[summary["method"] == "baseline_ijk"].set_index("n")["median_time"]
    summary["speedup"] = summary.apply(
        lambda row: baseline_medians.get(row["n"], np.nan) / row["median_time"]
        if row["n"] in baseline_medians else np.nan,
        axis=1
    )

    summary.to_csv(os.path.join(RESULTS_DIR, "dense_summary.csv"), index=False)
    print("Saved dense_summary.csv")

    # Plot 1: Dense execution time (semi-log)
    fig, ax = plt.subplots(figsize=(10, 6))
    for method in ["baseline_ijk", "reordered_ikj", "tiled_b32"]:
        sub = summary[summary["method"] == method].sort_values("n")
        if sub.empty:
            continue
        yerr = [sub["median_time"] - sub["q1_time"], sub["q3_time"] - sub["median_time"]]
        ax.errorbar(
            sub["n"], sub["median_time"], yerr=yerr,
            marker="o", linewidth=2.2, markersize=7, capsize=4,
            label=DENSE_LABELS.get(method, method),
            color=DENSE_COLORS.get(method, "blue")
        )

    ax.set_yscale("log")
    ax.set_xlabel("Matrix Dimension ($n$)", fontsize=12)
    ax.set_ylabel("Execution Time (seconds, log scale)", fontsize=12)
    ax.set_title("Dense Matrix Multiplication: Baseline vs. Optimised Methods", fontsize=14, fontweight="bold")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "dense_time_comparison.png"), dpi=300)
    plt.close(fig)
    print("Saved dense_time_comparison.png")

    # Plot 2: Speedup comparison bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    sizes = sorted(summary["n"].unique())
    x = np.arange(len(sizes))
    width = 0.35

    ikj_speedups = [
        summary[(summary["method"] == "reordered_ikj") & (summary["n"] == s)]["speedup"].values[0]
        if not summary[(summary["method"] == "reordered_ikj") & (summary["n"] == s)].empty else 0
        for s in sizes
    ]
    tiled_speedups = [
        summary[(summary["method"] == "tiled_b32") & (summary["n"] == s)]["speedup"].values[0]
        if not summary[(summary["method"] == "tiled_b32") & (summary["n"] == s)].empty else 0
        for s in sizes
    ]

    ax.bar(x - width/2, ikj_speedups, width, label="Reordered (i-k-j)", color=DENSE_COLORS["reordered_ikj"])
    ax.bar(x + width/2, tiled_speedups, width, label="Tiled Blocked (B_s=32)", color=DENSE_COLORS["tiled_b32"])
    ax.axhline(1.0, color="grey", linestyle="--", label="Baseline (1.0x)")

    ax.set_xlabel("Matrix Dimension ($n$)", fontsize=12)
    ax.set_ylabel("Speedup Factor vs. Baseline ($i$-$j$-$k$)", fontsize=12)
    ax.set_title("Dense Optimisation Speedup relative to Baseline", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "dense_speedup.png"), dpi=300)
    plt.close(fig)
    print("Saved dense_speedup.png")


def analyze_sparse():
    sparse_path = os.path.join(RESULTS_DIR, "sparse_results.csv")
    if not os.path.exists(sparse_path):
        print(f"File not found: {sparse_path}")
        return

    df = pd.read_csv(sparse_path)
    summary = df.groupby(["approach", "sparsity_pct", "density_pct"]).agg(
        median_kernel_time=("kernel_time_s", "median"),
        q1_kernel_time=("kernel_time_s", lambda x: np.percentile(x, 25)),
        q3_kernel_time=("kernel_time_s", lambda x: np.percentile(x, 75)),
        median_conv_time=("conversion_time_s", "median"),
        median_mem=("memory_mb", "median"),
        nnz_a=("nnz_a", "first"),
        nnz_b=("nnz_b", "first"),
        nnz_c=("nnz_c", "first"),
        reps=("repetition", "count")
    ).reset_index()

    summary["iqr_kernel_time"] = summary["q3_kernel_time"] - summary["q1_kernel_time"]
    summary.to_csv(os.path.join(RESULTS_DIR, "sparse_summary.csv"), index=False)
    print("Saved sparse_summary.csv")

    # Plot 3: Sparse vs Dense Crossover Curve
    fig, ax = plt.subplots(figsize=(10, 6))

    csr = summary[summary["approach"] == "sparse_csr"].sort_values("sparsity_pct")
    dense = summary[summary["approach"] == "dense_matched"].sort_values("sparsity_pct")

    ax.plot(
        csr["sparsity_pct"], csr["median_kernel_time"],
        marker="s", linewidth=2.5, markersize=8, color="#2b83ba",
        label="Sparse CSR SpGEMM (Kernel only)"
    )
    ax.plot(
        csr["sparsity_pct"], csr["median_kernel_time"] + csr["median_conv_time"],
        marker="^", linewidth=2.0, linestyle=":", markersize=7, color="#1a9641",
        label="Sparse CSR Total (Kernel + Conversion)"
    )
    ax.plot(
        dense["sparsity_pct"], dense["median_kernel_time"],
        marker="o", linewidth=2.5, linestyle="--", markersize=8, color="#d7191c",
        label="Matched Dense Method (i-k-j)"
    )

    ax.set_yscale("log")
    ax.set_xlabel("Sparsity Level (% of Zero Entries)", fontsize=12)
    ax.set_ylabel("Execution Time (seconds, log scale)", fontsize=12)
    ax.set_title("Sparse vs. Dense Performance Across Sparsity Levels (n=1000)", fontsize=14, fontweight="bold")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "sparse_vs_dense_crossover.png"), dpi=300)
    plt.close(fig)
    print("Saved sparse_vs_dense_crossover.png")

    # Plot 4: Fill-in Analysis (Input NNZ vs Output NNZ)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(csr["sparsity_pct"], csr["nnz_a"], marker="o", linewidth=2.2, label="Input Matrix A (NNZ)", color="#2b83ba")
    ax.plot(csr["sparsity_pct"], csr["nnz_c"], marker="D", linewidth=2.2, label="Output Matrix C (NNZ, Fill-in)", color="#e66101")

    ax.set_xlabel("Sparsity Level (% of Zero Entries)", fontsize=12)
    ax.set_ylabel("Non-Zero Elements (NNZ)", fontsize=12)
    ax.set_title("Sparse Matrix Multiplication Fill-in Phenomenon (n=1000)", fontsize=14, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "sparse_fillin_analysis.png"), dpi=300)
    plt.close(fig)
    print("Saved sparse_fillin_analysis.png")


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    print("=== Analyzing Assignment 2 Benchmarks ===")
    analyze_dense()
    analyze_sparse()
    print("All Assignment 2 analyses and plots completed.")


if __name__ == "__main__":
    main()
