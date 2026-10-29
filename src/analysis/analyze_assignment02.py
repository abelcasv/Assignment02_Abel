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
    "tiled_b32": "Tiled Blocked ($B_s=32$)"
}

EXPECTED_DENSE_METHODS = ["baseline_ijk", "reordered_ikj", "tiled_b32"]
EXPECTED_DENSE_SIZES = [250, 500, 750, 1000, 1500, 2000]
EXPECTED_SPARSE_APPROACHES = ["dense_matched", "sparse_csr"]
EXPECTED_SPARSITY_PCTS = [99, 95, 90, 80, 50, 20, 5]
EXPECTED_REPETITIONS_SET = {1, 2, 3, 4, 5}


def validate_dense_dataframe(df: pd.DataFrame, filepath: str) -> str:
    """Validate dense benchmark DataFrame integrity according to strict experimental protocol."""
    required_cols = {"method", "n", "repetition", "time_s", "numeric_type", "threads"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"[VALIDATION ERROR] Missing mandatory columns in {filepath}: {missing}")

    # Determine memory column
    if "peak_memory_mib" in df.columns:
        mem_col = "peak_memory_mib"
    elif "memory_mib" in df.columns:
        mem_col = "memory_mib"
    elif "memory_mb" in df.columns:
        mem_col = "memory_mb"
    else:
        raise ValueError(f"[VALIDATION ERROR] No valid memory column found in {filepath}")

    check_cols = list(required_cols) + [mem_col]
    if "structural_memory_mib" in df.columns:
        check_cols.append("structural_memory_mib")

    # Check for empty / null values
    if df[check_cols].isnull().any().any():
        null_counts = df[check_cols].isnull().sum().to_dict()
        raise ValueError(f"[VALIDATION ERROR] Null/NaN values found in {filepath}: {null_counts}")

    # Validate execution times (finite and > 0)
    if not np.isfinite(df["time_s"]).all() or (df["time_s"] <= 0.0).any():
        invalid = df[~np.isfinite(df["time_s"]) | (df["time_s"] <= 0.0)]
        raise ValueError(f"[VALIDATION ERROR] Invalid execution time (non-finite or <= 0) in {filepath}:\n{invalid}")

    # Validate memory (finite and > 0)
    if not np.isfinite(df[mem_col]).all() or (df[mem_col] <= 0.0).any():
        invalid = df[~np.isfinite(df[mem_col]) | (df[mem_col] <= 0.0)]
        raise ValueError(f"[VALIDATION ERROR] Invalid memory metric (non-finite or <= 0) in {filepath}:\n{invalid}")

    # Validate structural memory if present (finite and > 0)
    if "structural_memory_mib" in df.columns:
        if not np.isfinite(df["structural_memory_mib"]).all() or (df["structural_memory_mib"] <= 0.0).any():
            invalid = df[~np.isfinite(df["structural_memory_mib"]) | (df["structural_memory_mib"] <= 0.0)]
            raise ValueError(f"[VALIDATION ERROR] Invalid structural_memory_mib (non-finite or <= 0) in {filepath}:\n{invalid}")

    # Validate metadata consistency
    if (df["numeric_type"] != "double").any():
        bad_types = df["numeric_type"].unique()
        raise ValueError(f"[VALIDATION ERROR] Inconsistent numeric_type in {filepath}: expected 'double', found {bad_types}")
    if (df["threads"] != 1).any():
        bad_threads = df["threads"].unique()
        raise ValueError(f"[VALIDATION ERROR] Inconsistent threads in {filepath}: expected 1, found {bad_threads}")

    # Validate methods
    actual_methods = set(df["method"].unique())
    expected_m = set(EXPECTED_DENSE_METHODS)
    if not expected_m.issubset(actual_methods):
        raise ValueError(f"[VALIDATION ERROR] Expected methods {expected_m} not found in {filepath}. Found: {actual_methods}")

    # Validate sizes
    actual_sizes = set(df["n"].unique())
    expected_s = set(EXPECTED_DENSE_SIZES)
    if not expected_s.issubset(actual_sizes):
        raise ValueError(f"[VALIDATION ERROR] Expected sizes {expected_s} not found in {filepath}. Found: {actual_sizes}")

    # Validate exact repetition set {1, 2, 3, 4, 5} for each configuration
    for (m, s), group in df.groupby(["method", "n"]):
        rep_set = set(group["repetition"])
        if rep_set != EXPECTED_REPETITIONS_SET or len(group["repetition"]) != 5:
            raise ValueError(f"[VALIDATION ERROR] Configuration (method={m}, n={s}) does not have exact repetitions {EXPECTED_REPETITIONS_SET}. Found: {sorted(rep_set)} (count={len(group)})")

    print(f"[DATA INTEGRITY] Dense CSV ({filepath}): Validated {len(df)} rows across {len(actual_sizes)} sizes and {len(actual_methods)} methods.")
    print(f"                 Using memory metric column: '{mem_col}', exact repetitions {EXPECTED_REPETITIONS_SET} verified.")
    return mem_col


def validate_sparse_dataframe(df: pd.DataFrame, filepath: str) -> str:
    """Validate sparse benchmark DataFrame integrity according to strict experimental protocol."""
    required_cols = {
        "approach", "representation", "n", "sparsity_pct", "density_pct",
        "repetition", "kernel_time_s", "conversion_time_s",
        "nnz_a", "nnz_b", "nnz_c", "numeric_type", "threads"
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"[VALIDATION ERROR] Missing mandatory columns in {filepath}: {missing}")

    # Determine memory column
    if "peak_memory_mib" in df.columns:
        mem_col = "peak_memory_mib"
    elif "memory_mib" in df.columns:
        mem_col = "memory_mib"
    elif "memory_mb" in df.columns:
        mem_col = "memory_mb"
    else:
        raise ValueError(f"[VALIDATION ERROR] No valid memory column found in {filepath}")

    check_cols = list(required_cols) + [mem_col]
    if "structural_memory_mib" in df.columns:
        check_cols.append("structural_memory_mib")

    # Check for empty / null values
    if df[check_cols].isnull().any().any():
        null_counts = df[check_cols].isnull().sum().to_dict()
        raise ValueError(f"[VALIDATION ERROR] Null/NaN values found in {filepath}: {null_counts}")

    # Validate kernel and conversion times (finite, kernel > 0, conv >= 0)
    if not np.isfinite(df["kernel_time_s"]).all() or (df["kernel_time_s"] <= 0.0).any():
        invalid = df[~np.isfinite(df["kernel_time_s"]) | (df["kernel_time_s"] <= 0.0)]
        raise ValueError(f"[VALIDATION ERROR] Invalid kernel time (non-finite or <= 0) in {filepath}:\n{invalid}")
    if not np.isfinite(df["conversion_time_s"]).all() or (df["conversion_time_s"] < 0.0).any():
        invalid = df[~np.isfinite(df["conversion_time_s"]) | (df["conversion_time_s"] < 0.0)]
        raise ValueError(f"[VALIDATION ERROR] Invalid conversion time (non-finite or < 0) in {filepath}:\n{invalid}")

    # Validate memory (finite and > 0)
    if not np.isfinite(df[mem_col]).all() or (df[mem_col] <= 0.0).any():
        invalid = df[~np.isfinite(df[mem_col]) | (df[mem_col] <= 0.0)]
        raise ValueError(f"[VALIDATION ERROR] Invalid memory metric (non-finite or <= 0) in {filepath}:\n{invalid}")

    # Validate structural memory if present (finite and > 0)
    if "structural_memory_mib" in df.columns:
        if not np.isfinite(df["structural_memory_mib"]).all() or (df["structural_memory_mib"] <= 0.0).any():
            invalid = df[~np.isfinite(df["structural_memory_mib"]) | (df["structural_memory_mib"] <= 0.0)]
            raise ValueError(f"[VALIDATION ERROR] Invalid structural_memory_mib (non-finite or <= 0) in {filepath}:\n{invalid}")

    # Validate NNZ (finite integers, range [0, n^2], and nnz_a > 0, nnz_b > 0)
    max_nnz = df["n"] ** 2
    for nnz_col in ["nnz_a", "nnz_b", "nnz_c"]:
        if not np.isfinite(df[nnz_col]).all():
            invalid = df[~np.isfinite(df[nnz_col])]
            raise ValueError(f"[VALIDATION ERROR] Non-finite values in {nnz_col} in {filepath}:\n{invalid}")
        if not (df[nnz_col] % 1 == 0).all():
            invalid = df[df[nnz_col] % 1 != 0]
            raise ValueError(f"[VALIDATION ERROR] Non-integer values in {nnz_col} in {filepath}:\n{invalid}")
        if (df[nnz_col] < 0).any() or (df[nnz_col] > max_nnz).any():
            invalid = df[(df[nnz_col] < 0) | (df[nnz_col] > max_nnz)]
            raise ValueError(f"[VALIDATION ERROR] {nnz_col} out of range [0, n^2] in {filepath}:\n{invalid}")

    if (df["nnz_a"] <= 0).any():
        invalid = df[df["nnz_a"] <= 0]
        raise ValueError(f"[VALIDATION ERROR] Input nnz_a must be strictly positive in {filepath}:\n{invalid}")
    if (df["nnz_b"] <= 0).any():
        invalid = df[df["nnz_b"] <= 0]
        raise ValueError(f"[VALIDATION ERROR] Input nnz_b must be strictly positive in {filepath}:\n{invalid}")

    # Validate metadata consistency
    if (df["numeric_type"] != "double").any():
        bad_types = df["numeric_type"].unique()
        raise ValueError(f"[VALIDATION ERROR] Inconsistent numeric_type in {filepath}: expected 'double', found {bad_types}")
    if (df["threads"] != 1).any():
        bad_threads = df["threads"].unique()
        raise ValueError(f"[VALIDATION ERROR] Inconsistent threads in {filepath}: expected 1, found {bad_threads}")

    # Validate representation per approach
    dense_rep = df[df["approach"] == "dense_matched"]["representation"].unique()
    if len(dense_rep) != 1 or dense_rep[0] != "dense_row_major":
        raise ValueError(f"[VALIDATION ERROR] Expected representation 'dense_row_major' for dense_matched, found {dense_rep}")
    sparse_rep = df[df["approach"] == "sparse_csr"]["representation"].unique()
    if len(sparse_rep) != 1 or sparse_rep[0] != "CSR":
        raise ValueError(f"[VALIDATION ERROR] Expected representation 'CSR' for sparse_csr, found {sparse_rep}")

    # Validate sparsity and density relation
    if ((df["sparsity_pct"] + df["density_pct"]) != 100).any():
        raise ValueError(f"[VALIDATION ERROR] Inconsistent sparsity/density percentage in {filepath}")

    # Validate approaches
    actual_approaches = set(df["approach"].unique())
    expected_app = set(EXPECTED_SPARSE_APPROACHES)
    if not expected_app.issubset(actual_approaches):
        raise ValueError(f"[VALIDATION ERROR] Expected approaches {expected_app} not found in {filepath}. Found: {actual_approaches}")

    # Validate sparsities
    actual_sparsities = set(df["sparsity_pct"].unique())
    expected_sp = set(EXPECTED_SPARSITY_PCTS)
    if not expected_sp.issubset(actual_sparsities):
        raise ValueError(f"[VALIDATION ERROR] Expected sparsities {expected_sp} not found in {filepath}. Found: {actual_sparsities}")

    # Validate exact repetition set {1, 2, 3, 4, 5} for each configuration
    for (app, sp), group in df.groupby(["approach", "sparsity_pct"]):
        rep_set = set(group["repetition"])
        if rep_set != EXPECTED_REPETITIONS_SET or len(group["repetition"]) != 5:
            raise ValueError(f"[VALIDATION ERROR] Configuration (approach={app}, sparsity={sp}) does not have exact repetitions {EXPECTED_REPETITIONS_SET}. Found: {sorted(rep_set)} (count={len(group)})")

    print(f"[DATA INTEGRITY] Sparse CSV ({filepath}): Validated {len(df)} rows across {len(actual_sparsities)} sparsity levels and {len(actual_approaches)} approaches.")
    print(f"                 Using memory metric column: '{mem_col}', exact repetitions {EXPECTED_REPETITIONS_SET} verified.")
    return mem_col


def analyze_dense():
    dense_path = os.path.join(RESULTS_DIR, "dense_results.csv")
    if not os.path.exists(dense_path):
        print(f"File not found: {dense_path}")
        return

    df = pd.read_csv(dense_path)
    mem_col = validate_dense_dataframe(df, dense_path)

    agg_dict = {
        "median_time": ("time_s", "median"),
        "q1_time": ("time_s", lambda x: np.percentile(x, 25)),
        "q3_time": ("time_s", lambda x: np.percentile(x, 75)),
        "reps": ("repetition", "count"),
        "median_memory_mib": (mem_col, "median"),
        "q1_memory_mib": (mem_col, lambda x: np.percentile(x, 25)),
        "q3_memory_mib": (mem_col, lambda x: np.percentile(x, 75))
    }
    if "structural_memory_mib" in df.columns:
        agg_dict["structural_memory_mib"] = ("structural_memory_mib", "first")

    summary = df.groupby(["method", "n"]).agg(**agg_dict).reset_index()

    summary["iqr_time"] = summary["q3_time"] - summary["q1_time"]
    summary["iqr_memory_mib"] = summary["q3_memory_mib"] - summary["q1_memory_mib"]

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
    ax.bar(x + width/2, tiled_speedups, width, label="Tiled Blocked ($B_s=32$)", color=DENSE_COLORS["tiled_b32"])
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
    mem_col = validate_sparse_dataframe(df, sparse_path)

    agg_dict = {
        "median_kernel_time": ("kernel_time_s", "median"),
        "q1_kernel_time": ("kernel_time_s", lambda x: np.percentile(x, 25)),
        "q3_kernel_time": ("kernel_time_s", lambda x: np.percentile(x, 75)),
        "median_conv_time": ("conversion_time_s", "median"),
        "nnz_a": ("nnz_a", "first"),
        "nnz_b": ("nnz_b", "first"),
        "nnz_c": ("nnz_c", "first"),
        "reps": ("repetition", "count"),
        "median_memory_mib": (mem_col, "median")
    }
    if "structural_memory_mib" in df.columns:
        agg_dict["structural_memory_mib"] = ("structural_memory_mib", "first")

    summary = df.groupby(["approach", "sparsity_pct", "density_pct"]).agg(**agg_dict).reset_index()

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
