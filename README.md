# Assignment 2 — Optimised Matrix Multiplication Approaches and Sparse Matrices

**Course:** Big Data · Academic Year 2026-2027  
**Degree:** Grado en Ciencia e Ingeniería de Datos  
**Institution:** Universidad de Las Palmas de Gran Canaria  
**Student:** Abel Castro Villar  

---

## 1. Research Objectives & Hypotheses

### Research Questions
1. **Part A (Dense Optimisation):** How do loop reordering (`i-k-j`) and cache blocking (`Bs × Bs` tiling) affect the execution runtime, cache locality, and scalability of dense matrix multiplication compared to the canonical baseline (`i-j-k`)?
2. **Part B (Sparse Matrices):** Under which sparsity conditions does a Compressed Sparse Row (CSR) representation provide a computational advantage over a dense method, and where is the performance crossover threshold?

### Hypotheses
- **Dense Optimisation:** The baseline (`i-j-k`) suffers severe cache misses due to stride-`n` access on matrix `B`. Loop interchange (`i-k-j`) converts inner-loop memory access into contiguous unit-stride streams, providing major speedups (>2.5x). Cache blocking (`Bs = 32`) will further enhance performance on large matrices (`n ≥ 1000`) by confining working sets within the 48 KB L1 data cache.
- **Sparse Crossover & Fill-in:** CSR SpGEMM will drastically outperform dense multiplication at high sparsity (≥ 90%), but indexing and dynamic accumulation overhead will cause a performance crossover between 50% and 80% sparsity, where dense multiplication becomes faster. The product matrix `C` will experience substantial *fill-in*, becoming significantly denser than the input matrices.

---

## 2. Experimental Environment

All experiments were executed on a dedicated physical host under strictly controlled conditions:

| Parameter | Specification |
|---|---|
| **Operating System** | Windows 11 Home (64-bit, Build 22631) |
| **CPU Model** | 13th Gen Intel(R) Core(TM) i5-1335U |
| **Physical Cores / Threads** | 10 physical cores (2 P-cores + 8 E-cores) / 12 threads |
| **L1 / L2 / L3 Cache** | L1d: 48 KB per core · L2: 1.25 MB per P-core · L3: 12 MB shared |
| **RAM** | 16.0 GB DDR4 |
| **C Compiler & Flags** | GCC 16.1.0 (MSYS2 UCRT64), flags: `-Wall -Wextra -O3 -std=c11 -lpsapi` |
| **Python Version** | Python 3.14.0 (standard 64-bit) |
| **Random Seed** | 42 (Mersenne Twister standard generator) |
| **Numeric Type** | 64-bit floating point (`double` / IEEE 754) |

---

## 3. Experimental Protocol

1. **Controlled Datasets:** Generated with fixed seed (42). Dense matrices n in {250, 500, 750, 1000, 1500, 2000}; Sparse matrices n = 1000 across 7 sparsity levels (99%, 95%, 90%, 80%, 50%, 20%, 5%).
2. **Correctness Validation:** All methods verified against 4 × 4 identity, zero, and analytical fixture matrices before benchmarking (ε = 10⁻⁴).
3. **Timing Boundary:** Kernel execution only. File loading and memory allocations are excluded.
4. **Timers & Sinks:** Monotonic timer (`QueryPerformanceCounter`). Computation results assigned to `volatile double result_sink` to prevent dead-code elimination.
5. **Repetitions & Statistics:** 1 warm-up run (discarded) followed by 5 recorded repetitions. Reported using **Median** and **Interquartile Range (IQR)**.
6. **Conversion Costs:** In Part B, CSR conversion time from raw matrices is measured and reported separately from the multiplication kernel.

---

## 4. Empirical Results

### Part A: Dense Optimisation Summary (Median over 5 repetitions)

| Dimension (n) | Arithmetic Work (2n³ FLOPs) | Baseline i-j-k (s) | Reordered i-k-j (s) | Tiled Bs=32 (s) | i-k-j Speedup | Tiled Speedup |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **250**  | 3.12 × 10⁷ | 0.020869 | 0.009115 | 0.008408 | 2.29x | **2.48x** |
| **500**  | 2.50 × 10⁸ | 0.212222 | 0.089654 | 0.074155 | 2.37x | **2.86x** |
| **750**  | 8.44 × 10⁸ | 0.754074 | 0.306706 | 0.245148 | 2.46x | **3.08x** |
| **1000** | 2.00 × 10⁹ | 2.053200 | 0.893935 | 0.747627 | 2.30x | **2.75x** |
| **1500** | 6.75 × 10⁹ | *Budget Limit* | 3.801412 | 2.404100 | — | **1.58x vs ikj** |
| **2000** | 1.60 × 10¹⁰| *Budget Limit* | 9.176827 | 5.510334 | — | **1.67x vs ikj** |

### Part B: Sparse vs Dense Comparison (n = 1000, Median over 5 repetitions)

| Sparsity (%) | Density (%) | Input NNZ (A) | Output NNZ (C) | Fill-in Ratio | CSR Kernel (s) | CSR Conv (s) | Matched Dense (s) | Advantage |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **99%** | 1%  | 10,031  | 96,358    | 9.61x  | **0.001975** | 0.007786 | 0.018773 | **CSR 9.5x faster** |
| **95%** | 5%  | 49,885  | 918,582   | 18.41x | **0.043214** | 0.008563 | 0.062595 | **CSR 1.45x faster** |
| **90%** | 10% | 100,045 | 999,944   | 9.99x  | **0.064241** | 0.013579 | 0.100742 | **CSR 1.57x faster** |
| **80%** | 20% | 200,145 | 1,000,000 | 5.00x  | **0.132181** | 0.020991 | 0.185873 | **CSR 1.41x faster** |
| **50%** | 50% | 500,051 | 1,000,000 | 2.00x  | 0.596008     | 0.037773 | **0.481324** | **Dense 1.24x faster (Crossover)** |
| **20%** | 80% | 799,725 | 1,000,000 | 1.25x  | 1.556489     | 0.025888 | **0.704012** | **Dense 2.21x faster** |
| **5%**  | 95% | 950,094 | 1,000,000 | 1.05x  | 2.078866     | 0.017888 | **0.882064** | **Dense 2.36x faster** |

> **Detailed Research Report:**  
> 📄 [Assignment02_Report.md](Assignment02_Report.md) (or the compiled PDF: `Assignment02_Castro_Report.pdf`)

---

## 5. How to Reproduce

### Prerequisites
- Python 3.10+ (for dataset generation and analysis):
  ```bash
  pip install pandas matplotlib numpy
  ```
- GCC Compiler with C11 support (MSYS2 UCRT64 on Windows).

> **Note for Windows PowerShell (MSYS2):** Ensure GCC is accessible in your session:
> ```powershell
> $env:PATH = "C:\msys64\ucrt64\bin;$env:PATH"
> ```

---

### Step 1: Generate Datasets
```powershell
python src/generate_data.py
# (On Windows: py src/generate_data.py)
```

### Step 2: Compile Benchmark Binaries
```powershell
gcc -Wall -Wextra -O3 -std=c11 src/dense_benchmark.c -o src/dense_benchmark.exe -lpsapi
gcc -Wall -Wextra -O3 -std=c11 src/sparse_benchmark.c -o src/sparse_benchmark.exe -lpsapi
```

### Step 3: Run Dense Optimization Benchmark
```powershell
./src/dense_benchmark.exe .
```

### Step 4: Run Sparse vs Dense Benchmark
```powershell
./src/sparse_benchmark.exe .
```

### Step 5: Generate Summary Statistics and Visual Plots
```powershell
python src/analysis/analyze_assignment02.py
# (On Windows: py src/analysis/analyze_assignment02.py)
```

Generated metrics and plots are stored in `results/` and `results/plots/`.
