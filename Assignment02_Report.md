# Performance Study of Dense Matrix Multiplication Optimisations and Sparse CSR Representations

**Author:** Abel Castro Villar  
**Course:** Big Data (2026-2027)  
**Assignment:** Assignment 2  
**Date:** October 2026  

---

## Abstract
This empirical investigation evaluates high-performance matrix multiplication techniques across two distinct paradigms: sequential dense algorithmic optimisations (Part A) and Compressed Sparse Row (CSR) representations under varying sparsity conditions (Part B). Using a native C11 implementation compiled with GCC `-O3` under controlled deterministic inputs (Mersenne Twister, seed 42), Part A assesses the canonical baseline (`i-j-k`), loop interchange (`i-k-j`), and cache blocking (`Bs = 32`) across square dimensions n ∈ {250, 500, 750, 1000, 1500, 2000}. Loop interchange eliminates non-unit memory strides, yielding a 3.14× speedup over the baseline at n = 1000 and a 6.86× speedup at n = 2000. Cache blocking further confines active working sets within the 48 KB L1 data cache, achieving a 3.72× speedup at n = 1000 (0.524 s vs 1.947 s) and scaling stably to n = 2000 with a 10.33× speedup over the baseline (4.192 s vs 43.293 s, and 1.50× faster than pure `i-k-j`). Part B investigates custom Gustavson sparse matrix-matrix multiplication (SpGEMM) at n = 1000 across seven declared sparsity levels (99% to 5% zero entries) against a matched dense `i-k-j` baseline (≈ 0.51–0.53 s). At 99% sparsity (1% density), CSR SpGEMM completes in 0.001698 s—312.71× faster than matched dense execution (and 79.62× faster when the 0.004971 s dense-to-CSR conversion cost is included)—while reducing process memory from 26.89 MB to 5.73 MB. However, as sparsity decreases, sparse indexing overhead and substantial arithmetic fill-in (e.g., 5% density inputs producing 91.86% non-zero outputs) erode this advantage. At 50% sparsity, CSR retains only a marginal 1.17× kernel advantage (1.10× with conversion) while its memory footprint (27.09 MB) crosses over standalone dense storage (26.89 MB). A runtime crossover occurs between 50% and 20% sparsity (at ≈ 46.1%–47.6% sparsity), beyond which dense multiplication becomes 2.03× to 2.76× faster.

---

## 1. Introduction and Objectives

Dense matrix multiplication is a foundational operation in numerical linear algebra, machine learning, and scientific computing. In Assignment 1, the canonical O(n³) triple-loop algorithm was established across languages. However, naive implementations suffer from severe memory bandwidth bottlenecks and CPU cache thrashing due to non-contiguous memory access patterns [1]. Furthermore, in graph analytics, network science, and large-scale big data applications, matrices often contain an overwhelming majority of zero entries, rendering dense storage (8n² bytes) and unconditional multiplication over zeros computationally and spatially wasteful [3].

### Research Questions
1. **Part A (Dense Optimisation):** How do loop reordering (`i-k-j`) and cache blocking (`Bs × Bs` tiling) affect execution runtime, run-to-run variability, memory hierarchy utilisation, and scalability up to n = 2000 compared to the validated canonical baseline (`i-j-k`)?
2. **Part B (Sparse Matrices):** Under which tested sparsity conditions does a Compressed Sparse Row (CSR) representation provide an execution time and memory advantage over dense methods, and where does the performance crossover occur when accounting for dense-to-sparse conversion costs, indexing indirection, and matrix fill-in?

### Hypotheses
1. **Loop Interchange Advantage:** The baseline `i-j-k` loop incurs cache misses of order O(n) in its innermost loop because matrix B is traversed along columns with stride-n. Reordering to `i-k-j` enables contiguous stride-1 access for both B and C, allowing hardware prefetching and SIMD vectorisation to yield >2.0× speedups across all tested dimensions [1].
2. **Cache Tiling Scalability:** Cache blocking with block size Bs = 32 (sized to fit comfortably within the 48 KB L1 data cache) will outperform pure loop interchange at large dimensions (n ≥ 1500) where full matrices exceed the 12 MB L3 cache, maintaining both temporal and spatial locality across multiple loop passes [4].
3. **Sparse Crossover and Fill-in:** CSR SpGEMM will be dramatically faster and more memory-efficient than dense methods at high sparsity (≥ 80% zeros), but pointer indirection and dynamic accumulator scanning will cause a performance crossover between 50% and 20% sparsity. Multiplication will trigger substantial *fill-in*, making the output matrix significantly denser than the inputs [2, 3].

---

## 2. Background and Theoretical Framework

### 2.1 Cache Hierarchy and Memory Strides
Modern microprocessors access main memory through a multi-tiered cache hierarchy (L1d, L2, L3). A standard cache line holds 64 bytes (8 double-precision floats) [1]:
- **Canonical Baseline (`i-j-k`):** In the innermost loop over k, the program reads `A[i][k]` (contiguous, stride-1) and `B[k][j]` (column-wise, stride-n). For n = 1000, consecutive elements of B are spaced 8000 bytes apart, ensuring that almost every memory access to B produces a capacity cache miss across L1, L2, and L3 when working sets grow large [1].
- **Loop Interchange (`i-k-j`):** Permuting the middle and inner loops alters the innermost pass over index j:
  ```text
  C[i][j] += A[i][k] * B[k][j]
  ```
  Here, `A[i][k]` is invariant with respect to j and remains pinned in a scalar CPU register. Elements `B[k][j]` and `C[i][j]` are traversed consecutively in row-major order with unit stride (stride-1). Hardware stream prefetchers load full 64-byte cache lines ahead of execution and GCC auto-vectorisation emits packed AVX2 SIMD instructions, eliminating memory stalls [1].
- **Cache Blocking / Tiling (`Bs × Bs`):** At large dimensions (e.g., n = 2000, where a single matrix requires 32 MB and three matrices require 96 MB), full rows and matrices overflow the 12 MB L3 cache. Rather than traversing entire matrix rows of length n, the matrices are partitioned into square sub-blocks of size Bs × Bs [4]. For a block size Bs = 32, the memory footprint of three active sub-blocks (`A_tile`, `B_tile`, `C_tile`) is:
  ```text
  Memory_tiles = 3 × Bs² × 8 bytes = 3 × 1024 × 8 = 24,576 bytes (24.0 KiB)
  ```
  This fits comfortably within the 48 KB L1 data cache of the host Intel processor, ensuring that cached data is reused Bs = 32 times before eviction [4].

### 2.2 Compressed Sparse Row (CSR) Format and SpGEMM
When a matrix of dimension n × n contains NNZ ≪ n² non-zero elements, storing n² floating-point entries wastes both memory and arithmetic cycles [3]. We define **sparsity** as the percentage of zero entries (`100 × (1 - NNZ / n²)`) and **density** as the percentage of non-zero entries (`100 × NNZ / n²`):
- **CSR Data Structure:** A CSR matrix is represented by three contiguous arrays [3]:
  1. `values`: array of `double` containing non-zero entries (size NNZ, 8 bytes each).
  2. `col_indices`: array of `int` storing the column index of each non-zero entry (size NNZ, 4 bytes each).
  3. `row_ptr`: array of `int` pointing to the starting offset of each row in `values` and `col_indices` (size n + 1, 4 bytes each).
  Total structural memory consumption is `12 · NNZ + 4(n + 1)` bytes, compared to `8n²` bytes for dense storage. A single CSR matrix occupies less memory than a dense matrix only when density remains below ≈ 66.6% [3].
- **Gustavson's SpGEMM Algorithm:** Multiplying two CSR matrices C = A × B requires computing linear combinations of rows of B scaled by non-zero entries of A [2]. Because the output row sparsity pattern is unknown *a priori*, an intermediate dense accumulator array (`dense_accum[n]`), a visited marker array (`marker[n]`), and an active column index list (`active_cols[n]`) are used to aggregate row values in O(FLOPs) time without sorting overhead [2].
- **Fill-in Phenomenon:** The product of two sparse matrices is typically much denser than either input [2, 3]. An entry `C[i][j]` is non-zero whenever there exists at least one index k where both `A[i][k] ≠ 0` and `B[k][j] ≠ 0`. For independent uniform random matrices of density d, the expected output density is `d_C = 1 - (1 - d²)ⁿ`. At n = 1000, an input density of just d = 0.05 (95% sparsity) yields `d_C = 1 - (1 - 0.0025)¹⁰⁰⁰ ≈ 91.8%`, rapidly saturating the output matrix.

---

## 3. Methodology and Experimental Protocol

### 3.1 Hardware, System Specification, and Code Attribution
All benchmarks were performed sequentially (single-threaded) on a dedicated physical host with background activity minimised:
- **CPU:** 13th Gen Intel(R) Core(TM) i5-1335U (10 physical cores: 2 P-cores + 8 E-cores, 12 threads).
- **Cache Capacities:** L1d: 48 KB per core; L2: 1.25 MB per P-core; L3: 12 MB shared.
- **RAM:** 16.0 GB DDR4.
- **OS & Compiler:** Windows 11 Home 64-bit (Build 22631); GCC 16.1.0 (MSYS2 UCRT64), flags: `-Wall -Wextra -O3 -std=c11 -lpsapi`.
- **Own Code vs. Library Functionality:** All dense multiplication kernels (`multiply_ijk`, `multiply_ikj`, `multiply_tiled` in `src/dense_benchmark.c`) and all sparse CSR structures and kernels (`CSRMatrix`, `dense_to_csr`, `multiply_csr_spgemm`, `multiply_dense_ikj` in `src/sparse_benchmark.c`) were written entirely from scratch in standard C11. No external numerical or sparse libraries (such as BLAS, Intel MKL, or SuiteSparse) were used; only standard C library headers and Windows OS measurement APIs (`windows.h`, `psapi.h`) were invoked.

### 3.2 Dataset Generation
All datasets were generated using a deterministic pseudorandom generator (Python `random.Random(42)`, Mersenne Twister MT19937, fixed seed = 42) via `src/generate_data.py`:
- **Dense Datasets:** Square matrices A and B for n ∈ {250, 500, 750, 1000, 1500, 2000} with uniformly distributed `double` values in [0.0, 1.0) formatted to 6 decimal places.
- **Sparse Datasets:** Matrices of dimension n = 1000 generated across seven declared sparsity levels: **99%, 95%, 90%, 80%, 50%, 20%, 5%** zeros (corresponding to densities of **1%, 5%, 10%, 20%, 50%, 80%, 95%** non-zeros), where each entry is independently populated with probability `density = 1.0 - sparsity`.
- **Matched Logical Inputs:** The exact same CSV matrices were loaded to evaluate both the CSR SpGEMM approach and the matched dense `i-k-j` baseline (`multiply_dense_ikj`).

### 3.3 Correctness Validation Protocol
Before any timing measurements, all three dense methods and CSR SpGEMM were validated against 4 × 4 analytical fixture matrices (`A × B = Expected_C`), identity matrices (`A × I = A`), and zero matrices (`A × 0 = 0`), as well as direct cross-check agreement between CSR SpGEMM and dense `i-k-j` within an absolute tolerance threshold of ε = 10⁻⁴. All correctness checks passed.

### 3.4 Measurement Boundaries and Protocol
- **Kernel Isolation:** Timers strictly bracketed multiplication kernel execution. File I/O, CSV parsing, and initial matrix allocations were excluded from kernel timings.
- **Separate Conversion Tracking and Memory Isolation:** In Part B, conversion time from dense row-major storage to CSR format (`dense_to_csr` for A and B) was instrumented and recorded independently. Dense input arrays (`dense_a`, `dense_b`, `dense_c`) were freed prior to running the CSR benchmark loop so that CSR process resident set size (RSS) measurements reflect isolated CSR memory usage.
- **High-Resolution Timing & Memory Telemetry:** Monotonic Windows API timer `QueryPerformanceCounter` for execution time and `GetProcessMemoryInfo` (`WorkingSetSize`) for resident memory in MB.
- **Dead-Code Elimination Prevention:** Kernel outputs were accumulated into `volatile double result_sink`.
- **Repetitions & Variability Metrics:** 1 untimed warm-up run (discarded) followed by 5 timed repetitions per configuration. Central tendency and variability are reported non-parametrically as **Median** and **Interquartile Range (IQR = Q3 − Q1)**.
- **Declared Budget:** Maximum execution budget of 120 s per repetition. All three dense methods completed within budget up to the largest tested dimension **n = 2000** (where the baseline `i-j-k` required 43.29 s per repetition).

---

## 4. Empirical Results

### 4.1 Part A: Dense Optimisation Results
Table 1 summarises median execution time, IQR variability, process memory footprint (RSS), and speedup factors across all dense configurations.

**Table 1: Dense Matrix Multiplication Performance across Dimensions n (Median and IQR of 5 repetitions).**

| Dimension (n) | Arithmetic Work (2n³ FLOPs) | Baseline `i-j-k` (s) [IQR] | Reordered `i-k-j` (s) [IQR] | Tiled `Bs=32` (s) [IQR] | `i-k-j` Speedup | Tiled Speedup (vs `i-k-j`) | Process RSS (MB) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **250**  | 3.12 × 10⁷ | 0.019362 [0.001112] | 0.009281 [0.001178] | **0.008645** [0.000664] | 2.09× | **2.24×** (1.07×) | 5.42  |
| **500**  | 2.50 × 10⁸ | 0.193144 [0.003933] | 0.072269 [0.007113] | **0.066373** [0.000934] | 2.67× | **2.91×** (1.09×) | 9.72  |
| **750**  | 8.44 × 10⁸ | 0.716699 [0.089923] | 0.236948 [0.001982] | **0.224766** [0.007220] | 3.02× | **3.19×** (1.05×) | 16.87 |
| **1000** | 2.00 × 10⁹ | 1.947354 [0.123171] | 0.620054 [0.042797] | **0.523642** [0.018781] | 3.14× | **3.72×** (1.18×) | 26.89 |
| **1500** | 6.75 × 10⁹ | 9.684193 [0.040139] | 4.139564 [1.308037] | **2.641865** [0.160843] | 2.34× | **3.67×** (1.57×) | 55.48 |
| **2000** | 1.60 × 10¹⁰| 43.293093 [0.244237]| 6.308155 [0.696529] | **4.192394** [0.006176] | 6.86× | **10.33×** (1.50×)| 95.62 |

![Figure 1: Dense Runtime Comparison](results/plots/dense_time_comparison.png)
*Figure 1: Median execution time (seconds, logarithmic scale, with IQR error bars) versus matrix dimension n across dense implementations.*

![Figure 2: Dense Speedup Factors](results/plots/dense_speedup.png)
*Figure 2: Speedup factors relative to the canonical baseline (`i-j-k`). Cache blocking (`Bs = 32`) reaches a 3.72× speedup at n = 1000 and a 10.33× speedup at n = 2000.*

### 4.2 Part B: Sparse CSR vs. Matched Dense Results (n = 1000)
Table 2 consolidates input non-zero counts (`NNZ(A)` and `NNZ(B)`), output non-zero counts (`NNZ(C)`), fill-in ratios (`NNZ(C) / NNZ(A)`), median kernel and conversion times, IQR variability, isolated CSR process memory, and relative speedups across all seven tested sparsity levels.

**Table 2: Sparse CSR SpGEMM vs. Matched Dense (`i-k-j`) Performance Across Sparsity Levels (n = 1000, Median and IQR of 5 repetitions; standalone Dense RSS at n = 1000 is 26.89 MB).**

| Sparsity (%) | Density (%) | Input NNZ (A) | Input NNZ (B) | Output NNZ (C) | Fill-in Ratio | CSR Kernel (s) [IQR] | CSR Conv (s) | Matched Dense (s) [IQR] | CSR RSS (MB) | Advantage: Kernel (w/ Conv) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **99%** | 1%  | 10,031  | 10,054  | 96,358    | 9.61×  | **0.001698** [0.000103] | 0.004971 | 0.530988 [0.019662] | 5.73  | **CSR 312.71×** (79.62×) |
| **95%** | 5%  | 49,885  | 50,249  | 918,582   | 18.41× | **0.037238** [0.001261] | 0.007464 | 0.516842 [0.009631] | 15.84 | **CSR 13.88×** (11.56×)  |
| **90%** | 10% | 100,045 | 99,788  | 999,944   | 9.99×  | **0.051188** [0.000487] | 0.010571 | 0.523098 [0.004033] | 18.07 | **CSR 10.22×** (8.47×)   |
| **80%** | 20% | 200,145 | 199,586 | 1,000,000 | 5.00×  | **0.102735** [0.002494] | 0.016465 | 0.505236 [0.008643] | 20.58 | **CSR 4.92×** (4.24×)    |
| **50%** | 50% | 500,051 | 500,351 | 1,000,000 | 2.00×  | **0.439561** [0.009119] | 0.030609 | 0.516296 [0.001550] | 27.09 | **CSR 1.17×** (1.10×)    |
| **20%** | 80% | 799,725 | 800,084 | 1,000,000 | 1.25×  | 1.026605 [0.016841]     | 0.018936 | **0.505739** [0.001416] | 33.92 | **Dense 2.03×** (2.07×)  |
| **5%**  | 95% | 950,094 | 949,727 | 1,000,000 | 1.05×  | 1.471947 [0.031121]     | 0.013414 | **0.532557** [0.014193] | 37.37 | **Dense 2.76×** (2.79×)  |

![Figure 3: Sparse vs Dense Crossover](results/plots/sparse_vs_dense_crossover.png)
*Figure 3: Execution time versus sparsity level (% of zeros). The runtime performance crossover occurs between 50% and 20% sparsity (at ≈ 46.1%–47.6% sparsity).*

![Figure 4: Fill-in Analysis](results/plots/sparse_fillin_analysis.png)
*Figure 4: Non-zero count comparison between input matrix A and output matrix C, illustrating severe arithmetic fill-in.*

---

## 5. Discussion

### 5.1 Dense Optimisations: Stride-1 Locality, L3 Spill, and L1 Cache Blocking
Separating the effects of dense loop optimisations from sparse representation effects validates our architectural hypotheses (Table 1, Figures 1–2):
- **Loop Interchange (`i-k-j`):** Achieves substantial speedups across all matrix dimensions: 2.09× at n = 250, 2.67× at n = 500, 3.02× at n = 750, 3.14× at n = 1000, 2.34× at n = 1500, and 6.86× at n = 2000. Eliminating the stride-n column access on matrix B allows CPU hardware prefetchers to stream contiguous 64-byte cache lines and enables GCC (`-O3`) to auto-vectorise the innermost j loop using AVX2 instructions [1].
- **L3 Cache Spill and Cache Blocking (`Bs = 32`):** Between n = 1000 and n = 2000, theoretical arithmetic work grows by `(2000 / 1000)³ = 8.0×`, but the canonical `i-j-k` runtime increases by **22.23×** (from 1.947 s to 43.293 s). This super-linear degradation occurs because at n = 2000, a single dense matrix requires 32 MB (total process RSS = 95.62 MB), overflowing the processor's 12 MB shared L3 cache and forcing main-memory DRAM fetches on every column traversal. While pure `i-k-j` reduces this to 6.308 s (6.86× speedup), it still streams full 16 KB rows that exceed the 48 KB L1d cache across passes. Cache blocking (`Bs = 32`) confines active working tiles to 24.0 KiB inside L1d [4], outperforming baseline by **3.72× at n = 1000** (0.524 s vs 1.947 s), **3.67× at n = 1500** (2.642 s vs 9.684 s, 1.57× faster than `i-k-j`), and **10.33× at n = 2000** (4.192 s vs 43.293 s, 1.50× faster than `i-k-j`). Furthermore, tiling exhibits remarkably low run-to-run variability at n = 2000 (IQR = 0.006176 s vs 0.696529 s for `i-k-j`), proving that L1 cache residency shields the CPU from DRAM bandwidth contention.

### 5.2 Sparse Multiplication: Conversion Costs, Memory Crossover, and the Fill-in Penalty
In Part B, comparing CSR SpGEMM against the matched dense `i-k-j` kernel (which executes a constant 2 × 10⁹ FLOPs in ≈ 0.505–0.533 s regardless of zero content) reveals three distinct regimes (Table 2, Figures 3–4):
1. **The High-Sparsity Domain (Sparsity ≥ 80%):** At 99% sparsity (1% density, NNZ(A) = 10,031), CSR SpGEMM completes in **0.001698 s** compared to 0.530988 s for matched dense multiplication—a **312.71× kernel speedup**. Even when adding the dense-to-CSR conversion cost (0.004971 s), the total execution time (0.006669 s) remains **79.62× faster** than dense multiplication, while process memory drops from 26.89 MB to **5.73 MB** (a 4.69× reduction). Strong advantages also hold at 95% sparsity (13.88× kernel / 11.56× with conversion), 90% sparsity (10.22× / 8.47×), and 80% sparsity (4.92× / 4.24×).
2. **Catastrophic Fill-in and the 50% Sparsity Boundary:** Uniformly distributed non-zeros trigger severe arithmetic fill-in (Figure 4) [2, 3]. At 95% sparsity (5% input density), output non-zeros expand by **18.41×** to 918,582 entries (91.86% dense). By 80% sparsity, the output matrix C is 100% full (1,000,000 entries). At 50% sparsity, storing A (500,051 NNZ), B (500,351 NNZ), and the 100% dense output C (1,000,000 NNZ) in CSR format requires `12 × 2,000,402 + 12 × 1001 bytes ≈ 22.90 MiB` (measured process RSS = **27.09 MB**), which **crosses over** the standalone dense process footprint (**26.89 MB** RSS for three 8 MB matrices). At this 50% threshold, CSR's speed advantage shrinks to a marginal **1.17×** (0.439561 s vs 0.516296 s), or **1.10×** when conversion time (0.030609 s) is included.
3. **The Runtime Crossover and Low-Sparsity Penalty (Sparsity < 48%):** Linear interpolation between 50% and 20% sparsity locates the exact runtime crossover at **≈ 46.1% sparsity** (53.9% density) for the isolated kernel, and **≈ 47.6% sparsity** (52.4% density) when conversion cost is included. At 20% sparsity (80% density), matched dense multiplication is **2.03× faster** than CSR (0.505739 s vs 1.026605 s, or 2.07× faster including conversion). At 5% sparsity (95% density), dense multiplication is **2.76× faster** (0.532557 s vs 1.471947 s) and uses 26.89 MB vs CSR's **37.37 MB** RSS (since CSR pays 12 bytes per non-zero instead of 8 bytes). In this high-density regime, CSR performs nearly the same number of floating-point operations as dense multiplication but suffers heavy per-element overhead from indirect array indexing (`a->col_indices[p]`, `b->row_ptr[k]`), conditional branching on `marker[j]`, and dynamic buffer reallocations.

### 5.3 Threats to Validity and Limitations
- **Uniform Random Non-zero Distribution:** Test matrices used uniform random non-zero placement, which maximises output fill-in (`d_C ≈ 1 - (1 - d²)ⁿ`). Real-world sparse matrices (e.g., finite-element meshes, banded matrices, or web graphs) exhibit strong structural locality that limits fill-in and extends CSR's advantage to higher densities [3].
- **Single-Level Cache Blocking:** Block size Bs = 32 was selected statically for the 48 KB L1d cache. Multi-level blocking across L1, L2, and L3 caches alongside register micro-kernels could yield further dense speedups [4].
- **Sequential Scope:** All experiments were strictly single-threaded to isolate algorithmic and memory-hierarchy effects; parallel OpenMP or GPU execution remains an extension topic.

---

## 6. Conclusions

Within its tested experimental scope (n ∈ [250, 2000] in C11 on an Intel Core i5-1335U), this study establishes:
1. **Dense Optimisation Hierarchy:** Loop interchange (`i-k-j`) provides a 2.09×–6.86× speedup over the canonical baseline by replacing stride-n memory accesses with contiguous unit strides. Cache blocking (`Bs = 32`) provides the superior scaling architecture for large matrices (n ≥ 1000), preventing L3 cache-spill penalties to deliver a **3.72× speedup at n = 1000** and a **10.33× speedup at n = 2000** (4.192 s vs 43.293 s, outperforming unblocked `i-k-j` by 1.50×–1.57× at n ≥ 1500).
2. **Sparse Viability Boundary:** Compressed Sparse Row (CSR) SpGEMM delivers massive execution time and memory benefits under high sparsity (≥ 80% zeros), running up to **312.71× faster** (79.62× faster including conversion) and using **4.69× less memory** at 99% sparsity.
3. **The Dual Crossover Law:** For unstructured uniform matrices at n = 1000, rapid arithmetic fill-in saturates the output matrix by 80% sparsity, triggering a **memory crossover at ≈ 50% sparsity** (27.09 MB CSR vs 26.89 MB dense) and an **execution time crossover at ≈ 46%–48% sparsity**, below which dense multiplication is up to 2.76× faster. Future work should investigate how structured sparsity patterns (banded or block-sparse matrices) shift this crossover threshold.

---

## 7. Declaration of Generative AI Assistance

In compliance with the ULPGC course policy on Generative AI, the author discloses that generative AI tools (Google Antigravity and Gemini) were consulted during the preparation of this study exclusively as supporting tools for:
1. **Code Review, Debugging & Verification:** Assisting with peer-review of timing boundaries, ensuring pure dense `i-k-j` kernel parity and isolated CSR memory measurement in `sparse_benchmark.c`, removing artificial size truncation in `dense_benchmark.c`, and checking compliance against the assignment rubric.
2. **Clean Code & Refactoring:** Advising on C11 memory safety checks, consistent naming conventions, and statistical aggregation of medians and IQRs.
3. **Language Revision & Formatting:** Proofreading the technical English writing, ensuring grammatical accuracy, and formatting Markdown and LaTeX tables.

All research questions, experimental planning, architectural choices, and core implementations were designed and executed by the author. All benchmarks were compiled and run locally on physical hardware, raw measurements were collected without fabrication, all references were verified, and all scientific conclusions are defended solely by the author.

---

## 8. References and Materials

### 8.1 Reproducibility Package
All source code, data generation scripts, automated validation fixtures, raw CSV results, and plotting pipelines are available directly within this repository:
- Source code and benchmark kernels: [`src/`](src/)
- Raw experimental measurements and summary data: [`results/`](results/)
- Step-by-step reproduction instructions: [`README.md`](README.md)
- Public GitHub Repository: `https://github.com/abelcasv/Assignment02_Abel`

### 8.2 References
1. J. L. Hennessy and D. A. Patterson, *Computer Architecture: A Quantitative Approach*, 6th ed. Cambridge, MA, USA: Morgan Kaufmann, 2019.
2. F. G. Gustavson, "Two fast algorithms for sparse matrices: Multiplication and permuted transposition," *ACM Transactions on Mathematical Software (TOMS)*, vol. 4, no. 3, pp. 250–269, Sept. 1978.
3. Y. Saad, *Iterative Methods for Sparse Linear Systems*, 2nd ed. Philadelphia, PA, USA: SIAM, 2003.
4. M. S. Lam, E. E. Rothberg, and M. E. Wolf, "The cache performance and optimizations of blocked algorithms," in *Proc. 4th Int. Conf. Architectural Support for Programming Languages and Operating Systems (ASPLOS)*, Santa Clara, CA, USA, Apr. 1991, pp. 63–74.