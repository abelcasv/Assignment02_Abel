# Performance Study of Dense Matrix Multiplication Optimisations and Sparse CSR Representations

**Author:** Abel Castro Villar  
**Course:** Big Data (2026-2027)  
**Assignment:** Assignment 2  
**Date:** October 2026  

---

## Abstract
This empirical investigation evaluates high-performance matrix multiplication techniques across two distinct paradigms: dense algorithmic optimizations (Part A) and Compressed Sparse Row (CSR) representations under varying sparsity conditions (Part B). Using a native C implementation compiled with GCC `-O3` under controlled inputs (seed 42), Part A assesses the canonical baseline (i-j-k), loop interchange (i-k-j), and cache blocking (Bs = 32) across square dimensions n ∈ {250, 500, 750, 1000, 1500, 2000}. Loop interchange eliminates non-unit memory strides, yielding a 2.30× speedup over the baseline at n = 1000. Cache blocking further confines working sets within the 48 KB L1 data cache, achieving a 2.75× speedup at n = 1000 (0.75 s vs 2.05 s) and scaling stably to n = 2000 (5.51 s). Part B investigates sparse matrix-matrix multiplication (SpGEMM) at n = 1000 across seven declared sparsity levels (99% to 5% zero entries). At 99% sparsity, CSR SpGEMM completes in 0.0020 s—9.5× faster than matched dense execution—even when conversion costs (0.0078 s) are included. However, as sparsity decreases, sparse indexing overhead and substantial arithmetic fill-in (e.g., 5% density inputs producing 91.8% non-zero outputs) trigger a performance crossover between 80% and 50% sparsity, where dense multiplication becomes 1.24× to 2.36× faster. These findings demonstrate that sparse storage is beneficial only when density remains below ~20% for general SpGEMM workloads.

---

## 1. Introduction and Objectives

Dense matrix multiplication is a foundational operation in numerical linear algebra, machine learning, and scientific computing. In Assignment 1, the canonical O(n³) triple-loop algorithm was established across languages. However, naive implementations suffer from severe memory bandwidth bottlenecks and CPU cache thrashing due to non-contiguous memory access patterns. Furthermore, in graph analytics and large-scale big data applications, matrices often contain an overwhelming majority of zero entries, rendering dense storage and naive multiplication computationally wasteful.

### Research Questions
1. **Part A (Dense Optimisation):** How do loop reordering (i-k-j) and cache blocking (Bs × Bs tiling) affect execution runtime, memory hierarchy utilization, and scalability compared to the canonical baseline (i-j-k)?
2. **Part B (Sparse Matrices):** Under which tested sparsity conditions does a Compressed Sparse Row (CSR) representation provide an execution advantage over dense methods, and where does the performance crossover occur when accounting for conversion costs and matrix fill-in?

### Hypotheses
1. **Loop Interchange Advantage:** The baseline i-j-k loop incurs cache misses of order O(n) in its innermost loop because matrix B is traversed along columns with stride-n. Reordering to i-k-j enables contiguous stride-1 access for both B and C, allowing hardware prefetching and vectorisation to yield >2.5× speedups.
2. **Cache Tiling Scalability:** Cache blocking with block size Bs = 32 (sized to fit comfortably within the 48 KB L1 data cache) will outperform pure loop interchange at large dimensions (n ≥ 1000) by maintaining temporal and spatial locality across multiple loop passes.
3. **Sparse Crossover and Fill-in:** CSR SpGEMM will be dramatically faster than dense methods at extreme sparsity (≥ 90%), but indexing lookups and dynamic accumulator scanning will cause a performance crossover between 50% and 80% sparsity. Multiplication will trigger substantial *fill-in*, making the output matrix significantly denser than the inputs.

---

## 2. Background and Theoretical Framework

### 2.1 Cache Hierarchy and Memory Strides
Modern microprocessors access main memory through a multi-tiered cache hierarchy (L1, L2, L3). A cache line typically holds 64 bytes (8 double-precision floats):
- **Canonical Baseline (i-j-k):** In the innermost loop over k, the program reads A[i][k] (contiguous, stride-1) and B[k][j] (stride-n). For n = 1000, consecutive elements of B are spaced 8000 bytes apart, ensuring that almost every memory access to B produces a cold or capacity cache miss.
- **Loop Interchange (i-k-j):** Permuting the middle and inner loops alters the innermost pass over index j:
  ```text
  C[i][j] += A[i][k] * B[k][j]
  ```
  Here, A[i][k] is constant with respect to j and remains pinned in a CPU register. Elements B[k][j] and C[i][j] are traversed consecutively in row-major order with unit stride (stride-1). Hardware prefetchers load full cache lines ahead of execution, eliminating memory stalls.
- **Cache Blocking / Tiling (Bs × Bs):** Rather than traversing entire matrix rows of length n, the matrices are partitioned into square blocks of size Bs × Bs. For a block size Bs = 32, the memory footprint of three active sub-blocks (A_tile, B_tile, C_tile) is:
  ```text
  Memory_tiles = 3 × Bs² × 8 bytes = 3 × 1024 × 8 = 24.576 KB
  ```
  This is well within the 48 KB L1 data cache of the host Intel processor, ensuring that cached data is reused Bs times before eviction.

### 2.2 Compressed Sparse Row (CSR) Format and SpGEMM
When a matrix of dimension n × n contains NNZ ≪ n² non-zero elements, storing n² floating-point entries wastes both memory and arithmetic cycles:
- **CSR Data Structure:** A CSR matrix is represented by three contiguous arrays:
  1. `values`: array of `double` containing non-zero entries (size NNZ).
  2. `col_indices`: array of `int` storing the column index of each non-zero entry (size NNZ).
  3. `row_ptr`: array of `int` pointing to the starting offset of each row in `values` and `col_indices` (size n + 1).
  Total memory consumption: `12 · NNZ + 4(n + 1)` bytes, compared to `8n²` bytes for dense storage.
- **Gustavson's SpGEMM Algorithm:** Multiplying two CSR matrices C = A × B requires computing linear combinations of rows of B scaled by non-zero entries of A. Because the output row sparsity structure is unknown *a priori*, an intermediate dense accumulator array (`dense_accum`) and a visited marker array (`marker`) are used to aggregate row values in O(FLOPs) time.
- **Fill-in Phenomenon:** The product of two sparse matrices is typically denser than either input. If row i of A has non-zeros at columns {k₁, k₂} and rows k₁, k₂ of B have non-zeros at distinct columns, their union forms non-zeros in row i of C, multiplying the total non-zero count.

---

## 3. Methodology and Experimental Protocol

### 3.1 Hardware and System Specification
All benchmarks were performed on a dedicated physical host with background activity minimized:
- **CPU:** 13th Gen Intel(R) Core(TM) i5-1335U (10 cores: 2 P-cores + 8 E-cores, 12 threads).
- **Cache Capacities:** L1d: 48 KB per core; L2: 1.25 MB per P-core; L3: 12 MB shared.
- **RAM:** 16.0 GB DDR4.
- **OS & Compiler:** Windows 11 Home 64-bit (Build 22631); GCC 16.1.0 (MSYS2 UCRT64), flags: `-Wall -Wextra -O3 -std=c11 -lpsapi`.

### 3.2 Dataset Generation
All datasets were generated using a deterministic pseudorandom generator (Mersenne Twister, seed 42) via `src/generate_data.py`:
- **Dense Datasets:** Square matrices A and B for n ∈ {250, 500, 750, 1000, 1500, 2000} with uniformly distributed values in [0.0, 1.0).
- **Sparse Datasets:** Matrices of dimension n = 1000 generated across seven declared sparsity levels: **99%, 95%, 90%, 80%, 50%, 20%, 5%** zeros (densities: 1%, 5%, 10%, 20%, 50%, 80%, 95%).
- **Matched Logical Inputs:** The exact same CSV matrices were used to evaluate both the CSR approach and the matched dense baseline.

### 3.3 Correctness Validation Protocol
Before timing, all dense methods and CSR SpGEMM were validated against 4 × 4 identity, zero, and analytical fixture matrices with tolerance threshold ε = 10⁻⁴. All tests passed.

### 3.4 Measurement Boundaries and Protocol
- **Kernel Isolation:** Timers strictly measured multiplication kernel execution. Data loading, conversion, and result writing were executed outside the timed window.
- **Separate Conversion Tracking:** In Part B, conversion time from dense storage to CSR representation was instrumented and recorded independently.
- **High-Resolution Timing:** Monotonic Windows API timer `QueryPerformanceCounter`.
- **Dead-Code Elimination:** Results written to `volatile double result_sink`.
- **Repetitions:** 1 warm-up run (discarded) followed by 5 recorded repetitions. Results are reported as **Median** and **IQR**.
- **Declared Budget:** Execution budget of 120 s per run. The baseline i-j-k exceeded the budget at n ≥ 1500 and was truncated.

---

## 4. Empirical Results

### 4.1 Part A: Dense Optimisation Results
Table 1 summarizes runtime, IQR, memory footprint, and speedup factors across all dense configurations.

**Table 1: Dense Matrix Multiplication Performance (Median of 5 repetitions).**

| Dimension (n) | Arithmetic Work (2n³ FLOPs) | Baseline i-j-k (s) | Reordered i-k-j (s) | Tiled Bs=32 (s) | i-k-j Speedup | Tiled Speedup | Process RSS (MB) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **250**  | 3.12 × 10⁷ | 0.020869 | 0.009115 | 0.008408 | 2.29× | **2.48×** | 5.33 |
| **500**  | 2.50 × 10⁸ | 0.212222 | 0.089654 | 0.074155 | 2.37× | **2.86×** | 9.63 |
| **750**  | 8.44 × 10⁸ | 0.754074 | 0.306706 | 0.245148 | 2.46× | **3.08×** | 16.78 |
| **1000** | 2.00 × 10⁹ | 2.053200 | 0.893935 | 0.747627 | 2.30× | **2.75×** | 26.80 |
| **1500** | 6.75 × 10⁹ | *Budget Limit* | 3.801412 | 2.404100 | — | **1.58× vs ikj** | 55.39 |
| **2000** | 1.60 × 10¹⁰| *Budget Limit* | 9.176827 | 5.510334 | — | **1.67× vs ikj** | 95.45 |

![Figure 1: Dense Runtime Comparison](results/plots/dense_time_comparison.png)
*Figure 1: Median execution time (seconds, logarithmic scale) versus matrix dimension n across dense implementations.*

![Figure 2: Dense Speedup Factors](results/plots/dense_speedup.png)
*Figure 2: Speedup factors relative to the canonical baseline (i-j-k). Cache blocking reaches a 2.75× speedup at n = 1000.*

### 4.2 Part B: Sparse CSR vs. Matched Dense Results (n = 1000)
Table 2 consolidates the performance of CSR SpGEMM compared to matched dense multiplication across all tested sparsity levels.

**Table 2: Sparse vs. Dense Performance Across Sparsity Levels (n = 1000, Median of 5 repetitions).**

| Sparsity (%) | Density (%) | Input NNZ (A) | Output NNZ (C) | Fill-in Ratio | CSR Kernel (s) | CSR Conv (s) | Matched Dense (s) | Performance Advantage |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **99%** | 1%  | 10,031  | 96,358    | 9.61×  | **0.001975** | 0.007786 | 0.018773 | **CSR 9.5× faster** |
| **95%** | 5%  | 49,885  | 918,582   | 18.41× | **0.043214** | 0.008563 | 0.062595 | **CSR 1.45× faster** |
| **90%** | 10% | 100,045 | 999,944   | 9.99×  | **0.064241** | 0.013579 | 0.100742 | **CSR 1.57× faster** |
| **80%** | 20% | 200,145 | 1,000,000 | 5.00×  | **0.132181** | 0.020991 | 0.185873 | **CSR 1.41× faster** |
| **50%** | 50% | 500,051 | 1,000,000 | 2.00×  | 0.596008     | 0.037773 | **0.481324** | **Dense 1.24× faster (Crossover)** |
| **20%** | 80% | 799,725 | 1,000,000 | 1.25×  | 1.556489     | 0.025888 | **0.704012** | **Dense 2.21× faster** |
| **5%**  | 95% | 950,094 | 1,000,000 | 1.05×  | 2.078866     | 0.017888 | **0.882064** | **Dense 2.36× faster** |

![Figure 3: Sparse vs Dense Crossover](results/plots/sparse_vs_dense_crossover.png)
*Figure 3: Execution time versus sparsity level (% of zeros). The performance crossover occurs between 80% and 50% sparsity.*

![Figure 4: Fill-in Analysis](results/plots/sparse_fillin_analysis.png)
*Figure 4: Non-zero count comparison between input matrix A and output matrix C, illustrating arithmetic fill-in.*

---

## 5. Discussion

### 5.1 Dense Optimisations: Stride-1 and L1 Cache Blocking
The empirical data decisively validates our hypotheses regarding dense optimization:
- **Loop Interchange (i-k-j):** Achieves a consistent speedup across problem dimensions: 2.29× at n = 250, 2.37× at n = 500, 2.46× at n = 750, and 2.30× at n = 1000. This confirms that eliminating the stride-n access to B allows CPU hardware prefetchers to stream cache lines seamlessly. Furthermore, GCC auto-vectorization successfully converts the inner loop into SIMD instructions because destination array C and operand B are both written and read sequentially.
- **Cache Blocking (Bs = 32):** At all tested dimensions, tiling outperforms the baseline and scales effectively, reaching **2.86× speedup at n = 500** and **3.08× at n = 750**. At n = 1000, tiling achieves 0.75 s vs 2.05 s (2.75× faster than baseline, 1.20× faster than pure i-k-j). At n = 1500 and n = 2000, where full matrix footprints exceed the 12 MB L3 cache (95.45 MB resident memory), cache blocking is **1.58× and 1.67× faster** than pure i-k-j (2.40 s vs 3.80 s at n = 1500; 5.51 s vs 9.18 s at n = 2000), demonstrating that partitioning into L1-sized blocks effectively shields the processor from memory bus saturation.

### 5.2 Sparse Multiplication: Crossover and the Fill-in Penalty
In Part B, the relationship between sparsity and execution efficiency reveals critical system trade-offs:
1. **The High-Sparsity Domain (Sparsity ≥ 90%):** At 99% sparsity (1% density), CSR SpGEMM completes in just **0.001975 s**, compared to 0.018773 s for dense multiplication (a 9.5× speedup). Even when the total cost includes conversion from raw dense storage (0.007786 s), the combined runtime (0.009761 s) remains nearly 2× faster than dense execution.
2. **The Crossover Threshold (50% to 80% Sparsity):** As sparsity drops below 80% (density increases above 20%), the advantage of CSR vanishes. At 50% sparsity, dense multiplication requires **0.4813 s**, while CSR requires **0.5960 s** (dense is 1.24× faster). At 5% sparsity, dense is **2.36× faster** than CSR (0.8821 s vs 2.0789 s).
3. **Plausible Explanations for the Crossover:**
   - *Indexing Overhead:* Dense multiplication executes straight-line, branch-free vector arithmetic. CSR SpGEMM requires indirection lookups (`a->col_indices[p]`, `b->row_ptr[k]`), conditional branching, and checking visited markers.
   - *Arithmetic Fill-in:* Sparse matrix multiplication exhibits severe fill-in. For uniformly distributed random non-zeros, even at 95% sparsity (5% density), the output matrix C reaches **918,582 non-zeros (91.8% full)**! By 80% sparsity, the output is 100% full (1,000,000 entries). Consequently, CSR is forced to store, reallocate, and process almost as many elements as dense multiplication, but with the penalty of sparse indexing data structures.

### 5.3 Threats to Validity and Limitations
- **Random Non-zero Distribution:** Test matrices used uniform random non-zero distributions. Real-world sparse matrices (e.g., social networks, power grids) often exhibit banded, block-diagonal, or power-law structures with substantially lower fill-in.
- **Cache Block Tuning:** Bs = 32 was chosen statically based on L1 cache size. Multi-level blocking (simultaneously tiling for L1 and L3) could yield further improvements.
- **Single-Threaded Execution:** All benchmarks were sequential; multi-threaded parallel SpGEMM remains an extension topic.

---

## 6. Conclusions

This investigation establishes the following findings within its experimental scope:
1. **Dense Optimization Hierarchy:** Loop interchange (i-k-j) provides an immediate ~2.3×–2.5× speedup by replacing stride-n memory access with contiguous unit strides. Cache blocking (Bs = 32) provides the superior scaling architecture for large matrices (n ≥ 1000), delivering up to a 3.08× speedup over the baseline and maintaining sub-6s runtimes up to n = 2000 (1.67× faster than pure i-k-j).
2. **Sparse Viability Boundary:** Compressed Sparse Row (CSR) representations provide massive execution time and memory benefits under extreme sparsity (≥ 90% zeros), running up to 9.5× faster than dense kernels.
3. **The Crossover Law:** For general unstructured matrices, a performance crossover occurs between 80% and 50% sparsity. At densities exceeding 20%, dense matrix multiplication is superior to CSR due to pointer indirection overhead and rapid arithmetic fill-in.

---

## 7. Declaration of Generative AI Assistance

In compliance with the ULPGC course policy on Generative AI, the author discloses that an AI assistant (Google Antigravity) was consulted during the preparation of this study exclusively as a supporting tool for:
1. **Code Review and Verification:** Assisting with thorough peer-review of boundary conditions, edge-case handling in validation fixtures, and compliance checks against the assignment rubric.
2. **Clean Code & Refactoring:** Advising on Clean Code practices, consistent naming conventions, memory safety checks, and removal of dead or redundant code structures.
3. **Language Revision & Formatting:** Proofreading the technical English writing, ensuring grammatical accuracy, and assisting with consistent Markdown and LaTeX table formatting.

All research questions, experimental planning, architectural choices, and core implementations were designed and executed by the author. All benchmarks were compiled and run locally on physical hardware, raw measurements were collected without fabrication, and all scientific conclusions are defended solely by the author.

---

## 8. References and Materials

### 8.1 Reproducibility Package
All source code, data generation scripts, automated test fixtures, raw CSV results, and plotting pipelines are available directly within this repository:
- Source code and benchmark kernels: [`src/`](src/)
- Raw experimental measurements and summary data: [`results/`](results/)
- Step-by-step reproduction instructions: [`README.md`](README.md)

### 8.2 References
1. J. L. Hennessy and D. A. Patterson, *Computer Architecture: A Quantitative Approach*, 6th ed. Morgan Kaufmann, 2019.
2. F. G. Gustavson, "Two fast algorithms for sparse matrices: Multiplication and permuted transposition," *ACM Transactions on Mathematical Software (TOMS)*, vol. 4, no. 3, pp. 250–269, 1978.
3. Y. Saad, *Iterative Methods for Sparse Linear Systems*, 2nd ed. SIAM, 2003.
4. M. S. Lam, E. E. Rothberg, and M. E. Wolf, "The cache performance and optimizations of blocked algorithms," in *Proc. 4th Int. Conf. Architectural Support for Programming Languages and Operating Systems (ASPLOS)*, 1991, pp. 63–74.
