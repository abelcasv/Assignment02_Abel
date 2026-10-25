# Assignment 2 — Optimised Matrix Multiplication Approaches and Sparse Matrices

**Course:** Big Data · Academic Year 2026-2027  
**Degree:** Grado en Ciencia e Ingeniería de Datos  
**Institution:** Universidad de Las Palmas de Gran Canaria  
**Student:** Abel Castro Villar  

---

## 1. Research Objectives & Hypotheses

### Research Questions
1. **Part A (Dense Optimisation):** How do loop reordering (`i-k-j`) and cache blocking ($B_s \times B_s$ tiling) affect execution time, run-to-run variability, cache locality, and scalability up to $n = 2000$ compared to the canonical baseline (`i-j-k`)?
2. **Part B (Sparse Matrices):** Under which sparsity conditions does a custom Compressed Sparse Row (CSR) representation provide a computational and memory advantage over a matched dense method, and where is the performance crossover threshold when accounting for conversion costs and matrix fill-in?

### Hypotheses
- **Dense Optimisation:** The baseline (`i-j-k`) suffers severe cache misses due to stride-$n$ access on matrix $B$. Loop interchange (`i-k-j`) converts inner-loop memory access into contiguous unit-stride streams, providing major speedups ($>2.0\times$). Cache blocking ($B_s = 32$) will further enhance performance on large matrices ($n \ge 1500$) by confining working sets within the $48\text{ KB}$ L1d data cache when full matrices exceed the $12\text{ MB}$ L3 cache.
- **Sparse Crossover & Fill-in:** CSR SpGEMM will drastically outperform dense multiplication at high sparsity ($\ge 80\%$ zeros), but indexing and dynamic accumulation overhead alongside severe output *fill-in* will cause a performance crossover between $50\%$ and $20\%$ sparsity, where dense multiplication becomes faster and more memory-efficient.

---

## 2. Experimental Environment

All experiments were executed sequentially on a dedicated physical host under controlled conditions:

| Parameter | Specification |
|---|---|
| **Operating System** | Windows 11 Home (64-bit, Build 22631) |
| **CPU Model** | 13th Gen Intel(R) Core(TM) i5-1335U |
| **Physical Cores / Threads** | 10 physical cores (2 P-cores + 8 E-cores) / 12 threads |
| **L1 / L2 / L3 Cache** | L1d: $48\text{ KB}$ per core · L2: $1.25\text{ MB}$ per P-core · L3: $12\text{ MB}$ shared |
| **RAM** | $16.0\text{ GB}$ DDR4 |
| **C Compiler & Flags** | GCC 16.1.0 (MSYS2 UCRT64), flags: `-Wall -Wextra -O3 -std=c11 -lpsapi` |
| **Python Version** | Python 3.14.0 (standard 64-bit) |
| **Random Seed & Distribution** | Seed $42$ (Mersenne Twister `random.Random(42)`), Uniform $[0.0, 1.0)$ |
| **Numeric Type** | 64-bit floating point (`double` / IEEE 754) |

---

## 3. Experimental Protocol

1. **Controlled Datasets:** Generated with fixed seed ($42$). Dense matrices $n \in \{250, 500, 750, 1000, 1500, 2000\}$; Sparse matrices $n = 1000$ across $7$ sparsity levels ($99\%, 95\%, 90\%, 80\%, 50\%, 20\%, 5\%$ zeros).
2. **Correctness Validation:** All methods verified against $4 \times 4$ identity, zero, and analytical fixture matrices before benchmarking ($\varepsilon = 10^{-4}$).
3. **Timing Boundary:** Kernel execution only. File loading and memory allocations are excluded.
4. **Timers & Sinks:** Monotonic timer (`QueryPerformanceCounter`). Computation results assigned to `volatile double result_sink` to prevent dead-code elimination.
5. **Repetitions & Statistics:** $1$ warm-up run (discarded) followed by $5$ recorded repetitions under a $120\text{ s}$ per-run budget. Reported using **Median** and **Interquartile Range (IQR)**.
6. **Conversion Costs & Memory Isolation:** In Part B, CSR conversion time from raw matrices is measured and reported separately from the multiplication kernel, and dense buffers are freed before measuring CSR resident memory.

---

## 4. Empirical Results

### Part A: Dense Optimisation Summary (Median and IQR over 5 repetitions)

| Size ($n$) | Work ($2n^3$ FLOPs) | Baseline `i-j-k` (s) [IQR] | Reordered `i-k-j` (s) [IQR] | Tiled $B_s=32$ (s) [IQR] | `i-k-j` Speedup | Tiled Speedup (vs. `i-k-j`) | Process RSS (MB) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **250**  | $3.12 \times 10^7$  | 0.019362 [0.001112] | 0.009281 [0.001178] | **0.008645** [0.000664] | $2.09\times$ | **$2.24\times$** ($1.07\times$) | 5.42  |
| **500**  | $2.50 \times 10^8$  | 0.193144 [0.003933] | 0.072269 [0.007113] | **0.066373** [0.000934] | $2.67\times$ | **$2.91\times$** ($1.09\times$) | 9.72  |
| **750**  | $8.44 \times 10^8$  | 0.716699 [0.089923] | 0.236948 [0.001982] | **0.224766** [0.007220] | $3.02\times$ | **$3.19\times$** ($1.05\times$) | 16.87 |
| **1000** | $2.00 \times 10^9$  | 1.947354 [0.123171] | 0.620054 [0.042797] | **0.523642** [0.018781] | $3.14\times$ | **$3.72\times$** ($1.18\times$) | 26.89 |
| **1500** | $6.75 \times 10^9$  | 9.684193 [0.040139] | 4.139564 [1.308037] | **2.641865** [0.160843] | $2.34\times$ | **$3.67\times$** ($1.57\times$) | 55.48 |
| **2000** | $1.60 \times 10^{10}$| 43.293093 [0.244237]| 6.308155 [0.696529] | **4.192394** [0.006176] | $6.86\times$ | **$10.33\times$** ($1.50\times$)| 95.62 |

### Part B: Sparse vs Dense Comparison ($n = 1000$, Median and IQR over 5 repetitions)

| Sparsity (%) | Density (%) | NNZ($A$) | NNZ($B$) | NNZ($C$) | Fill-in Ratio | CSR Kernel (s) [IQR] | CSR Conv (s) | Dense Kernel (s) [IQR] | CSR RSS (MB) | Kernel Speedup (w/ Conv) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **99%** | 1%  | 10,031  | 10,054  | 96,358    | $9.61\times$  | **0.001698** [0.000103] | 0.004971 | 0.530988 [0.019662] | 5.73  | **CSR $312.71\times$** ($79.62\times$) |
| **95%** | 5%  | 49,885  | 50,249  | 918,582   | $18.41\times$ | **0.037238** [0.001261] | 0.007464 | 0.516842 [0.009631] | 15.84 | **CSR $13.88\times$** ($11.56\times$)  |
| **90%** | 10% | 100,045 | 99,788  | 999,944   | $9.99\times$  | **0.051188** [0.000487] | 0.010571 | 0.523098 [0.004033] | 18.07 | **CSR $10.22\times$** ($8.47\times$)   |
| **80%** | 20% | 200,145 | 199,586 | 1,000,000 | $5.00\times$  | **0.102735** [0.002494] | 0.016465 | 0.505236 [0.008643] | 20.58 | **CSR $4.92\times$** ($4.24\times$)    |
| **50%** | 50% | 500,051 | 500,351 | 1,000,000 | $2.00\times$  | **0.439561** [0.009119] | 0.030609 | 0.516296 [0.001550] | 27.09 | **CSR $1.17\times$** ($1.10\times$)    |
| **20%** | 80% | 799,725 | 800,084 | 1,000,000 | $1.25\times$  | 1.026605 [0.016841]     | 0.018936 | **0.505739** [0.001416] | 33.92 | **Dense $2.03\times$** ($2.07\times$)  |
| **5%**  | 95% | 950,094 | 949,727 | 1,000,000 | $1.05\times$  | 1.471947 [0.031121]     | 0.013414 | **0.532557** [0.014193] | 37.37 | **Dense $2.76\times$** ($2.79\times$)  |

> **Detailed Research Report:**  
> [Assignment02_Report.md](Assignment02_Report.md) (or the compiled PDF: `Assignment02_Castro_Report.pdf`)

---

## 5. How to Reproduce

### Prerequisites
- Python 3.10+ (for dataset generation and analysis):

~~~bash
pip install pandas matplotlib numpy
~~~

- GCC Compiler with C11 support (MSYS2 UCRT64 on Windows).

> **Note for Windows PowerShell (MSYS2):** Ensure GCC is accessible in your session:

~~~powershell
$env:PATH = "C:\msys64\ucrt64\bin;$env:PATH"
~~~

### Step 1: Generate Datasets

~~~powershell
py src/generate_data.py
~~~

### Step 2: Compile Benchmark Binaries

~~~powershell
gcc -Wall -Wextra -O3 -std=c11 src/dense_benchmark.c -o src/dense_benchmark.exe -lpsapi
gcc -Wall -Wextra -O3 -std=c11 src/sparse_benchmark.c -o src/sparse_benchmark.exe -lpsapi
~~~

### Step 3: Run Dense Optimization Benchmark

~~~powershell
./src/dense_benchmark.exe .
~~~

### Step 4: Run Sparse vs Dense Benchmark

~~~powershell
./src/sparse_benchmark.exe .
~~~

### Step 5: Generate Summary Statistics and Visual Plots

~~~powershell
py src/analysis/analyze_assignment02.py
~~~

Generated metrics and plots are stored in `results/` and `results/plots/`.