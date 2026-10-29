# Assignment 2 — Optimised Matrix Multiplication Approaches and Sparse Matrices

**Course:** Big Data · Academic Year 2026–2027  
**Degree:** Grado en Ciencia e Ingeniería de Datos  
**Institution:** Universidad de Las Palmas de Gran Canaria (ULPGC)  
**Student / Author:** Abel Castro Villar  

---

## 1. Research Objectives & Hypotheses

### Research Questions
1. **Part A (Dense Optimisation):** How do loop reordering (`i-k-j`) and cache blocking ($B_s \times B_s$ tiling, $B_s = 32$) affect execution runtime, run-to-run variability, cache locality, and scalability up to $n = 2000$ compared to the canonical baseline (`i-j-k`)?
2. **Part B (Sparse Matrices):** Under which tested sparsity conditions does a custom Compressed Sparse Row (CSR) representation provide a computational and memory advantage over a matched dense method, and where is the performance crossover threshold when accounting for conversion costs and matrix fill-in?

### Hypotheses
- **Dense Optimisation:** The baseline (`i-j-k`) exhibits poor cache spatial locality due to stride-$n$ traversal across column elements of matrix $B$. Loop interchange (`i-k-j`) converts inner-loop memory access into contiguous unit-stride streams, providing major speedups (>2.0×). Cache blocking ($B_s = 32$) will further enhance performance on large matrices ($n \ge 1500$) by confining sub-matrix tiles within the 48 KiB L1d data cache when full matrices exceed the 12 MiB L3 cache.
- **Sparse Crossover & Fill-in:** CSR SpGEMM will drastically outperform dense multiplication at high sparsity ($\ge 80\%$ zeros), but indexing overhead, dynamic accumulation, and severe arithmetic *fill-in* will cause a performance crossover between 50% and 20% sparsity, where dense multiplication becomes faster in the saved kernel timings. The sampled process working-set measurements do not establish which method uses less memory; a structural memory model must be used for rigorous comparison.

---

## 2. Experimental Environment & Attribution

All experiments were executed sequentially on a single thread on a physical host under controlled conditions:

| Parameter | Specification |
|---|---|
| **Operating System** | Windows 11 Home (64-bit, Build 22631) |
| **CPU Model** | 13th Gen Intel(R) Core(TM) i5-1335U |
| **Cores / Threads** | 10 physical cores (2 P-cores + 8 E-cores) / 12 threads |
| **Cache Hierarchy** | L1d: 48 KiB per core · L2: 1.25 MiB per P-core · L3: 12 MiB shared |
| **RAM** | 16.0 GB DDR4 |
| **C Compiler & Flags** | GCC 16.1.0 (MSYS2 UCRT64), flags: `-Wall -Wextra -O3 -std=c11 -lpsapi` |
| **Python Environment** | Python 3.14.0 (MSC v.1944 64-bit) |
| **Scientific Libraries** | NumPy 2.5.4 · Pandas 3.0.6 · Matplotlib 3.11.2 |
| **Random Seed & Distribution** | Seed 42 (Mersenne Twister `random.Random(42)`), Uniform $[0.0, 1.0)$ |
| **Numeric Precision** | 64-bit floating point (`double` / IEEE 754) |

> **Implementation Attribution Notice:**  
> All dense and sparse matrix multiplication kernels were written completely from scratch in standard C11. **No external linear algebra libraries** (such as BLAS, LAPACK, Intel MKL, or SuiteSparse) were used. The sparse multiplication kernel implements Gustavson's row-wise SpGEMM algorithm (1978) with custom dynamic memory management, dense accumulator vectors, and visited column markers.

---

## 3. Experimental Protocol & Memory Modeling

1. **Controlled Datasets:** Generated with fixed seed (42). Dense matrices $n \in \{250, 500, 750, 1000, 1500, 2000\}$; Sparse matrices $n = 1000$ across 7 declared sparsity levels (99%, 95%, 90%, 80%, 50%, 20%, 5% zeros).
2. **Correctness Validation:** All methods verified against $4 \times 4$ identity, zero, and analytical fixture matrices before benchmarking using mixed absolute and relative tolerance ($|a - b| \le 10^{-4} + 10^{-4} \cdot |b|$). In addition, full cross-validation against matched dense reference is executed on benchmark inputs ($n=250$ for dense, $n=1000$ at 90% sparsity for sparse).
3. **Timing Boundary:** Kernel execution only. File loading and initial memory allocations of input matrices are strictly excluded. Array zeroing (`memset`) is included in dense kernels; dynamic output reallocation (`realloc`) and workspace initialization (`marker`, `dense_accum`) are included in CSR kernels because output NNZ is unknown *a priori*.
4. **Timer & Full Output Consumption:** Monotonic timer (`QueryPerformanceCounter` on Windows, `clock_gettime(CLOCK_MONOTONIC)` on Linux). The full output matrix (all $n \times n$ entries or all CSR structural arrays) is accumulated into a volatile checksum sink strictly outside the timed window to consume the output completely and avoid compiler dead-code elimination.
5. **Budgets & Thresholds:**
   - **Preventive Memory Budget:** A 2048 MiB (2.0 GiB) structural memory check is evaluated before allocating buffers. If a configuration exceeds this threshold, it is skipped preventively.
   - **Reactive Runtime Budget:** A 120.0 s threshold is evaluated after each completed repetition. If exceeded, remaining repetitions for that method are skipped.
6. **Repetitions & Statistics:** 1 untimed warm-up run (discarded) followed by 5 recorded repetitions. Statistics reported using **Median** and **Interquartile Range (IQR)**.
7. **Conversion Costs & Memory Modeling:**
   - In Part B, CSR conversion time is measured independently across 5 repetitions. In the code, the true median is computed after sorting the array.
   - **Memory Metrics:** Process working set (`PeakWorkingSetSize` / `ru_maxrss`) is cumulative over process lifetime. To ensure scientific rigor, structural memory models are defined analytically:
     - **Dense Structural Memory:** $3 \times n^2 \times 8\text{ bytes} = 24n^2\text{ bytes}$.
     - **CSR Structural Memory:** $12\,\text{NNZ}_A + 12\,\text{NNZ}_B + 12\,\text{Capacity}_C + 12(n+1) + 16n\text{ bytes}$.

---

## 4. Empirical Results

> **Experimental Data Provenance:**  
> The measurements in `results/dense_results.csv` and `results/sparse_results.csv` correspond to the final physical benchmark execution. All values were collected using the C benchmark programs (`dense_benchmark.c` and `sparse_benchmark.c`) configured with 5 repetitions per dimension/sparsity level, full output matrix checksums outside the timed region, preventive memory checks (2048 MiB), reactive time limits (120 s), independent per-repetition CSR conversion measurements, and dual tracking of both peak process working set (RSS) and analytical structural memory.

### Part A: Dense Optimisation Summary (Median and IQR over 5 repetitions)

*Working-set values (Sampled RSS) are descriptive telemetry from a shared process history; analytical structural memory reflects theoretical memory requirements.*

| Size ($n$) | Work ($2n^3$ FLOPs) | Baseline `i-j-k` (s) [IQR] | Reordered `i-k-j` (s) [IQR] | Tiled $B_s=32$ (s) [IQR] | `i-k-j` Speedup | Tiled Speedup (vs. `i-k-j`) | Peak RSS (MiB) | Structural Mem (MiB) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **250**  | $3.12 \times 10^7$  | 0.024026 [0.000809] | 0.011429 [0.003841] | **0.013393** [0.001034] | 2.10× | **1.79×** (0.85×) | 6.23  | 1.43  |
| **500**  | $2.50 \times 10^8$  | 0.229233 [0.019189] | 0.083574 [0.006324] | **0.081358** [0.009331] | 2.74× | **2.82×** (1.03×) | 11.21 | 5.72  |
| **750**  | $8.44 \times 10^8$  | 0.883345 [0.045146] | 0.376566 [0.068781] | **0.269894** [0.027923] | 2.35× | **3.27×** (1.40×) | 18.36 | 12.87 |
| **1000** | $2.00 \times 10^9$  | 2.164377 [0.065673] | 0.798179 [0.055807] | **0.606513** [0.023022] | 2.71× | **3.57×** (1.32×) | 28.38 | 22.89 |
| **1500** | $6.75 \times 10^9$  | 8.254691 [0.503477] | 3.006954 [0.029573] | **2.065816** [0.074447] | 2.75× | **4.00×** (1.46×) | 56.97 | 51.50 |
| **2000** | $1.60 \times 10^{10}$ | 60.164282 [1.001185]| 8.198711 [2.147175] | **4.799349** [0.206364] | 7.34× | **12.54×** (1.71×)| 97.14 | 91.55 |

### Part B: Sparse vs Dense Comparison ($n = 1000$, Median and IQR over 5 repetitions)

*Conversion time reflects the median of 5 independent conversion runs. Fill-in causes $C$ to become completely dense ($\text{NNZ}=1,000,000$) at $\le 80\%$ sparsity.*

| Sparsity (%) | Density (%) | NNZ(A) | NNZ(B) | NNZ(C) | Fill-in Ratio | CSR Kernel (s) [IQR] | CSR Conv (s) | Dense Kernel (s) [IQR] | Peak RSS (MiB) | CSR Struct (MiB) | Kernel Speedup (w/ Conv) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **99%** | 1%  | 10,031  | 10,054  | 96,358    | 9.61×  | **0.001864** [0.000135] | 0.006117 | 0.805301 [0.020568] | 48.20 | 2.10  | **CSR 432.03×** (100.90×) |
| **95%** | 5%  | 49,885  | 50,249  | 918,582   | 18.41× | **0.044970** [0.004970] | 0.009094 | 0.702969 [0.033723] | 48.20 | 12.62 | **CSR 15.63×** (13.00×)   |
| **90%** | 10% | 100,045 | 99,788  | 999,944   | 9.99×  | **0.057869** [0.004052] | 0.012534 | 0.697725 [0.029357] | 48.20 | 13.76 | **CSR 12.06×** (9.91×)    |
| **80%** | 20% | 200,145 | 199,586 | 1,000,000 | 5.00×  | **0.122965** [0.003700] | 0.021152 | 0.664370 [0.014776] | 48.20 | 16.05 | **CSR 5.40×** (4.61×)     |
| **50%** | 50% | 500,051 | 500,351 | 1,000,000 | 2.00×  | **0.573003** [0.009383] | 0.033526 | 0.799045 [0.034370] | 48.20 | 22.92 | **CSR 1.39×** (1.32×)     |
| **20%** | 80% | 799,725 | 800,084 | 1,000,000 | 1.25×  | 1.377105 [0.054361]     | 0.022017 | **0.788356** [0.014740] | 48.20 | 29.78 | **Dense 1.75×** (1.77×)   |
| **5%**  | 95% | 950,094 | 949,727 | 1,000,000 | 1.05×  | 1.724761 [0.046693]     | 0.016956 | **0.679094** [0.013705] | 48.59 | 33.21 | **Dense 2.54×** (2.56×)   |

> **Crossover Summary:**  
> In the measured points, CSR outperforms dense multiplication at $\ge 50\%$ sparsity. Dense multiplication becomes faster at 20% and 5% sparsity. By linear interpolation between the 50% and 20% sparsity data points, the break-even crossover is estimated near **42% sparsity (kernel-only)** and **43% sparsity (including input conversion)**.

---

## 5. How to Reproduce

### Prerequisites
- **Python 3.10+** (tested on Python 3.14.0):
  ```bash
  pip install -r requirements.txt
  ```
- **GCC Compiler with C11 support:**
  - **Windows (MSYS2 UCRT64):**
    ```powershell
    $env:PATH = "C:\msys64\ucrt64\bin;C:\msys64\usr\bin;$env:PATH"
    gcc --version
    ```
  - **Linux (Debian/Ubuntu):**
    ```bash
    sudo apt update && sudo apt install build-essential python3 python3-pip
    ```

---

### Step 1: Generate Deterministic Datasets (Optional if `data/` already exists)
```powershell
py src/generate_data.py
# (On Linux: python3 src/generate_data.py)
```
Generates all validation fixtures (`data/validation/`), dense matrices (`data/dense/`), and sparse matrices (`data/sparse/`) with fixed seed 42.

---

### Step 2: Compile Benchmark Binaries
Using the included `Makefile`:
```bash
cd src
make clean
make
cd ..
```
Or invoking GCC directly from root:
- **Windows (MSYS2 / MinGW):**
  ```powershell
  gcc -Wall -Wextra -O3 -std=c11 src/dense_benchmark.c -o src/dense_benchmark.exe -lpsapi
  gcc -Wall -Wextra -O3 -std=c11 src/sparse_benchmark.c -o src/sparse_benchmark.exe -lpsapi
  ```
- **Linux:**
  ```bash
  gcc -Wall -Wextra -O3 -std=c11 src/dense_benchmark.c -o src/dense_benchmark -lm
  gcc -Wall -Wextra -O3 -std=c11 src/sparse_benchmark.c -o src/sparse_benchmark -lm
  ```

---

### Step 3: Run Validation Only (Fast Correctness Check)
Before launching lengthy benchmarks, run the `--validate-only` flag to verify all analytical, identity, zero, and benchmark cross-validation fixtures:
```powershell
# From project root:
.\src\dense_benchmark.exe --validate-only
.\src\sparse_benchmark.exe --validate-only

# Or from inside src/ (providing '..' as base directory):
cd src
.\dense_benchmark.exe --validate-only ..
.\sparse_benchmark.exe --validate-only ..
cd ..
```
*(On Linux, replace `.\*.exe` with `./dense_benchmark` and `./sparse_benchmark`).*

---

### Step 4: Run Full Benchmarks
```powershell
# From project root:
.\src\dense_benchmark.exe
.\src\sparse_benchmark.exe

# Or from inside src/ (providing '..' as base directory):
cd src
.\dense_benchmark.exe ..
.\sparse_benchmark.exe ..
cd ..
```
*Note: Due to Windows Defender Application Control (WDAC), running newly compiled `.exe` files should be executed in your local interactive console session.*

---

### Step 5: Run Analysis and Regenerate Summary CSVs & Plots
```powershell
# From project root:
py src/analysis/analyze_assignment02.py
# (On Linux: python3 src/analysis/analyze_assignment02.py)
```
Validates the CSV data integrity (all rows, columns, rep counts, non-negative times) and updates:
- `results/dense_summary.csv`
- `results/sparse_summary.csv`
- `results/plots/dense_time_comparison.png`
- `results/plots/dense_speedup.png`
- `results/plots/sparse_vs_dense_crossover.png`
- `results/plots/sparse_fillin_analysis.png`

---

### Step 6: LaTeX Report Compilation
The complete LaTeX report is provided in `Assignment02_Report.tex`. Compile using `pdflatex` (the bibliography is embedded directly using `thebibliography`, so running `bibtex` is neither required nor supported):
```bash
pdflatex Assignment02_Report.tex
pdflatex Assignment02_Report.tex
```
Or upload directly to Overleaf.
