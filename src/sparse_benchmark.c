#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <direct.h>
#include <windows.h>
#include <psapi.h>

typedef struct {
    int n;
    int nnz;
    double* values;
    int* col_indices;
    int* row_ptr;
} CSRMatrix;

static const int SPARSITY_PCTS[] = {99, 95, 90, 80, 50, 20, 5};
static const int NUM_SPARSITIES = sizeof(SPARSITY_PCTS) / sizeof(SPARSITY_PCTS[0]);
static const int BENCHMARK_N = 1000;
static const int NUM_REPETITIONS = 5;

static volatile double result_sink;

static CSRMatrix* create_csr_empty(int n, int max_nnz) {
    CSRMatrix* mat = (CSRMatrix*)malloc(sizeof(CSRMatrix));
    mat->n = n;
    mat->nnz = 0;
    mat->row_ptr = (int*)calloc((size_t)n + 1, sizeof(int));
    mat->col_indices = (int*)malloc((size_t)max_nnz * sizeof(int));
    mat->values = (double*)malloc((size_t)max_nnz * sizeof(double));
    return mat;
}

static void free_csr(CSRMatrix* mat) {
    if (!mat) return;
    if (mat->values) free(mat->values);
    if (mat->col_indices) free(mat->col_indices);
    if (mat->row_ptr) free(mat->row_ptr);
    free(mat);
}

static double* load_dense_matrix(const char* filepath, int n) {
    double* matrix = (double*)malloc((size_t)n * n * sizeof(double));
    if (!matrix) return NULL;

    FILE* file = fopen(filepath, "r");
    if (!file) {
        free(matrix);
        return NULL;
    }

    char line[65536];
    int row = 0;
    while (fgets(line, sizeof(line), file) && row < n) {
        char* token = strtok(line, ",\r\n");
        int col = 0;
        while (token && col < n) {
            matrix[row * n + col] = atof(token);
            token = strtok(NULL, ",\r\n");
            col++;
        }
        row++;
    }
    fclose(file);
    return matrix;
}

static CSRMatrix* dense_to_csr(const double* dense, int n) {
    int nnz = 0;
    for (int i = 0; i < n * n; i++) {
        if (fabs(dense[i]) > 1e-12) nnz++;
    }

    CSRMatrix* csr = (CSRMatrix*)malloc(sizeof(CSRMatrix));
    csr->n = n;
    csr->nnz = nnz;
    csr->row_ptr = (int*)malloc(((size_t)n + 1) * sizeof(int));
    csr->col_indices = (int*)malloc((size_t)nnz * sizeof(int));
    csr->values = (double*)malloc((size_t)nnz * sizeof(double));

    int idx = 0;
    csr->row_ptr[0] = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double val = dense[i * n + j];
            if (fabs(val) > 1e-12) {
                csr->values[idx] = val;
                csr->col_indices[idx] = j;
                idx++;
            }
        }
        csr->row_ptr[i + 1] = idx;
    }
    return csr;
}

/* Sparse-Sparse Multiplication (Gustavson's CSR x CSR algorithm) */
static CSRMatrix* multiply_csr_spgemm(const CSRMatrix* a, const CSRMatrix* b) {
    int n = a->n;

    /* Dynamic capacity for output C */
    int capacity = (a->nnz + b->nnz) * 2;
    if (capacity < 1024) capacity = 1024;

    CSRMatrix* c = create_csr_empty(n, capacity);

    double* dense_accum = (double*)calloc((size_t)n, sizeof(double));
    int* marker = (int*)malloc((size_t)n * sizeof(int));
    for (int i = 0; i < n; i++) marker[i] = -1;

    int* active_cols = (int*)malloc((size_t)n * sizeof(int));
    int c_nnz = 0;
    c->row_ptr[0] = 0;

    for (int i = 0; i < n; i++) {
        int num_active = 0;
        int a_start = a->row_ptr[i];
        int a_end = a->row_ptr[i + 1];

        for (int p = a_start; p < a_end; p++) {
            int k = a->col_indices[p];
            double val_a = a->values[p];

            int b_start = b->row_ptr[k];
            int b_end = b->row_ptr[k + 1];

            for (int q = b_start; q < b_end; q++) {
                int j = b->col_indices[q];
                double val_b = b->values[q];

                if (marker[j] != i) {
                    marker[j] = i;
                    active_cols[num_active++] = j;
                    dense_accum[j] = val_a * val_b;
                } else {
                    dense_accum[j] += val_a * val_b;
                }
            }
        }

        /* Append active nonzeros to C */
        for (int m = 0; m < num_active; m++) {
            int j = active_cols[m];
            double val = dense_accum[j];
            if (fabs(val) > 1e-12) {
                if (c_nnz >= capacity) {
                    capacity *= 2;
                    c->values = (double*)realloc(c->values, (size_t)capacity * sizeof(double));
                    c->col_indices = (int*)realloc(c->col_indices, (size_t)capacity * sizeof(int));
                }
                c->values[c_nnz] = val;
                c->col_indices[c_nnz] = j;
                c_nnz++;
            }
            dense_accum[j] = 0.0;
        }
        c->row_ptr[i + 1] = c_nnz;
    }

    c->nnz = c_nnz;
    free(dense_accum);
    free(marker);
    free(active_cols);
    return c;
}

/* Pure Dense i-k-j multiplication kernel (identical to Part A reordered_ikj) */
static void multiply_dense_ikj(const double* a, const double* b, double* c, int n) {
    memset(c, 0, (size_t)n * n * sizeof(double));
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            double r = a[i * n + k];
            for (int j = 0; j < n; j++) {
                c[i * n + j] += r * b[k * n + j];
            }
        }
    }
}

static double* csr_to_dense(const CSRMatrix* csr) {
    int n = csr->n;
    double* dense = (double*)calloc((size_t)n * n, sizeof(double));
    if (!dense) return NULL;
    for (int i = 0; i < n; i++) {
        int start = csr->row_ptr[i];
        int end = csr->row_ptr[i + 1];
        for (int p = start; p < end; p++) {
            int j = csr->col_indices[p];
            dense[i * n + j] = csr->values[p];
        }
    }
    return dense;
}

static int matrices_are_close(const double* a, const double* b, int n, double tol) {
    for (int i = 0; i < n * n; i++) {
        if (fabs(a[i] - b[i]) > tol) return 0;
    }
    return 1;
}

static int validate_sparse_methods(const char* val_dir) {
    printf("=== Correctness Validation (Sparse CSR SpGEMM) ===\n");
    int n = 4;
    char path[1024];

    snprintf(path, sizeof(path), "%s/known_A_4.csv", val_dir);
    double* known_a = load_dense_matrix(path, n);
    snprintf(path, sizeof(path), "%s/known_B_4.csv", val_dir);
    double* known_b = load_dense_matrix(path, n);
    snprintf(path, sizeof(path), "%s/expected_C_4.csv", val_dir);
    double* expected_c = load_dense_matrix(path, n);
    snprintf(path, sizeof(path), "%s/identity_4.csv", val_dir);
    double* identity = load_dense_matrix(path, n);
    snprintf(path, sizeof(path), "%s/zero_4.csv", val_dir);
    double* zero = load_dense_matrix(path, n);

    if (!known_a || !known_b || !expected_c || !identity || !zero) {
        fprintf(stderr, "Error loading validation matrices from: %s\n", val_dir);
        if (known_a) free(known_a);
        if (known_b) free(known_b);
        if (expected_c) free(expected_c);
        if (identity) free(identity);
        if (zero) free(zero);
        return 0;
    }

    CSRMatrix* csr_a = dense_to_csr(known_a, n);
    CSRMatrix* csr_b = dense_to_csr(known_b, n);
    CSRMatrix* csr_ident = dense_to_csr(identity, n);
    CSRMatrix* csr_zero = dense_to_csr(zero, n);

    /* 1. Analytical test (A * B == Expected C) */
    CSRMatrix* csr_prod = multiply_csr_spgemm(csr_a, csr_b);
    double* dense_from_csr = csr_to_dense(csr_prod);
    int ok_known = matrices_are_close(dense_from_csr, expected_c, n, 1e-4);

    /* 2. Direct consistency with matched dense ikj kernel */
    double* dense_ikj_res = (double*)malloc((size_t)n * n * sizeof(double));
    multiply_dense_ikj(known_a, known_b, dense_ikj_res, n);
    int ok_matched = matrices_are_close(dense_from_csr, dense_ikj_res, n, 1e-4);

    /* 3. Identity matrix test (A * I == A) */
    CSRMatrix* csr_prod_ident = multiply_csr_spgemm(csr_a, csr_ident);
    double* dense_from_ident = csr_to_dense(csr_prod_ident);
    int ok_ident = matrices_are_close(dense_from_ident, known_a, n, 1e-4);

    /* 4. Zero matrix test (A * 0 == 0) */
    CSRMatrix* csr_prod_zero = multiply_csr_spgemm(csr_a, csr_zero);
    double* dense_from_zero = csr_to_dense(csr_prod_zero);
    int ok_zero = matrices_are_close(dense_from_zero, zero, n, 1e-4);

    int all_ok = ok_known && ok_matched && ok_ident && ok_zero;
    printf("  [SpGEMM Analytical]  %s\n", ok_known ? "PASS" : "FAIL");
    printf("  [SpGEMM vs Dense]    %s\n", ok_matched ? "PASS" : "FAIL");
    printf("  [SpGEMM Identity]    %s\n", ok_ident ? "PASS" : "FAIL");
    printf("  [SpGEMM Zero]        %s\n", ok_zero ? "PASS" : "FAIL");

    free(known_a); free(known_b); free(expected_c); free(identity); free(zero);
    free_csr(csr_a); free_csr(csr_b); free_csr(csr_ident); free_csr(csr_zero);
    free_csr(csr_prod); free_csr(csr_prod_ident); free_csr(csr_prod_zero);
    free(dense_from_csr); free(dense_ikj_res); free(dense_from_ident); free(dense_from_zero);

    return all_ok;
}

static double get_working_set_mb(void) {
    PROCESS_MEMORY_COUNTERS mem_info;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &mem_info, sizeof(mem_info))) {
        return mem_info.WorkingSetSize / (1024.0 * 1024.0);
    }
    return 0.0;
}

int main(int argc, char* argv[]) {
    const char* base_dir = (argc > 1) ? argv[1] : ".";
    char val_dir[1024], sparse_dir[1024], results_dir[1024], results_path[1024];

    snprintf(val_dir, sizeof(val_dir), "%s/data/validation", base_dir);
    snprintf(sparse_dir, sizeof(sparse_dir), "%s/data/sparse", base_dir);
    snprintf(results_dir, sizeof(results_dir), "%s/results", base_dir);
    snprintf(results_path, sizeof(results_path), "%s/sparse_results.csv", results_dir);

    _mkdir(results_dir);

    if (!validate_sparse_methods(val_dir)) {
        fprintf(stderr, "\n[FATAL] Sparse correctness validation failed.\n");
        return EXIT_FAILURE;
    }
    printf("All sparse methods validated successfully.\n\n");

    FILE* writer = fopen(results_path, "w");
    if (!writer) {
        fprintf(stderr, "Cannot open results file: %s\n", results_path);
        return EXIT_FAILURE;
    }
    fprintf(writer, "approach,n,sparsity_pct,density_pct,repetition,kernel_time_s,conversion_time_s,memory_mb,nnz_a,nnz_b,nnz_c\n");

    LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq);

    int n = BENCHMARK_N;
    printf("================ Sparse vs Dense Benchmark (n=%d) ================\n\n", n);

    for (int s = 0; s < NUM_SPARSITIES; s++) {
        int pct = SPARSITY_PCTS[s];
        int density_pct = 100 - pct;
        printf("\n>>> Testing Sparsity %d%% (Density %d%%)...\n", pct, density_pct);

        char file_a[1024], file_b[1024];
        snprintf(file_a, sizeof(file_a), "%s/sparse_A_%d_%d.csv", sparse_dir, n, pct);
        snprintf(file_b, sizeof(file_b), "%s/sparse_B_%d_%d.csv", sparse_dir, n, pct);

        double* dense_a = load_dense_matrix(file_a, n);
        double* dense_b = load_dense_matrix(file_b, n);
        double* dense_c = (double*)malloc((size_t)n * n * sizeof(double));

        if (!dense_a || !dense_b || !dense_c) {
            fprintf(stderr, "Error loading sparse matrices for %d%%\n", pct);
            if (dense_a) free(dense_a);
            if (dense_b) free(dense_b);
            if (dense_c) free(dense_c);
            continue;
        }

        /* Measure conversion cost separately */
        LARGE_INTEGER conv_t0, conv_t1;
        QueryPerformanceCounter(&conv_t0);
        CSRMatrix* csr_a = dense_to_csr(dense_a, n);
        CSRMatrix* csr_b = dense_to_csr(dense_b, n);
        QueryPerformanceCounter(&conv_t1);
        double conv_time = (double)(conv_t1.QuadPart - conv_t0.QuadPart) / (double)freq.QuadPart;
        int nnz_a = csr_a->nnz;
        int nnz_b = csr_b->nnz;

        printf("  CSR Conversion Time: %.6f s | NNZ(A)=%d, NNZ(B)=%d\n",
               conv_time, nnz_a, nnz_b);

        /* Pre-compute output NNZ via warm-up of CSR SpGEMM */
        printf("  [CSR SpGEMM] Warm-up... ");
        CSRMatrix* warm_c = multiply_csr_spgemm(csr_a, csr_b);
        result_sink = (warm_c->nnz > 0) ? warm_c->values[0] : 0.0;
        int out_nnz = warm_c->nnz;
        free_csr(warm_c);
        printf("Done (Output NNZ=%d, Fill-in=%.2fx).\n",
               out_nnz, (double)out_nnz / (double)nnz_a);

        /* 1. Matched Dense Kernel (measured first, before freeing dense_a/b/c) */
        printf("  [Dense ikj]  Warm-up... ");
        multiply_dense_ikj(dense_a, dense_b, dense_c, n);
        result_sink = dense_c[0];
        printf("Done. Running %d reps:\n", NUM_REPETITIONS);

        for (int r = 1; r <= NUM_REPETITIONS; r++) {
            LARGE_INTEGER t0, t1;
            QueryPerformanceCounter(&t0);
            multiply_dense_ikj(dense_a, dense_b, dense_c, n);
            QueryPerformanceCounter(&t1);

            result_sink = dense_c[0];
            double elapsed = (double)(t1.QuadPart - t0.QuadPart) / (double)freq.QuadPart;
            double mem_mb = get_working_set_mb();

            printf("    [Dense] Rep %d: %9.6f s | Mem: %.2f MB\n", r, elapsed, mem_mb);
            fprintf(writer, "dense_matched,%d,%d,%d,%d,%.6f,0.000000,%.2f,%d,%d,%d\n",
                    n, pct, density_pct, r, elapsed, mem_mb, nnz_a, nnz_b, out_nnz);
            fflush(writer);
        }

        /* Free dense matrices so CSR memory is measured without dense footprint */
        free(dense_a);
        free(dense_b);
        free(dense_c);

        /* 2. Sparse CSR Kernel */
        printf("  [CSR SpGEMM] Running %d reps:\n", NUM_REPETITIONS);
        for (int r = 1; r <= NUM_REPETITIONS; r++) {
            LARGE_INTEGER t0, t1;
            QueryPerformanceCounter(&t0);
            CSRMatrix* c_csr = multiply_csr_spgemm(csr_a, csr_b);
            QueryPerformanceCounter(&t1);

            result_sink = (c_csr->nnz > 0) ? c_csr->values[0] : 0.0;
            double elapsed = (double)(t1.QuadPart - t0.QuadPart) / (double)freq.QuadPart;
            double mem_mb = get_working_set_mb();

            printf("    [CSR]   Rep %d: %9.6f s | Mem: %.2f MB\n", r, elapsed, mem_mb);
            fprintf(writer, "sparse_csr,%d,%d,%d,%d,%.6f,%.6f,%.2f,%d,%d,%d\n",
                    n, pct, density_pct, r, elapsed, conv_time, mem_mb, nnz_a, nnz_b, c_csr->nnz);
            fflush(writer);
            free_csr(c_csr);
        }

        free_csr(csr_a);
        free_csr(csr_b);
    }

    fclose(writer);
    printf("\nSparse benchmark complete. Results saved to: %s\n", results_path);
    return EXIT_SUCCESS;
}