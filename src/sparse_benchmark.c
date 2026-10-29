#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#include <direct.h>
#define MKDIR(dir) _mkdir(dir)
#else
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>
#define MKDIR(dir) mkdir(dir, 0755)
#endif

#define MAX_DIR_LEN 1024
#define MAX_PATH_LEN 2048
#define LINE_BUFFER_SIZE 131072

typedef struct {
    int n;
    int nnz;
    int capacity;
    double* values;
    int* col_indices;
    int* row_ptr;
} CSRMatrix;

static const int SPARSITY_PCTS[] = {99, 95, 90, 80, 50, 20, 5};
static const int NUM_SPARSITIES = sizeof(SPARSITY_PCTS) / sizeof(SPARSITY_PCTS[0]);
static const int BENCHMARK_N = 1000;
static const int NUM_REPETITIONS = 5;
static const double ABSOLUTE_TOLERANCE = 1e-4;
static const double RELATIVE_TOLERANCE = 1e-4;
static const double PREVENTIVE_MAX_MEM_MIB = 2048.0; /* Preventive memory budget: 2.0 GiB */
static const double BUDGET_SECONDS = 120.0;          /* Reactive threshold per repetition */

static volatile double result_sink;

static int compare_doubles(const void* a, const void* b) {
    double da = *(const double*)a;
    double db = *(const double*)b;
    if (da < db) return -1;
    if (da > db) return 1;
    return 0;
}

static double get_time_seconds(void) {
#ifdef _WIN32
    static LARGE_INTEGER freq;
    static int initialized = 0;
    if (!initialized) {
        QueryPerformanceFrequency(&freq);
        initialized = 1;
    }
    LARGE_INTEGER counter;
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#endif
}

static double get_peak_memory_mib(void) {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS mem_info;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &mem_info, sizeof(mem_info))) {
        return (double)mem_info.PeakWorkingSetSize / (1024.0 * 1024.0);
    }
    return 0.0;
#else
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        #ifdef __APPLE__
        return (double)usage.ru_maxrss / (1024.0 * 1024.0);
        #else
        return (double)usage.ru_maxrss / 1024.0;
        #endif
    }
    return 0.0;
#endif
}

static CSRMatrix* create_csr_empty(int n, int max_nnz) {
    if (max_nnz < 1) max_nnz = 1;
    CSRMatrix* mat = (CSRMatrix*)malloc(sizeof(CSRMatrix));
    if (!mat) {
        fprintf(stderr, "[ERROR] Memory allocation failed for CSR header.\n");
        return NULL;
    }

    mat->n = n;
    mat->nnz = 0;
    mat->capacity = max_nnz;
    mat->row_ptr = (int*)calloc((size_t)n + 1, sizeof(int));
    if (!mat->row_ptr) {
        fprintf(stderr, "[ERROR] Memory allocation failed for row_ptr.\n");
        free(mat);
        return NULL;
    }

    mat->col_indices = (int*)malloc((size_t)max_nnz * sizeof(int));
    if (!mat->col_indices) {
        fprintf(stderr, "[ERROR] Memory allocation failed for col_indices.\n");
        free(mat->row_ptr);
        free(mat);
        return NULL;
    }

    mat->values = (double*)malloc((size_t)max_nnz * sizeof(double));
    if (!mat->values) {
        fprintf(stderr, "[ERROR] Memory allocation failed for values.\n");
        free(mat->col_indices);
        free(mat->row_ptr);
        free(mat);
        return NULL;
    }

    return mat;
}

static void free_csr(CSRMatrix* mat) {
    if (!mat) return;
    if (mat->values) free(mat->values);
    if (mat->col_indices) free(mat->col_indices);
    if (mat->row_ptr) free(mat->row_ptr);
    free(mat);
}

/* Parse a single CSV row of exactly n doubles, rejecting empty fields, trailing commas, or incorrect column counts */
static int parse_csv_row(char* line, double* row_dest, int n, int row_idx, const char* filepath) {
    size_t len = strlen(line);
    while (len > 0 && (line[len - 1] == '\r' || line[len - 1] == '\n')) {
        line[--len] = '\0';
    }
    if (len == 0) return -1;

    int col = 0;
    char* cur = line;

    while (*cur != '\0') {
        char* next_comma = strchr(cur, ',');
        if (next_comma) {
            *next_comma = '\0';
        }

        /* Check for empty field (e.g. consecutive commas or leading comma) */
        if (*cur == '\0') {
            fprintf(stderr, "[ERROR] Empty field at row %d, col %d in %s\n", row_idx, col, filepath);
            return 0;
        }

        if (col >= n) {
            fprintf(stderr, "[ERROR] Too many columns in %s: row %d has more than %d columns\n",
                    filepath, row_idx, n);
            return 0;
        }

        char* endptr = NULL;
        errno = 0;
        double val = strtod(cur, &endptr);

        if (errno == ERANGE) {
            fprintf(stderr, "[ERROR] Numerical range error (ERANGE) '%s' at (%d, %d) in %s\n",
                    cur, row_idx, col, filepath);
            return 0;
        }
        if (endptr == cur || *endptr != '\0') {
            fprintf(stderr, "[ERROR] Non-numeric entry '%s' at (%d, %d) in %s\n",
                    cur, row_idx, col, filepath);
            return 0;
        }
        if (isnan(val) || isinf(val)) {
            fprintf(stderr, "[ERROR] NaN/Inf entry at (%d, %d) in %s\n", row_idx, col, filepath);
            return 0;
        }

        row_dest[col] = val;
        col++;

        if (next_comma) {
            cur = next_comma + 1;
            /* Detect trailing comma with empty trailing field */
            if (*cur == '\0') {
                fprintf(stderr, "[ERROR] Trailing comma / empty field at end of row %d in %s\n", row_idx, filepath);
                return 0;
            }
        } else {
            break;
        }
    }

    if (col != n) {
        fprintf(stderr, "[ERROR] Column count mismatch in %s: row %d has %d columns, expected exactly %d\n",
                filepath, row_idx, col, n);
        return 0;
    }

    return 1;
}

/* Strengthened CSV Matrix Reader with comprehensive error checking */
static double* load_dense_matrix(const char* filepath, int n) {
    FILE* file = fopen(filepath, "r");
    if (!file) {
        fprintf(stderr, "[ERROR] Cannot open matrix file: %s\n", filepath);
        return NULL;
    }

    double* matrix = (double*)malloc((size_t)n * n * sizeof(double));
    if (!matrix) {
        fprintf(stderr, "[ERROR] Memory allocation failed for %dx%d matrix from %s\n", n, n, filepath);
        fclose(file);
        return NULL;
    }

    char* line = (char*)malloc(LINE_BUFFER_SIZE);
    if (!line) {
        fprintf(stderr, "[ERROR] Line buffer allocation failed for %s\n", filepath);
        free(matrix);
        fclose(file);
        return NULL;
    }

    int row = 0;
    while (fgets(line, LINE_BUFFER_SIZE, file)) {
        size_t len = strlen(line);
        if (len == 0) continue;

        /* Validate that the entire line was read without buffer truncation */
        if (line[len - 1] != '\n' && !feof(file)) {
            fprintf(stderr, "[ERROR] Line buffer overflow: row %d exceeds %d bytes in %s\n",
                    row, LINE_BUFFER_SIZE, filepath);
            free(line);
            free(matrix);
            fclose(file);
            return NULL;
        }

        /* Check for extra rows beyond n */
        if (row >= n) {
            char* p = line;
            while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') p++;
            if (*p != '\0') {
                fprintf(stderr, "[ERROR] Extra row(s) beyond %d detected in %s\n", n, filepath);
                free(line);
                free(matrix);
                fclose(file);
                return NULL;
            }
            continue;
        }

        int res = parse_csv_row(line, &matrix[row * n], n, row, filepath);
        if (res <= 0) {
            if (res < 0) {
                fprintf(stderr, "[ERROR] Unexpected empty row %d in %s\n", row, filepath);
            }
            free(line);
            free(matrix);
            fclose(file);
            return NULL;
        }
        row++;
    }

    free(line);
    fclose(file);

    if (row != n) {
        fprintf(stderr, "[ERROR] Incomplete matrix in %s: read %d rows, expected exactly %d\n",
                filepath, row, n);
        free(matrix);
        return NULL;
    }
    return matrix;
}

static CSRMatrix* dense_to_csr(const double* dense, int n) {
    if (!dense) return NULL;
    int nnz = 0;
    for (int i = 0; i < n * n; i++) {
        if (fabs(dense[i]) > 1e-12) nnz++;
    }

    CSRMatrix* csr = (CSRMatrix*)malloc(sizeof(CSRMatrix));
    if (!csr) return NULL;
    csr->n = n;
    csr->nnz = nnz;
    csr->capacity = (nnz > 0) ? nnz : 1;

    csr->row_ptr = (int*)malloc(((size_t)n + 1) * sizeof(int));
    if (!csr->row_ptr) {
        free(csr);
        return NULL;
    }

    csr->col_indices = (int*)malloc((size_t)csr->capacity * sizeof(int));
    if (!csr->col_indices) {
        free(csr->row_ptr);
        free(csr);
        return NULL;
    }

    csr->values = (double*)malloc((size_t)csr->capacity * sizeof(double));
    if (!csr->values) {
        free(csr->col_indices);
        free(csr->row_ptr);
        free(csr);
        return NULL;
    }

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
    if (!a || !b) return NULL;
    int n = a->n;

    /* Initial dynamic capacity for output C */
    int capacity = (a->nnz + b->nnz) * 2;
    if (capacity < 1024) capacity = 1024;
    if (capacity > n * n) capacity = n * n;

    CSRMatrix* c = create_csr_empty(n, capacity);
    if (!c) return NULL;

    double* dense_accum = (double*)calloc((size_t)n, sizeof(double));
    if (!dense_accum) {
        free_csr(c);
        return NULL;
    }

    int* marker = (int*)malloc((size_t)n * sizeof(int));
    if (!marker) {
        free(dense_accum);
        free_csr(c);
        return NULL;
    }
    for (int i = 0; i < n; i++) marker[i] = -1;

    int* active_cols = (int*)malloc((size_t)n * sizeof(int));
    if (!active_cols) {
        free(marker);
        free(dense_accum);
        free_csr(c);
        return NULL;
    }

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
                    int new_capacity = capacity * 2;
                    if (new_capacity > n * n) new_capacity = n * n;
                    double* new_values = (double*)realloc(c->values, (size_t)new_capacity * sizeof(double));
                    int* new_cols = (int*)realloc(c->col_indices, (size_t)new_capacity * sizeof(int));
                    if (!new_values || !new_cols) {
                        if (new_values) c->values = new_values;
                        if (new_cols) c->col_indices = new_cols;
                        free(active_cols);
                        free(marker);
                        free(dense_accum);
                        free_csr(c);
                        return NULL;
                    }
                    c->values = new_values;
                    c->col_indices = new_cols;
                    capacity = new_capacity;
                    c->capacity = capacity;
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
    if (!csr) return NULL;
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

static int matrices_are_close(const double* a, const double* b, int n) {
    for (int i = 0; i < n * n; i++) {
        double diff = fabs(a[i] - b[i]);
        double bound = ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * fabs(b[i]);
        if (diff > bound) return 0;
    }
    return 1;
}

static int validate_sparse_methods(const char* val_dir, const char* sparse_dir) {
    printf("=== Correctness Validation (Sparse CSR SpGEMM) ===\n");
    int n = 4;
    char path[MAX_PATH_LEN];
    int len;

    len = snprintf(path, sizeof(path), "%s/known_A_4.csv", val_dir);
    if (len < 0 || (size_t)len >= sizeof(path)) return 0;
    double* known_a = load_dense_matrix(path, n);

    len = snprintf(path, sizeof(path), "%s/known_B_4.csv", val_dir);
    if (len < 0 || (size_t)len >= sizeof(path)) return 0;
    double* known_b = load_dense_matrix(path, n);

    len = snprintf(path, sizeof(path), "%s/expected_C_4.csv", val_dir);
    if (len < 0 || (size_t)len >= sizeof(path)) return 0;
    double* expected_c = load_dense_matrix(path, n);

    len = snprintf(path, sizeof(path), "%s/identity_4.csv", val_dir);
    if (len < 0 || (size_t)len >= sizeof(path)) return 0;
    double* identity = load_dense_matrix(path, n);

    len = snprintf(path, sizeof(path), "%s/zero_4.csv", val_dir);
    if (len < 0 || (size_t)len >= sizeof(path)) return 0;
    double* zero = load_dense_matrix(path, n);

    if (!known_a || !known_b || !expected_c || !identity || !zero) {
        fprintf(stderr, "[ERROR] Failed loading sparse validation fixtures from %s\n", val_dir);
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

    if (!csr_a || !csr_b || !csr_ident || !csr_zero) {
        fprintf(stderr, "[ERROR] CSR conversion failed for validation fixtures.\n");
        free(known_a); free(known_b); free(expected_c); free(identity); free(zero);
        if (csr_a) free_csr(csr_a);
        if (csr_b) free_csr(csr_b);
        if (csr_ident) free_csr(csr_ident);
        if (csr_zero) free_csr(csr_zero);
        return 0;
    }

    /* 1. Analytical test (A * B == Expected C) */
    CSRMatrix* csr_prod = multiply_csr_spgemm(csr_a, csr_b);
    if (!csr_prod) {
        fprintf(stderr, "[ERROR] SpGEMM failed on analytical fixture.\n");
        return 0;
    }
    double* dense_from_csr = csr_to_dense(csr_prod);
    int ok_known = dense_from_csr ? matrices_are_close(dense_from_csr, expected_c, n) : 0;

    /* 2. Direct consistency with matched dense ikj kernel */
    double* dense_ikj_res = (double*)malloc((size_t)n * n * sizeof(double));
    if (!dense_ikj_res) return 0;
    multiply_dense_ikj(known_a, known_b, dense_ikj_res, n);
    int ok_matched = (dense_from_csr && dense_ikj_res) ? matrices_are_close(dense_from_csr, dense_ikj_res, n) : 0;

    /* 3. Identity matrix test (A * I == A) */
    CSRMatrix* csr_prod_ident = multiply_csr_spgemm(csr_a, csr_ident);
    double* dense_from_ident = csr_to_dense(csr_prod_ident);
    int ok_ident = dense_from_ident ? matrices_are_close(dense_from_ident, known_a, n) : 0;

    /* 4. Zero matrix test (A * 0 == 0) */
    CSRMatrix* csr_prod_zero = multiply_csr_spgemm(csr_a, csr_zero);
    double* dense_from_zero = csr_to_dense(csr_prod_zero);
    int ok_zero = dense_from_zero ? matrices_are_close(dense_from_zero, zero, n) : 0;

    int all_ok = ok_known && ok_matched && ok_ident && ok_zero;
    printf("  [SpGEMM Analytical]  %s\n", ok_known ? "PASS" : "FAIL");
    printf("  [SpGEMM vs Dense]    %s\n", ok_matched ? "PASS" : "FAIL");
    printf("  [SpGEMM Identity]    %s\n", ok_ident ? "PASS" : "FAIL");
    printf("  [SpGEMM Zero]        %s\n", ok_zero ? "PASS" : "FAIL");

    free(known_a); free(known_b); free(expected_c); free(identity); free(zero);
    free_csr(csr_a); free_csr(csr_b); free_csr(csr_ident); free_csr(csr_zero);
    free_csr(csr_prod); free_csr(csr_prod_ident); free_csr(csr_prod_zero);
    free(dense_from_csr); free(dense_ikj_res); free(dense_from_ident); free(dense_from_zero);

    /* 5. Benchmark Input Cross-Validation: Check n=1000 at 90% sparsity */
    int ok_bench = 1;
    if (sparse_dir) {
        char bench_file_a[MAX_PATH_LEN], bench_file_b[MAX_PATH_LEN];
        snprintf(bench_file_a, sizeof(bench_file_a), "%s/sparse_A_1000_90.csv", sparse_dir);
        snprintf(bench_file_b, sizeof(bench_file_b), "%s/sparse_B_1000_90.csv", sparse_dir);
        double* bench_a = load_dense_matrix(bench_file_a, 1000);
        double* bench_b = load_dense_matrix(bench_file_b, 1000);
        if (!bench_a || !bench_b) {
            fprintf(stderr, "[ERROR] Failed loading sparse benchmark matrices (n=1000, 90%%) for validation.\n");
            if (bench_a) free(bench_a);
            if (bench_b) free(bench_b);
            return 0;
        }

        CSRMatrix* b_csr_a = dense_to_csr(bench_a, 1000);
        CSRMatrix* b_csr_b = dense_to_csr(bench_b, 1000);
        if (!b_csr_a || !b_csr_b) {
            fprintf(stderr, "[ERROR] CSR conversion failed for n=1000 cross-validation.\n");
            if (b_csr_a) free_csr(b_csr_a);
            if (b_csr_b) free_csr(b_csr_b);
            free(bench_a); free(bench_b);
            return 0;
        }

        CSRMatrix* b_csr_c = multiply_csr_spgemm(b_csr_a, b_csr_b);
        if (!b_csr_c) {
            fprintf(stderr, "[ERROR] SpGEMM failed for n=1000 cross-validation.\n");
            free_csr(b_csr_a); free_csr(b_csr_b);
            free(bench_a); free(bench_b);
            return 0;
        }

        double* b_dense_from_csr = csr_to_dense(b_csr_c);
        double* b_dense_c = (double*)malloc((size_t)1000 * 1000 * sizeof(double));
        if (!b_dense_from_csr || !b_dense_c) {
            fprintf(stderr, "[ERROR] Allocation failed for n=1000 cross-validation dense buffers.\n");
            if (b_dense_from_csr) free(b_dense_from_csr);
            if (b_dense_c) free(b_dense_c);
            free_csr(b_csr_a); free_csr(b_csr_b); free_csr(b_csr_c);
            free(bench_a); free(bench_b);
            return 0;
        }

        multiply_dense_ikj(bench_a, bench_b, b_dense_c, 1000);
        ok_bench = matrices_are_close(b_dense_from_csr, b_dense_c, 1000);
        printf("  [Benchmark Cross-Check (n=1000, 90%% sparsity)] %s\n",
               ok_bench ? "PASS" : "FAIL");

        free(bench_a); free(bench_b); free(b_dense_c);
        free_csr(b_csr_a); free_csr(b_csr_b); free_csr(b_csr_c);
        free(b_dense_from_csr);
    }

    return all_ok && ok_bench;
}

int main(int argc, char* argv[]) {
    const char* base_dir = ".";
    int validate_only = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--validate-only") == 0) {
            validate_only = 1;
        } else if (argv[i][0] != '-') {
            base_dir = argv[i];
        }
    }

    char val_dir[MAX_DIR_LEN], sparse_dir[MAX_DIR_LEN], results_dir[MAX_DIR_LEN], results_path[MAX_PATH_LEN];

    if (snprintf(val_dir, sizeof(val_dir), "%s/data/validation", base_dir) >= (int)sizeof(val_dir) ||
        snprintf(sparse_dir, sizeof(sparse_dir), "%s/data/sparse", base_dir) >= (int)sizeof(sparse_dir) ||
        snprintf(results_dir, sizeof(results_dir), "%s/results", base_dir) >= (int)sizeof(results_dir) ||
        snprintf(results_path, sizeof(results_path), "%s/sparse_results.csv", results_dir) >= (int)sizeof(results_path)) {
        fprintf(stderr, "[ERROR] Path buffer length exceeded.\n");
        return EXIT_FAILURE;
    }

    if (!validate_sparse_methods(val_dir, sparse_dir)) {
        fprintf(stderr, "\n[FATAL] Sparse correctness validation failed.\n");
        return EXIT_FAILURE;
    }
    printf("All sparse methods validated successfully.\n\n");

    if (validate_only) {
        printf("[INFO] --validate-only specified. Exiting with success.\n");
        return EXIT_SUCCESS;
    }

    MKDIR(results_dir);

    FILE* writer = fopen(results_path, "w");
    if (!writer) {
        fprintf(stderr, "[ERROR] Cannot open results file: %s\n", results_path);
        return EXIT_FAILURE;
    }
    fprintf(writer, "approach,representation,n,sparsity_pct,density_pct,repetition,kernel_time_s,conversion_time_s,peak_memory_mib,structural_memory_mib,nnz_a,nnz_b,nnz_c,numeric_type,threads\n");

    int n = BENCHMARK_N;
    printf("================ Sparse vs Dense Benchmark (n=%d) ================\n\n", n);

    /* Preventive memory check */
    double struct_mem_dense = (3.0 * (double)n * (double)n * sizeof(double)) / (1024.0 * 1024.0);
    if (struct_mem_dense > PREVENTIVE_MAX_MEM_MIB) {
        fprintf(stderr, "[PREVENTIVE BUDGET] n=%d requires %.2f MiB, exceeding limit %.2f MiB.\n",
                n, struct_mem_dense, PREVENTIVE_MAX_MEM_MIB);
        fclose(writer);
        return EXIT_FAILURE;
    }

    for (int s = 0; s < NUM_SPARSITIES; s++) {
        int pct = SPARSITY_PCTS[s];
        int density_pct = 100 - pct;
        printf("\n>>> Testing Sparsity %d%% (Density %d%%)...\n", pct, density_pct);

        char file_a[MAX_PATH_LEN], file_b[MAX_PATH_LEN];
        snprintf(file_a, sizeof(file_a), "%s/sparse_A_%d_%d.csv", sparse_dir, n, pct);
        snprintf(file_b, sizeof(file_b), "%s/sparse_B_%d_%d.csv", sparse_dir, n, pct);

        double* dense_a = load_dense_matrix(file_a, n);
        double* dense_b = load_dense_matrix(file_b, n);
        double* dense_c = (double*)malloc((size_t)n * n * sizeof(double));

        if (!dense_a || !dense_b || !dense_c) {
            fprintf(stderr, "[FATAL] Failed loading sparse matrices for %d%%\n", pct);
            if (dense_a) free(dense_a);
            if (dense_b) free(dense_b);
            if (dense_c) free(dense_c);
            fclose(writer);
            return EXIT_FAILURE;
        }

        /* 1. Matched Dense Kernel: Execute strictly BEFORE CSR conversion to prevent memory bias */
        printf("  [Dense ikj]  Warm-up... ");
        multiply_dense_ikj(dense_a, dense_b, dense_c, n);
        double dense_warm_chk = 0.0;
        for (size_t idx = 0; idx < (size_t)n * n; idx++) {
            dense_warm_chk += dense_c[idx];
        }
        result_sink = dense_warm_chk;
        printf("Done. Running %d reps:\n", NUM_REPETITIONS);

        double dense_elapsed[NUM_REPETITIONS];
        double dense_peak_mem[NUM_REPETITIONS];

        for (int r = 1; r <= NUM_REPETITIONS; r++) {
            double t0 = get_time_seconds();
            multiply_dense_ikj(dense_a, dense_b, dense_c, n);
            double t1 = get_time_seconds();

            /* Checksum full matrix strictly outside timed window */
            double chk = 0.0;
            for (size_t idx = 0; idx < (size_t)n * n; idx++) {
                chk += dense_c[idx];
            }
            result_sink = chk;

            dense_elapsed[r - 1] = t1 - t0;
            dense_peak_mem[r - 1] = get_peak_memory_mib();

            printf("    [Dense] Rep %d: %9.6f s | Peak RSS: %.2f MiB | Struct: %.2f MiB\n",
                   r, dense_elapsed[r - 1], dense_peak_mem[r - 1], struct_mem_dense);
        }

        /* 2. Measure CSR Conversion Cost: 5 independent repetitions to obtain true distribution */
        double conv_times[NUM_REPETITIONS];
        CSRMatrix* csr_a = NULL;
        CSRMatrix* csr_b = NULL;

        for (int r = 1; r <= NUM_REPETITIONS; r++) {
            if (csr_a) free_csr(csr_a);
            if (csr_b) free_csr(csr_b);

            double conv_t0 = get_time_seconds();
            csr_a = dense_to_csr(dense_a, n);
            csr_b = dense_to_csr(dense_b, n);
            double conv_t1 = get_time_seconds();

            conv_times[r - 1] = conv_t1 - conv_t0;
        }

        if (!csr_a || !csr_b) {
            fprintf(stderr, "[FATAL] CSR conversion failed for sparsity %d%%\n", pct);
            free(dense_a); free(dense_b); free(dense_c);
            fclose(writer);
            return EXIT_FAILURE;
        }

        int nnz_a = csr_a->nnz;
        int nnz_b = csr_b->nnz;

        /* Compute real median of conversion times via sorting */
        double sorted_conv[NUM_REPETITIONS];
        memcpy(sorted_conv, conv_times, sizeof(conv_times));
        qsort(sorted_conv, NUM_REPETITIONS, sizeof(double), compare_doubles);
        double median_conv = sorted_conv[NUM_REPETITIONS / 2];

        printf("  CSR Conversion Time (5 reps): Median %.6f s | NNZ(A)=%d, NNZ(B)=%d\n",
               median_conv, nnz_a, nnz_b);

        /* 3. Free dense matrices immediately so CSR memory measurement has no dense footprint */
        free(dense_a); dense_a = NULL;
        free(dense_b); dense_b = NULL;
        free(dense_c); dense_c = NULL;

        /* Pre-compute output NNZ and capacity via warm-up of CSR SpGEMM */
        printf("  [CSR SpGEMM] Warm-up... ");
        CSRMatrix* warm_c = multiply_csr_spgemm(csr_a, csr_b);
        if (!warm_c) {
            fprintf(stderr, "[FATAL] multiply_csr_spgemm returned NULL during warm-up for %d%%\n", pct);
            free_csr(csr_a);
            free_csr(csr_b);
            fclose(writer);
            return EXIT_FAILURE;
        }
        int out_nnz = warm_c->nnz;
        int out_capacity = warm_c->capacity;

        double warm_csr_chk = 0.0;
        for (int idx = 0; idx < warm_c->nnz; idx++) warm_csr_chk += warm_c->values[idx];
        result_sink = warm_csr_chk;
        free_csr(warm_c);

        printf("Done (Output NNZ=%d, Fill-in=%.2fx).\n",
               out_nnz, (nnz_a > 0) ? ((double)out_nnz / (double)nnz_a) : 1.0);

        /* Calculate structural CSR memory: inputs A, B, output C with capacity, plus workspace */
        double struct_mem_csr = ((double)(12LL * (nnz_a + nnz_b + out_capacity) + 12LL * (n + 1) + 16LL * n)) / (1024.0 * 1024.0);

        /* Record buffered dense results now that NNZ counts and structural sizes are known */
        for (int r = 1; r <= NUM_REPETITIONS; r++) {
            fprintf(writer, "dense_matched,dense_row_major,%d,%d,%d,%d,%.6f,0.000000,%.2f,%.2f,%d,%d,%d,double,1\n",
                    n, pct, density_pct, r, dense_elapsed[r - 1], dense_peak_mem[r - 1], struct_mem_dense,
                    nnz_a, nnz_b, out_nnz);
            fflush(writer);
        }

        /* 4. Sparse CSR Kernel */
        printf("  [CSR SpGEMM] Running %d reps:\n", NUM_REPETITIONS);
        for (int r = 1; r <= NUM_REPETITIONS; r++) {
            double t0 = get_time_seconds();
            CSRMatrix* c_csr = multiply_csr_spgemm(csr_a, csr_b);
            double t1 = get_time_seconds();

            if (!c_csr) {
                fprintf(stderr, "[FATAL] multiply_csr_spgemm returned NULL during repetition %d for %d%%\n", r, pct);
                free_csr(csr_a);
                free_csr(csr_b);
                fclose(writer);
                return EXIT_FAILURE;
            }

            /* Checksum full sparse result strictly outside timed window */
            double csr_chk = 0.0;
            for (int idx = 0; idx < c_csr->nnz; idx++) {
                csr_chk += c_csr->values[idx] + (double)c_csr->col_indices[idx];
            }
            for (int idx = 0; idx <= c_csr->n; idx++) {
                csr_chk += (double)c_csr->row_ptr[idx];
            }
            result_sink = csr_chk;

            double elapsed = t1 - t0;
            double peak_mem_mib = get_peak_memory_mib();
            int c_nnz = c_csr->nnz;

            printf("    [CSR]   Rep %d: %9.6f s | Conv: %.6f s | Peak RSS: %.2f MiB | Struct: %.2f MiB\n",
                   r, elapsed, conv_times[r - 1], peak_mem_mib, struct_mem_csr);
            fprintf(writer, "sparse_csr,CSR,%d,%d,%d,%d,%.6f,%.6f,%.2f,%.2f,%d,%d,%d,double,1\n",
                    n, pct, density_pct, r, elapsed, conv_times[r - 1], peak_mem_mib, struct_mem_csr,
                    nnz_a, nnz_b, c_nnz);
            fflush(writer);
            free_csr(c_csr);

            if (elapsed > BUDGET_SECONDS) {
                printf("    [REACTIVE BUDGET EXCEEDED] Repetition %d elapsed time %.2f s exceeded declared reactive threshold (%.0f s).\n",
                       r, elapsed, BUDGET_SECONDS);
                break;
            }
        }

        free_csr(csr_a);
        free_csr(csr_b);
    }

    fclose(writer);
    printf("\nSparse benchmark complete. Results saved to: %s\n", results_path);
    return EXIT_SUCCESS;
}
