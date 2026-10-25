#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <direct.h>
#include <windows.h>
#include <psapi.h>

#define BLOCK_SIZE 32

static const int SIZES[] = {250, 500, 750, 1000, 1500, 2000};
static const int NUM_SIZES = sizeof(SIZES) / sizeof(SIZES[0]);
static const int VALIDATION_SIZE = 4;
static const int NUM_REPETITIONS = 5;
static const double ABSOLUTE_TOLERANCE = 1e-4;

static volatile double result_sink;

static double* load_matrix(const char* filepath, int n) {
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

/* Method 1: Canonical i-j-k baseline */
static void multiply_ijk(const double* a, const double* b, double* c, int n) {
    memset(c, 0, (size_t)n * n * sizeof(double));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double s = 0.0;
            for (int k = 0; k < n; k++) {
                s += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = s;
        }
    }
}

/* Method 2: Loop interchange i-k-j (spatial locality & vectorization) */
static void multiply_ikj(const double* a, const double* b, double* c, int n) {
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

/* Method 3: Cache blocking / Tiling (block size B_s x B_s) */
static void multiply_tiled(const double* a, const double* b, double* c, int n) {
    memset(c, 0, (size_t)n * n * sizeof(double));
    for (int ii = 0; ii < n; ii += BLOCK_SIZE) {
        int i_max = (ii + BLOCK_SIZE < n) ? ii + BLOCK_SIZE : n;
        for (int kk = 0; kk < n; kk += BLOCK_SIZE) {
            int k_max = (kk + BLOCK_SIZE < n) ? kk + BLOCK_SIZE : n;
            for (int jj = 0; jj < n; jj += BLOCK_SIZE) {
                int j_max = (jj + BLOCK_SIZE < n) ? jj + BLOCK_SIZE : n;

                for (int i = ii; i < i_max; i++) {
                    for (int k = kk; k < k_max; k++) {
                        double r = a[i * n + k];
                        for (int j = jj; j < j_max; j++) {
                            c[i * n + j] += r * b[k * n + j];
                        }
                    }
                }
            }
        }
    }
}

static int matrices_are_close(const double* a, const double* b, int n) {
    for (int i = 0; i < n * n; i++) {
        if (fabs(a[i] - b[i]) > ABSOLUTE_TOLERANCE) return 0;
    }
    return 1;
}

static int validate_methods(const char* val_dir) {
    printf("=== Correctness Validation (Dense Methods) ===\n");
    int n = VALIDATION_SIZE;
    char path[1024];

    snprintf(path, sizeof(path), "%s/known_A_4.csv", val_dir);
    double* known_a = load_matrix(path, n);
    snprintf(path, sizeof(path), "%s/known_B_4.csv", val_dir);
    double* known_b = load_matrix(path, n);
    snprintf(path, sizeof(path), "%s/expected_C_4.csv", val_dir);
    double* expected_c = load_matrix(path, n);
    snprintf(path, sizeof(path), "%s/identity_4.csv", val_dir);
    double* identity = load_matrix(path, n);
    snprintf(path, sizeof(path), "%s/zero_4.csv", val_dir);
    double* zero = load_matrix(path, n);

    if (!known_a || !known_b || !expected_c || !identity || !zero) {
        fprintf(stderr, "Error loading validation matrices from: %s\n", val_dir);
        if (known_a) free(known_a);
        if (known_b) free(known_b);
        if (expected_c) free(expected_c);
        if (identity) free(identity);
        if (zero) free(zero);
        return 0;
    }

    double* res_ijk = (double*)malloc((size_t)n * n * sizeof(double));
    double* res_ikj = (double*)malloc((size_t)n * n * sizeof(double));
    double* res_tiled = (double*)malloc((size_t)n * n * sizeof(double));

    /* 1. Analytical fixture comparison */
    multiply_ijk(known_a, known_b, res_ijk, n);
    multiply_ikj(known_a, known_b, res_ikj, n);
    multiply_tiled(known_a, known_b, res_tiled, n);

    int ok_ijk_known = matrices_are_close(res_ijk, expected_c, n);
    int ok_ikj_known = matrices_are_close(res_ikj, expected_c, n);
    int ok_tiled_known = matrices_are_close(res_tiled, expected_c, n);

    /* 2. Identity matrix comparison (A * I = A) */
    multiply_ijk(known_a, identity, res_ijk, n);
    multiply_ikj(known_a, identity, res_ikj, n);
    multiply_tiled(known_a, identity, res_tiled, n);

    int ok_ijk_ident = matrices_are_close(res_ijk, known_a, n);
    int ok_ikj_ident = matrices_are_close(res_ikj, known_a, n);
    int ok_tiled_ident = matrices_are_close(res_tiled, known_a, n);

    /* 3. Zero matrix comparison (A * 0 = 0) */
    multiply_ijk(known_a, zero, res_ijk, n);
    multiply_ikj(known_a, zero, res_ikj, n);
    multiply_tiled(known_a, zero, res_tiled, n);

    int ok_ijk_zero = matrices_are_close(res_ijk, zero, n);
    int ok_ikj_zero = matrices_are_close(res_ikj, zero, n);
    int ok_tiled_zero = matrices_are_close(res_tiled, zero, n);

    int ok_ijk = ok_ijk_known && ok_ijk_ident && ok_ijk_zero;
    int ok_ikj = ok_ikj_known && ok_ikj_ident && ok_ikj_zero;
    int ok_tiled = ok_tiled_known && ok_tiled_ident && ok_tiled_zero;

    printf("  [Method 1: Baseline i-j-k]   %s (Known: %s, Identity: %s, Zero: %s)\n",
           ok_ijk ? "PASS" : "FAIL",
           ok_ijk_known ? "OK" : "FAIL", ok_ijk_ident ? "OK" : "FAIL", ok_ijk_zero ? "OK" : "FAIL");
    printf("  [Method 2: Reordered i-k-j]  %s (Known: %s, Identity: %s, Zero: %s)\n",
           ok_ikj ? "PASS" : "FAIL",
           ok_ikj_known ? "OK" : "FAIL", ok_ikj_ident ? "OK" : "FAIL", ok_ikj_zero ? "OK" : "FAIL");
    printf("  [Method 3: Tiled Blocked]    %s (Known: %s, Identity: %s, Zero: %s)\n",
           ok_tiled ? "PASS" : "FAIL",
           ok_tiled_known ? "OK" : "FAIL", ok_tiled_ident ? "OK" : "FAIL", ok_tiled_zero ? "OK" : "FAIL");

    free(known_a); free(known_b); free(expected_c); free(identity); free(zero);
    free(res_ijk); free(res_ikj); free(res_tiled);

    return ok_ijk && ok_ikj && ok_tiled;
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
    char val_dir[1024], dense_dir[1024], results_dir[1024], results_path[1024];

    snprintf(val_dir, sizeof(val_dir), "%s/data/validation", base_dir);
    snprintf(dense_dir, sizeof(dense_dir), "%s/data/dense", base_dir);
    snprintf(results_dir, sizeof(results_dir), "%s/results", base_dir);
    snprintf(results_path, sizeof(results_path), "%s/dense_results.csv", results_dir);

    _mkdir(results_dir);

    if (!validate_methods(val_dir)) {
        fprintf(stderr, "\n[FATAL] Correctness validation failed.\n");
        return EXIT_FAILURE;
    }
    printf("All dense methods validated successfully.\n\n");

    FILE* writer = fopen(results_path, "w");
    if (!writer) {
        fprintf(stderr, "Cannot open results file: %s\n", results_path);
        return EXIT_FAILURE;
    }
    fprintf(writer, "method,n,repetition,time_s,memory_mb\n");

    LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq);

    const char* method_names[] = {"baseline_ijk", "reordered_ikj", "tiled_b32"};
    typedef void (*MatMulFn)(const double*, const double*, double*, int);
    MatMulFn functions[] = {multiply_ijk, multiply_ikj, multiply_tiled};
    int num_methods = 3;

    for (int s = 0; s < NUM_SIZES; s++) {
        int n = SIZES[s];
        printf("\n================ Matrix Size n=%d ================\n", n);

        char file_a[1024], file_b[1024];
        snprintf(file_a, sizeof(file_a), "%s/A_%d.csv", dense_dir, n);
        snprintf(file_b, sizeof(file_b), "%s/B_%d.csv", dense_dir, n);

        double* a = load_matrix(file_a, n);
        double* b = load_matrix(file_b, n);
        double* c = (double*)malloc((size_t)n * n * sizeof(double));

        if (!a || !b || !c) {
            fprintf(stderr, "Error loading data for n=%d\n", n);
            if (a) free(a);
            if (b) free(b);
            if (c) free(c);
            continue;
        }

        for (int m = 0; m < num_methods; m++) {
            const char* m_name = method_names[m];
            MatMulFn fn = functions[m];

            printf("  [Method: %-14s] Warm-up... ", m_name);
            fn(a, b, c, n);
            result_sink = c[0];
            printf("Done. Running %d repetitions:\n", NUM_REPETITIONS);

            int skip_remaining = 0;
            for (int r = 1; r <= NUM_REPETITIONS; r++) {
                if (skip_remaining) break;

                LARGE_INTEGER t0, t1;
                QueryPerformanceCounter(&t0);
                fn(a, b, c, n);
                QueryPerformanceCounter(&t1);

                result_sink = c[0] + c[(size_t)n * n - 1];
                double elapsed = (double)(t1.QuadPart - t0.QuadPart) / (double)freq.QuadPart;
                double mem_mb = get_working_set_mb();

                printf("    Rep %d: %9.6f s | Memory: %.2f MB\n", r, elapsed, mem_mb);
                fprintf(writer, "%s,%d,%d,%.6f,%.2f\n", m_name, n, r, elapsed, mem_mb);
                fflush(writer);

                if (elapsed > 120.0) {
                    printf("    [BUDGET] Execution time %.2f s exceeded 120s threshold. Truncating.\n", elapsed);
                    skip_remaining = 1;
                }
            }
        }

        free(a);
        free(b);
        free(c);
    }

    fclose(writer);
    printf("\nDense benchmark complete. Results saved to: %s\n", results_path);
    return EXIT_SUCCESS;
}