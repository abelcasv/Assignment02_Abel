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

#define BLOCK_SIZE 32
#define MAX_DIR_LEN 1024
#define MAX_PATH_LEN 2048
#define LINE_BUFFER_SIZE 131072

static const int SIZES[] = {250, 500, 750, 1000, 1500, 2000};
static const int NUM_SIZES = sizeof(SIZES) / sizeof(SIZES[0]);
static const int VALIDATION_SIZE = 4;
static const int NUM_REPETITIONS = 5;
static const double ABSOLUTE_TOLERANCE = 1e-4;
static const double RELATIVE_TOLERANCE = 1e-4;
static const double PREVENTIVE_MAX_MEM_MIB = 2048.0; /* Preventive memory budget: 2.0 GiB */
static const double BUDGET_SECONDS = 120.0;          /* Reactive threshold per repetition */

static volatile double result_sink;

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
static double* load_matrix(const char* filepath, int n) {
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
        double diff = fabs(a[i] - b[i]);
        double bound = ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * fabs(b[i]);
        if (diff > bound) return 0;
    }
    return 1;
}

static int validate_methods(const char* val_dir, const char* dense_dir) {
    printf("=== Correctness Validation (Dense Methods) ===\n");
    int n = VALIDATION_SIZE;
    char path[MAX_PATH_LEN];
    int len;

    len = snprintf(path, sizeof(path), "%s/known_A_4.csv", val_dir);
    if (len < 0 || (size_t)len >= sizeof(path)) return 0;
    double* known_a = load_matrix(path, n);

    len = snprintf(path, sizeof(path), "%s/known_B_4.csv", val_dir);
    if (len < 0 || (size_t)len >= sizeof(path)) return 0;
    double* known_b = load_matrix(path, n);

    len = snprintf(path, sizeof(path), "%s/expected_C_4.csv", val_dir);
    if (len < 0 || (size_t)len >= sizeof(path)) return 0;
    double* expected_c = load_matrix(path, n);

    len = snprintf(path, sizeof(path), "%s/identity_4.csv", val_dir);
    if (len < 0 || (size_t)len >= sizeof(path)) return 0;
    double* identity = load_matrix(path, n);

    len = snprintf(path, sizeof(path), "%s/zero_4.csv", val_dir);
    if (len < 0 || (size_t)len >= sizeof(path)) return 0;
    double* zero = load_matrix(path, n);

    if (!known_a || !known_b || !expected_c || !identity || !zero) {
        fprintf(stderr, "[ERROR] Failed loading validation fixtures from %s\n", val_dir);
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

    if (!res_ijk || !res_ikj || !res_tiled) {
        fprintf(stderr, "[ERROR] Memory allocation failed for validation result buffers.\n");
        if (res_ijk) free(res_ijk);
        if (res_ikj) free(res_ikj);
        if (res_tiled) free(res_tiled);
        free(known_a); free(known_b); free(expected_c); free(identity); free(zero);
        return 0;
    }

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

    /* 4. Cross-validation on full benchmark input (n = 250) */
    int ok_cross = 1;
    if (dense_dir) {
        char file_a[MAX_PATH_LEN], file_b[MAX_PATH_LEN];
        snprintf(file_a, sizeof(file_a), "%s/A_250.csv", dense_dir);
        snprintf(file_b, sizeof(file_b), "%s/B_250.csv", dense_dir);
        double* bench_a = load_matrix(file_a, 250);
        double* bench_b = load_matrix(file_b, 250);
        if (!bench_a || !bench_b) {
            fprintf(stderr, "[ERROR] Failed loading benchmark inputs A_250 / B_250 for cross-validation.\n");
            if (bench_a) free(bench_a);
            if (bench_b) free(bench_b);
            return 0;
        }

        double* ref_c = (double*)malloc((size_t)250 * 250 * sizeof(double));
        double* test_ikj = (double*)malloc((size_t)250 * 250 * sizeof(double));
        double* test_tiled = (double*)malloc((size_t)250 * 250 * sizeof(double));

        if (!ref_c || !test_ikj || !test_tiled) {
            fprintf(stderr, "[ERROR] Memory allocation failed for n=250 cross-validation buffers.\n");
            if (ref_c) free(ref_c);
            if (test_ikj) free(test_ikj);
            if (test_tiled) free(test_tiled);
            free(bench_a); free(bench_b);
            return 0;
        }

        multiply_ijk(bench_a, bench_b, ref_c, 250);
        multiply_ikj(bench_a, bench_b, test_ikj, 250);
        multiply_tiled(bench_a, bench_b, test_tiled, 250);

        int match_ikj = matrices_are_close(ref_c, test_ikj, 250);
        int match_tiled = matrices_are_close(ref_c, test_tiled, 250);
        ok_cross = match_ikj && match_tiled;

        printf("  [Benchmark Cross-Check (n=250)] %s (ikj: %s, tiled: %s)\n",
               ok_cross ? "PASS" : "FAIL",
               match_ikj ? "OK" : "FAIL", match_tiled ? "OK" : "FAIL");

        free(ref_c); free(test_ikj); free(test_tiled);
        free(bench_a); free(bench_b);
    }

    return ok_ijk && ok_ikj && ok_tiled && ok_cross;
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

    char val_dir[MAX_DIR_LEN], dense_dir[MAX_DIR_LEN], results_dir[MAX_DIR_LEN], results_path[MAX_PATH_LEN];

    if (snprintf(val_dir, sizeof(val_dir), "%s/data/validation", base_dir) >= (int)sizeof(val_dir) ||
        snprintf(dense_dir, sizeof(dense_dir), "%s/data/dense", base_dir) >= (int)sizeof(dense_dir) ||
        snprintf(results_dir, sizeof(results_dir), "%s/results", base_dir) >= (int)sizeof(results_dir) ||
        snprintf(results_path, sizeof(results_path), "%s/dense_results.csv", results_dir) >= (int)sizeof(results_path)) {
        fprintf(stderr, "[ERROR] Base directory path is too long.\n");
        return EXIT_FAILURE;
    }

    if (!validate_methods(val_dir, dense_dir)) {
        fprintf(stderr, "\n[FATAL] Correctness validation failed.\n");
        return EXIT_FAILURE;
    }
    printf("All dense methods validated successfully.\n\n");

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
    fprintf(writer, "method,n,repetition,time_s,peak_memory_mib,structural_memory_mib,numeric_type,threads\n");

    const char* method_names[] = {"baseline_ijk", "reordered_ikj", "tiled_b32"};
    typedef void (*MatMulFn)(const double*, const double*, double*, int);
    MatMulFn functions[] = {multiply_ijk, multiply_ikj, multiply_tiled};
    int num_methods = 3;

    for (int s = 0; s < NUM_SIZES; s++) {
        int n = SIZES[s];
        printf("\n================ Matrix Size n=%d ================\n", n);

        /* Preventive memory check before allocating large buffers */
        double structural_mem_mib = (3.0 * (double)n * (double)n * sizeof(double)) / (1024.0 * 1024.0);
        if (structural_mem_mib > PREVENTIVE_MAX_MEM_MIB) {
            printf("[PREVENTIVE BUDGET] Dimension n=%d requires %.2f MiB structural memory, which exceeds declared preventive limit (%.2f MiB). Skipping.\n",
                   n, structural_mem_mib, PREVENTIVE_MAX_MEM_MIB);
            continue;
        }

        char file_a[MAX_PATH_LEN], file_b[MAX_PATH_LEN];
        snprintf(file_a, sizeof(file_a), "%s/A_%d.csv", dense_dir, n);
        snprintf(file_b, sizeof(file_b), "%s/B_%d.csv", dense_dir, n);

        double* a = load_matrix(file_a, n);
        double* b = load_matrix(file_b, n);
        double* c = (double*)malloc((size_t)n * n * sizeof(double));

        if (!a || !b || !c) {
            fprintf(stderr, "[FATAL] Failed loading data or allocating buffers for n=%d\n", n);
            if (a) free(a);
            if (b) free(b);
            if (c) free(c);
            fclose(writer);
            return EXIT_FAILURE;
        }

        for (int m = 0; m < num_methods; m++) {
            const char* m_name = method_names[m];
            MatMulFn fn = functions[m];

            printf("  [Method: %-14s] Warm-up... ", m_name);
            fn(a, b, c, n);

            /* Full matrix checksum outside timing region to consume result */
            double warm_chk = 0.0;
            for (size_t idx = 0; idx < (size_t)n * n; idx++) {
                warm_chk += c[idx];
            }
            result_sink = warm_chk;
            printf("Done. Running %d repetitions:\n", NUM_REPETITIONS);

            int skip_remaining = 0;
            for (int r = 1; r <= NUM_REPETITIONS; r++) {
                if (skip_remaining) break;

                double t0 = get_time_seconds();
                fn(a, b, c, n);
                double t1 = get_time_seconds();

                /* Full matrix checksum strictly outside timed kernel window */
                double chk = 0.0;
                for (size_t idx = 0; idx < (size_t)n * n; idx++) {
                    chk += c[idx];
                }
                result_sink = chk;

                double elapsed = t1 - t0;
                double peak_mem_mib = get_peak_memory_mib();

                printf("    Rep %d: %9.6f s | Peak RSS: %.2f MiB | Struct: %.2f MiB\n",
                       r, elapsed, peak_mem_mib, structural_mem_mib);
                fprintf(writer, "%s,%d,%d,%.6f,%.2f,%.2f,%s,%d\n",
                        m_name, n, r, elapsed, peak_mem_mib, structural_mem_mib, "double", 1);
                fflush(writer);

                /* Reactive time budget check */
                if (elapsed > BUDGET_SECONDS) {
                    printf("    [REACTIVE BUDGET EXCEEDED] Repetition %d elapsed time %.2f s exceeded declared reactive threshold (%.0f s). Skipping remaining repetitions for %s at n=%d.\n",
                           r, elapsed, BUDGET_SECONDS, m_name, n);
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
