"""
Dataset Generation Script for Assignment 2:
Optimised Matrix Multiplication Approaches and Sparse Matrices.

Input Generation Process & Distribution Specification:
- Deterministic Pseudorandom Number Generator: Python Standard Library `random.Random(42)`
  implementing the Mersenne Twister MT19937 algorithm with fixed seed = 42.
- Non-zero Distribution: Continuous Uniform Distribution over the half-open interval [0.0, 1.0).
- Precision: IEEE 754 64-bit double precision, written as comma-separated values formatted
  to 6 decimal places (e.g., "0.123456").
- Dense Datasets: Full square matrices for dimensions n in {250, 500, 750, 1000, 1500, 2000}.
- Sparse Datasets: Square matrices for dimension n = 1000 across 7 declared sparsity levels:
  99%, 95%, 90%, 80%, 50%, 20%, 5% zero entries (densities: 1%, 5%, 10%, 20%, 50%, 80%, 95%).
- Validation Fixtures: Deterministic 4x4 matrices (Identity, Zero, Known A, Known B, Expected C).
"""

import csv
import os
import random

DENSE_SIZES = [250, 500, 750, 1000, 1500, 2000]
SPARSE_SIZES = [1000]
SPARSITY_LEVELS = [0.99, 0.95, 0.90, 0.80, 0.50, 0.20, 0.05]
VALIDATION_SIZE = 4
SEED = 42


def write_csv_matrix(filepath: str, matrix: list[list[float]]) -> None:
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        for row in matrix:
            writer.writerow([f"{val:.6f}" for val in row])


def reference_multiply(matrix_a: list[list[float]],
                       matrix_b: list[list[float]],
                       n: int) -> list[list[float]]:
    c = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += matrix_a[i][k] * matrix_b[k][j]
            c[i][j] = s
    return c


def generate_dense_matrices(data_dir: str, rng: random.Random) -> None:
    dense_dir = os.path.join(data_dir, "dense")
    os.makedirs(dense_dir, exist_ok=True)

    for n in DENSE_SIZES:
        file_a = os.path.join(dense_dir, f"A_{n}.csv")
        file_b = os.path.join(dense_dir, f"B_{n}.csv")

        if os.path.exists(file_a) and os.path.exists(file_b):
            print(f"  Dense matrices {n}x{n} already exist, skipping.")
            continue

        print(f"  Generating dense {n}x{n} matrices...")
        matrix_a = [[rng.random() for _ in range(n)] for _ in range(n)]
        matrix_b = [[rng.random() for _ in range(n)] for _ in range(n)]

        write_csv_matrix(file_a, matrix_a)
        write_csv_matrix(file_b, matrix_b)


def generate_sparse_matrices(data_dir: str, rng: random.Random) -> None:
    sparse_dir = os.path.join(data_dir, "sparse")
    os.makedirs(sparse_dir, exist_ok=True)

    for n in SPARSE_SIZES:
        for sparsity in SPARSITY_LEVELS:
            pct = int(sparsity * 100)
            file_a = os.path.join(sparse_dir, f"sparse_A_{n}_{pct}.csv")
            file_b = os.path.join(sparse_dir, f"sparse_B_{n}_{pct}.csv")

            if os.path.exists(file_a) and os.path.exists(file_b):
                print(f"  Sparse matrices {n}x{n} (sparsity {pct}%) already exist, skipping.")
                continue

            print(f"  Generating sparse {n}x{n} matrices (sparsity {pct}%)...")
            density = 1.0 - sparsity

            matrix_a = [
                [rng.random() if rng.random() < density else 0.0 for _ in range(n)]
                for _ in range(n)
            ]
            matrix_b = [
                [rng.random() if rng.random() < density else 0.0 for _ in range(n)]
                for _ in range(n)
            ]

            write_csv_matrix(file_a, matrix_a)
            write_csv_matrix(file_b, matrix_b)


def generate_validation_matrices(data_dir: str) -> None:
    val_dir = os.path.join(data_dir, "validation")
    os.makedirs(val_dir, exist_ok=True)
    n = VALIDATION_SIZE

    identity = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    zero = [[0.0] * n for _ in range(n)]

    known_a = [
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0, 7.0],
    ]
    known_b = [
        [7.0, 6.0, 5.0, 4.0],
        [3.0, 2.0, 1.0, 9.0],
        [8.0, 7.0, 6.0, 5.0],
        [1.0, 2.0, 3.0, 4.0],
    ]
    expected_c = reference_multiply(known_a, known_b, n)

    write_csv_matrix(os.path.join(val_dir, "identity_4.csv"), identity)
    write_csv_matrix(os.path.join(val_dir, "zero_4.csv"), zero)
    write_csv_matrix(os.path.join(val_dir, "known_A_4.csv"), known_a)
    write_csv_matrix(os.path.join(val_dir, "known_B_4.csv"), known_b)
    write_csv_matrix(os.path.join(val_dir, "expected_C_4.csv"), expected_c)
    print("  Validation matrices (4x4) created.")


def main() -> None:
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    print(f"Initializing RNG with seed {SEED}...")
    rng = random.Random(SEED)

    print("\n--- 1. Validation Fixtures ---")
    generate_validation_matrices(data_dir)

    print("\n--- 2. Dense Datasets ---")
    generate_dense_matrices(data_dir, rng)

    print("\n--- 3. Sparse Datasets ---")
    generate_sparse_matrices(data_dir, rng)

    print("\nAll datasets generated successfully.")


if __name__ == "__main__":
    main()
