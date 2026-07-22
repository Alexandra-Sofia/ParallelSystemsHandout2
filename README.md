# Parallel Computing Systems (M127) — Assignment 2 (MPI & CUDA)

| Source      | Exercise | Topic                                     |
| ----------- | -------- | ----------------------------------------- |
| `src/ex1.c` | 2.1      | Polynomial multiplication with MPI        |
| `src/ex2.c` | 2.2      | Sparse matrix-vector multiplication (CSR) |
| `src/ex3.c` | 2.3      | Polynomial multiplication with SIMD       |

## Layout

```text
.
├── Makefile
├── README.md
├── src/
│   ├── ex1.c              Exercise 2.1 — MPI polynomial multiplication
│   ├── ex2.c              Exercise 2.2 — MPI sparse matrix-vector (CSR)
│   ├── ex3.c              Exercise 2.3 — SIMD polynomial multiplication
├── scripts/
│   ├── lib.sh             shared functions sourced by all bench and avg scripts
│   ├── bench_ex1.sh       benchmark sweeps for exercise 2.1
│   ├── bench_ex2.sh       benchmark sweeps for exercise 2.2
│   ├── bench_ex3.sh       benchmark sweeps for exercise 2.3
│   ├── avg_ex1.sh         averages the repeats of exercise 2.1
│   ├── avg_ex2.sh         averages the repeats of exercise 2.2
│   └── avg_ex3.sh         averages the repeats of exercise 2.3
├── tests/
│   └── smoke-test-all.sh  quick correctness check across all exercises
└── results/
    ├── ex1/               bench_ex1_results.csv, _averages.csv, _system.txt
    ├── ex2/
    ├── ex3/
    └── ex4/
```

## Requirements

Open MPI (`mpicc`, `mpirun`), GCC with OpenMP, GNU Make, and a CPU with AVX2 for
exercise 2.3. Exercises 2.1 and 2.2 run on the department network
(`linux01`..`linux30`). Exercise 2.4 runs in the NVIDIA DLI environment. The
optional plotter needs `matplotlib`.

## Build and run

```bash
make

mpirun -np <procs> ./bin/ex1 <degree> [num_threads]
mpirun -np <procs> ./bin/ex2 <matrix_size> <sparsity_pct> <iterations>
./bin/ex3 <degree>
```

Each program prints its timings and ends with a `Correctness: [OK]` line. Exit
status is 0 on success, 1 on an argument error, 2 on a correctness failure.

## Benchmarks

```bash
bash scripts/bench_exN.sh          # runs the sweeps, writes results/exN/
bash scripts/avg_exN.sh            # averages the repeats
python3 scripts/plot_results.py    # optional, writes results/figures/
```

Sweeps: ex1 varies processes and degree; ex2 varies processes, sparsity, size,
and iterations; ex3 varies degree. Each configuration is repeated four times and
the machine details are recorded in `bench_exN_system.txt`.

## Smoke test

```bash
bash tests/smoke-test-all.sh
```

## Note

Each exercise runs on a single machine of the department network. Comparisons
are made within an exercise on the same machine; absolute times are not compared
across machines.