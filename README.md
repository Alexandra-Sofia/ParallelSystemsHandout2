# Parallel Computing Systems — Programming Assignment 2 (MPI & CUDA)

Implementations for the second programming assignment: distributed memory
parallelism with **MPI**, vectorisation with **SIMD** extensions, and GPU
acceleration with **CUDA**.

| Source     | Exercise | Topic                                        |
| ---------- | -------- | -------------------------------------------- |
| `src/ex1.c`| 2.1      | Polynomial multiplication with MPI           |
| `src/ex2.c`| 2.2      | Sparse matrix-vector multiplication (CSR)    |
| `src/ex3.c`| 2.3      | Polynomial multiplication with SIMD          |
| `src/ex4/` | 2.4      | Maxwell's equations simulator with CUDA      |

## Repository layout

```text
.
├── Makefile
├── README.md
├── src/
│   ├── ex1.c              Exercise 2.1 — MPI polynomial multiplication
│   ├── ex2.c              Exercise 2.2 — MPI sparse matrix-vector (CSR)
│   ├── ex3.c              Exercise 2.3 — SIMD polynomial multiplication
│   └── ex4/               Exercise 2.4 — CUDA sources from the NVIDIA DLI course
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

```text
GCC with OpenMP support
Open MPI (mpicc, mpirun)
GNU Make
bash, awk, sort
A CPU with AVX2 (for 2.3)
```


## Build

```bash
make
```

Exercises 2.1 and 2.2 are compiled with `mpicc`, exercise 2.3 with `gcc -mavx2`.
All use `-O2 -Wall -Wextra -Wpedantic -std=c11`.

```bash
make clean
```

## Running

```bash
mpirun -np <procs> ./bin/ex1 <degree> [omp_threads]
mpirun -np <procs> ./bin/ex2 <matrix_size> <sparsity_pct> <iterations>
./bin/ex3 <degree>
```

Examples:

```bash
mpirun -np 4 ./bin/ex1 100000
mpirun -np 4 ./bin/ex2 4000 90 10
./bin/ex3 100000
```

Each program prints its timings to stdout and progress messages to stderr, and
ends with a `Correctness: [OK]` line. Exit status is 0 on success, 1 on an
argument error, and 2 if verification fails.

## Benchmarking

```bash
bash scripts/bench_ex1.sh
bash scripts/avg_ex1.sh
```

Each `bench_exN.sh` runs several sweeps, repeats every configuration four
times, and writes `results/exN/bench_exN_results.csv` together with
`bench_exN_system.txt` recording the processor model, core layout, and SIMD
support. Each `avg_exN.sh` averages the repeats into `bench_exN_averages.csv`.

The sweeps are:

```text
ex1  processes (1,2,4,8) at degree 100000
     degree (10k,50k,100k,200k) at 4 processes

ex2  processes (1,2,4,8) at 4000x4000, 90% sparsity, 10 iterations
     sparsity (0,50,75,90,99) at 4000x4000, 4 processes
     size (1000,2000,4000,8000) at 90% sparsity, 4 processes
     iterations (1,5,10,20) at 4000x4000, 4 processes

ex3  degree (10k,50k,100k,200k,500k)
```


```bash
MPIRUN_FLAGS="--allow-run-as-root --oversubscribe" bash scripts/bench_ex1.sh
```

## Smoke test

```bash
bash tests/smoke-test-all.sh
```

Builds everything and verifies correctness across process counts, sparsity
levels, and uneven row distributions.

## Exercises

### Exercise 2.1 — Polynomial multiplication with MPI

Rank 0 generates both polynomials, scatters the coefficients of A, and
broadcasts B in full. Broadcasting B is necessary rather than wasteful: every
coefficient of A must be convolved against all of B. Each rank convolves its
slice into a full-length partial product, and because the partials overlap they
are combined with a sum reduction rather than concatenated.

Partitioning A rather than the output coefficients gives perfect load balance,
since each coefficient of A performs exactly `degree+1` multiply-adds. The
output coefficients, by contrast, are triangular in work: the middle of the
result array has far more contributing terms than the edges.

The three phases required by the assignment are timed separately: the scatter
and broadcast (send), the local convolution (compute), and the reduction
(receive). Rank 0 also runs the serial algorithm and an OpenMP version with a
matching worker count, so the single-node MPI versus shared-memory comparison
can be made directly from one run.

### Exercise 2.2 — Sparse matrix-vector multiplication with MPI and CSR

Rank 0 builds the sparse matrix, converts it to CSR, and sends each rank only
the CSR data for the rows that rank owns. Rows are the natural unit of
decomposition: each output element depends on exactly one matrix row and the
whole input vector.

The multiplication is iterative, with the result of each iteration feeding the
next. Since a row may touch any column, every rank needs the complete input
vector at the start of each iteration, so the partial result vectors are
combined with an allgather after every iteration. That allgather is the
synchronisation point and the dominant communication cost of the algorithm.

The same iteration is also run with the dense representation across the same
processes, so the CSR and dense forms are compared under identical conditions.
Rank 0 runs the serial CSR version as the correctness reference and the speedup
baseline.

### Exercise 2.3 — Polynomial multiplication with SIMD

The scalar kernel is compared against an AVX2 kernel that broadcasts one
coefficient of A into a vector register and multiplies it against eight
coefficients of B per instruction. The trailing coefficients that do not fill a
full vector are handled with scalar code, and the two results are compared
element by element.

Coefficients and accumulators are 32-bit. AVX2 offers a packed 32-bit integer
multiply but no efficient packed 64-bit multiply, so 32-bit storage is what
makes the vector path worthwhile. Coefficients are kept small so products stay
well inside the 32-bit range, and both paths use the same type, so they agree
exactly.

### Exercise 2.4 — Maxwell's equations simulator with CUDA

Completed in the NVIDIA DLI course "Getting Started with Accelerated Computing
in Modern CUDA C++". The assessment accelerates a 2D Maxwell's equations
simulator in three stages, each of which must clear a throughput threshold in
cells per second: port the CPU simulator to the GPU, accelerate it with fancy
iterators, and coarsen the grid with cooperative algorithms.

## Notes on comparing results

Each exercise is run on a dedicated machine from the department network.
Comparisons within an exercise use results from the same machine, and absolute
times are not compared across different machines. The lab nodes are 4-core, so
single-node scaling is expected to saturate around four workers.
