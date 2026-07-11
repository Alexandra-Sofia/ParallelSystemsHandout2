/**
 * @file ex2.c
 * @brief Exercise 2.2 — Sparse matrix-vector multiplication with MPI and CSR.
 *
 * Rank 0 builds a random sparse square matrix with a given percentage of zero
 * elements, converts it to Compressed Sparse Row (CSR) format, and distributes
 * the rows across the ranks. Each rank owns a contiguous block of rows and
 * receives only the CSR data for those rows, which is the minimal amount of
 * matrix data needed for the parallel computation.
 *
 * The multiplication is iterative: the result vector of each iteration becomes
 * the input vector of the next, modelling an iterative solver. Because a row
 * of the matrix may touch any column, every rank needs the complete input
 * vector at the start of each iteration. The partial result vectors are
 * therefore combined with an allgather after every iteration, which is the
 * synchronisation point of the algorithm.
 *
 * The same iteration is also run with the dense representation over MPI, so
 * the CSR and dense forms can be compared at identical process counts. Rank 0
 * additionally runs the serial CSR version, which provides the correctness
 * reference and the speedup baseline.
 *
 * Usage:
 *   mpirun -np <procs> ./ex2 <matrix_size> <sparsity_pct> <iterations>
 *
 * sparsity_pct: percentage of elements that are zero, e.g. 90 for 90% zeros.
 *
 * Example:
 *   mpirun -np 4 ./ex2 4000 90 10
 */

#include <errno.h>
#include <limits.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SEED 42
#define MAX_SIZE 20000
#define MAX_ITERATIONS 10000
#define ROOT 0

/**
 * @brief Compressed Sparse Row representation of a matrix block.
 *
 * For a block of rows with nnz non-zero elements:
 *   values  : non-zero values in row-major order (size nnz)
 *   col_idx : column index of each non-zero (size nnz)
 *   row_ptr : row_ptr[i] is the index in values/col_idx where local row i
 *             starts; row_ptr[rows] == nnz (sentinel)
 *
 * On rank 0 the struct holds the whole matrix; on the other ranks it holds
 * only the rows that rank owns, with row_ptr rebased to start at zero.
 */
typedef struct {
    int *values;  /**< Non-zero element values (size nnz). */
    int *col_idx; /**< Column indices of non-zeros (size nnz). */
    int *row_ptr; /**< Row start pointers (size rows+1). */
    int rows;     /**< Number of rows held in this block. */
    int nnz;      /**< Number of non-zero elements in this block. */
} csr_matrix_t;

/**
 * @brief The timings the assignment requires the program to report.
 */
typedef struct {
    double csr_build;   /**< Building the CSR representation on rank 0. */
    double csr_send;    /**< Distributing the CSR rows and the input vector. */
    double csr_compute; /**< The iterative parallel CSR multiplication. */
    double csr_total;   /**< Build plus send plus compute. */
    double dense_total; /**< The same iteration using the dense form. */
    double csr_serial;  /**< The serial CSR reference run on rank 0. */
} timings_t;

/**
 * @brief Parse a string as a positive integer using strtol.
 *
 * @param str   Input string.
 * @param out   Output value on success.
 * @param max   Maximum accepted value (inclusive).
 * @param label Parameter name for error messages.
 * @return 1 on success, 0 on any error.
 */
static int parse_positive_int(const char *str, int *out, int max,
                              const char *label)
{
    if (str == NULL || str[0] == '\0') {
        fprintf(stderr, "Error: %s is empty\n", label);
        return 0;
    }

    char *end;
    errno = 0;
    long val = strtol(str, &end, 10);

    if (errno == ERANGE || val > INT_MAX || val < INT_MIN) {
        fprintf(stderr, "Error: %s '%s' overflows\n", label, str);
        return 0;
    }
    if (end == str || *end != '\0') {
        fprintf(stderr, "Error: %s '%s' is not a valid integer\n", label, str);
        return 0;
    }
    if (val <= 0) {
        fprintf(stderr, "Error: %s must be positive, got %ld\n", label, val);
        return 0;
    }
    if (val > max) {
        fprintf(stderr, "Error: %s %ld exceeds maximum of %d\n",
                label, val, max);
        return 0;
    }

    *out = (int)val;
    return 1;
}

/**
 * @brief Validate and parse all command-line arguments.
 *
 * @param argc       Argument count from main.
 * @param argv       Argument vector from main.
 * @param n          Output: matrix dimension.
 * @param sparsity   Output: percentage of zero elements [0,99].
 * @param iterations Output: number of SpMV repetitions.
 * @return 1 if all arguments are valid, 0 otherwise.
 */
static int parse_args(int argc, char *argv[], int *n, int *sparsity,
                      int *iterations)
{
    if (argc != 4) {
        fprintf(stderr,
                "Usage: mpirun -np <procs> %s <matrix_size> <sparsity_pct>"
                " <iterations>\n", argv[0]);
        return 0;
    }

    if (!parse_positive_int(argv[1], n, MAX_SIZE, "matrix_size")) {
        return 0;
    }

    char *end;
    errno = 0;
    long sp = strtol(argv[2], &end, 10);
    if (errno == ERANGE || end == argv[2] || *end != '\0' ||
        sp < 0 || sp > 99) {
        fprintf(stderr,
                "Error: sparsity_pct must be an integer in [0,99], got '%s'\n",
                argv[2]);
        return 0;
    }
    *sparsity = (int)sp;

    if (!parse_positive_int(argv[3], iterations, MAX_ITERATIONS,
                            "iterations")) {
        return 0;
    }

    return 1;
}

/**
 * @brief Allocate memory and abort the job on failure.
 *
 * @param n    Number of elements.
 * @param size Size of each element in bytes.
 * @return Pointer to allocated memory, never NULL.
 */
static void *xmalloc(size_t n, size_t size)
{
    void *ptr = malloc((n > 0 ? n : 1) * size);
    if (ptr == NULL) {
        fprintf(stderr, "Error: malloc failed for %zu bytes\n", n * size);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return ptr;
}

/**
 * @brief Allocate zero-initialised memory and abort the job on failure.
 *
 * @param n    Number of elements.
 * @param size Size of each element in bytes.
 * @return Pointer to zeroed memory, never NULL.
 */
static void *xcalloc(size_t n, size_t size)
{
    void *ptr = calloc(n > 0 ? n : 1, size);
    if (ptr == NULL) {
        fprintf(stderr, "Error: calloc failed for %zu bytes\n", n * size);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return ptr;
}

/**
 * @brief Release the arrays held by a CSR matrix.
 *
 * @param csr CSR matrix to free.
 */
static void csr_free(csr_matrix_t *csr)
{
    free(csr->values);
    free(csr->col_idx);
    free(csr->row_ptr);
    csr->values = NULL;
    csr->col_idx = NULL;
    csr->row_ptr = NULL;
}

/**
 * @brief Allocate and fill a random sparse integer matrix in dense storage.
 *
 * Values are kept small to reduce overflow risk during repeated SpMV.
 *
 * @param n            Matrix dimension.
 * @param sparsity_pct Percentage of zero elements [0, 99].
 * @return Heap-allocated n x n matrix in row-major order. Caller must free.
 */
static int *generate_matrix(int n, int sparsity_pct)
{
    int *mat = xmalloc((size_t)n * n, sizeof(int));
    srand(SEED);

    for (long i = 0; i < (long)n * n; i++) {
        mat[i] = (rand() % 100) < sparsity_pct ? 0 : (rand() % 3) + 1;
    }

    return mat;
}

/**
 * @brief Allocate and fill a random long long vector of length n.
 *
 * @param n Vector length.
 * @return Heap-allocated array. Caller must free.
 */
static long long *generate_vector(int n)
{
    long long *vec = xmalloc((size_t)n, sizeof(long long));

    for (int i = 0; i < n; i++) {
        vec[i] = (rand() % 3) + 1;
    }

    return vec;
}

/**
 * @brief Build the CSR representation of a dense matrix.
 *
 * @param mat Dense matrix in row-major order (size n x n).
 * @param n   Matrix dimension.
 * @return Initialised CSR matrix holding all n rows. Call csr_free to release.
 */
static csr_matrix_t csr_build(int *mat, int n)
{
    csr_matrix_t csr;
    csr.rows = n;
    csr.row_ptr = xcalloc((size_t)(n + 1), sizeof(int));

    for (int i = 0; i < n; i++) {
        int count = 0;
        for (int j = 0; j < n; j++) {
            if (mat[(long)i * n + j] != 0) {
                count++;
            }
        }
        csr.row_ptr[i + 1] = count;
    }

    for (int i = 1; i <= n; i++) {
        csr.row_ptr[i] += csr.row_ptr[i - 1];
    }
    csr.nnz = csr.row_ptr[n];

    csr.values = xmalloc((size_t)csr.nnz, sizeof(int));
    csr.col_idx = xmalloc((size_t)csr.nnz, sizeof(int));

    for (int i = 0; i < n; i++) {
        int pos = csr.row_ptr[i];
        for (int j = 0; j < n; j++) {
            int val = mat[(long)i * n + j];
            if (val != 0) {
                csr.values[pos] = val;
                csr.col_idx[pos] = j;
                pos++;
            }
        }
    }

    return csr;
}

/**
 * @brief Compute the block-row distribution of the matrix over the ranks.
 *
 * The first (n % num_procs) ranks receive one extra row. Rows are the natural
 * unit of decomposition for CSR, since each output element depends on exactly
 * one row of the matrix and the whole input vector.
 *
 * @param n         Matrix dimension.
 * @param num_procs Number of MPI ranks.
 * @param counts    Output: row count per rank (size num_procs).
 * @param displs    Output: first row index per rank (size num_procs).
 */
static void compute_distribution(int n, int num_procs, int *counts, int *displs)
{
    int base = n / num_procs;
    int remainder = n % num_procs;
    int offset = 0;

    for (int r = 0; r < num_procs; r++) {
        counts[r] = base + (r < remainder ? 1 : 0);
        displs[r] = offset;
        offset += counts[r];
    }
}

/**
 * @brief Serial CSR SpMV over a block of rows: y = A_block * x.
 *
 * Each row is an independent dot product against the full input vector.
 *
 * @param csr CSR block.
 * @param x   Full input vector (size n).
 * @param y   Output vector for this block (size csr->rows).
 */
static void spmv_csr(const csr_matrix_t *csr, const long long *x, long long *y)
{
    for (int i = 0; i < csr->rows; i++) {
        long long sum = 0;
        for (int k = csr->row_ptr[i]; k < csr->row_ptr[i + 1]; k++) {
            sum += (long long)csr->values[k] * x[csr->col_idx[k]];
        }
        y[i] = sum;
    }
}

/**
 * @brief Serial dense SpMV over a block of rows: y = A_block * x.
 *
 * @param mat  Dense block in row-major order (size rows x n).
 * @param x    Full input vector (size n).
 * @param y    Output vector for this block (size rows).
 * @param rows Number of rows in the block.
 * @param n    Matrix dimension.
 */
static void spmv_dense(const int *mat, const long long *x, long long *y,
                       int rows, int n)
{
    for (int i = 0; i < rows; i++) {
        long long sum = 0;
        const int *row = mat + (long)i * n;
        for (int j = 0; j < n; j++) {
            sum += (long long)row[j] * x[j];
        }
        y[i] = sum;
    }
}

/**
 * @brief Scatter the CSR matrix by rows to all ranks.
 *
 * Rank 0 sends each rank the row_ptr slice, values, and column indices for the
 * rows that rank owns, and nothing else. The receiving rank rebases its
 * row_ptr so that local row 0 starts at offset zero.
 *
 * @param full      Full CSR matrix on rank 0; ignored on other ranks.
 * @param local     Output: the CSR block owned by this rank.
 * @param counts    Row count per rank (size num_procs).
 * @param displs    First row index per rank (size num_procs).
 * @param rank      This rank.
 * @param num_procs Number of MPI ranks.
 */
static void csr_scatter(const csr_matrix_t *full, csr_matrix_t *local,
                        const int *counts, const int *displs, int rank,
                        int num_procs)
{
    int local_rows = counts[rank];

    local->rows = local_rows;
    local->row_ptr = xcalloc((size_t)(local_rows + 1), sizeof(int));

    if (rank == ROOT) {
        for (int i = 0; i <= local_rows; i++) {
            local->row_ptr[i] = full->row_ptr[i];
        }

        for (int r = 1; r < num_procs; r++) {
            int first = displs[r];
            int rows = counts[r];
            int start = full->row_ptr[first];
            int nnz = full->row_ptr[first + rows] - start;

            int *rebased = xmalloc((size_t)(rows + 1), sizeof(int));
            for (int i = 0; i <= rows; i++) {
                rebased[i] = full->row_ptr[first + i] - start;
            }

            MPI_Send(&nnz, 1, MPI_INT, r, 0, MPI_COMM_WORLD);
            MPI_Send(rebased, rows + 1, MPI_INT, r, 1, MPI_COMM_WORLD);
            MPI_Send(full->values + start, nnz, MPI_INT, r, 2, MPI_COMM_WORLD);
            MPI_Send(full->col_idx + start, nnz, MPI_INT, r, 3, MPI_COMM_WORLD);

            free(rebased);
        }

        local->nnz = local->row_ptr[local_rows];
        local->values = xmalloc((size_t)local->nnz, sizeof(int));
        local->col_idx = xmalloc((size_t)local->nnz, sizeof(int));
        memcpy(local->values, full->values,
               (size_t)local->nnz * sizeof(int));
        memcpy(local->col_idx, full->col_idx,
               (size_t)local->nnz * sizeof(int));
    } else {
        MPI_Recv(&local->nnz, 1, MPI_INT, ROOT, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        MPI_Recv(local->row_ptr, local_rows + 1, MPI_INT, ROOT, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        local->values = xmalloc((size_t)local->nnz, sizeof(int));
        local->col_idx = xmalloc((size_t)local->nnz, sizeof(int));

        MPI_Recv(local->values, local->nnz, MPI_INT, ROOT, 2, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        MPI_Recv(local->col_idx, local->nnz, MPI_INT, ROOT, 3, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
    }
}

/**
 * @brief Compare two long long vectors of length n for equality.
 *
 * @param a First vector.
 * @param b Second vector.
 * @param n Length.
 * @return 1 if all elements are equal, 0 otherwise.
 */
static int vectors_match(long long *a, long long *b, int n)
{
    for (int i = 0; i < n; i++) {
        if (a[i] != b[i]) {
            return 0;
        }
    }
    return 1;
}


/**
 * @brief Run the iterative parallel CSR multiplication.
 *
 * Each rank multiplies the rows it owns, then all ranks exchange their partial
 * result vectors so that every rank holds the complete input vector for the
 * next iteration. That exchange is the synchronisation point of the algorithm.
 *
 * @param local       CSR block owned by this rank.
 * @param x0          Initial input vector (size n), identical on all ranks.
 * @param x           Scratch vector (size n) reused as the iteration input.
 * @param y_local     Scratch vector (size local_rows) for this rank's output.
 * @param out         Output: final result vector (size n).
 * @param counts      Row count per rank.
 * @param displs      First row index per rank.
 * @param n           Matrix dimension.
 * @param local_rows  Rows owned by this rank.
 * @param iterations  Number of multiplications to perform.
 * @return Elapsed time of the iterative computation, in seconds.
 */
static double run_csr_parallel(const csr_matrix_t *local, const long long *x0,
                               long long *x, long long *y_local, long long *out,
                               const int *counts, const int *displs, int n,
                               int local_rows, int iterations)
{
    memcpy(x, x0, (size_t)n * sizeof(long long));

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    for (int it = 0; it < iterations; it++) {
        spmv_csr(local, x, y_local);
        MPI_Allgatherv(y_local, local_rows, MPI_LONG_LONG,
                       x, counts, displs, MPI_LONG_LONG, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double elapsed = MPI_Wtime() - start;

    memcpy(out, x, (size_t)n * sizeof(long long));
    return elapsed;
}

/**
 * @brief Run the same iteration using the dense representation.
 *
 * The dense rows are scattered with the same row decomposition, so the two
 * representations are compared at the same process count. The scatter is inside
 * the timed region because the CSR total likewise includes its distribution
 * step, which keeps the two totals comparable.
 *
 * @param mat        Full dense matrix on rank 0; ignored on other ranks.
 * @param x0         Initial input vector (size n).
 * @param x          Scratch vector (size n) reused as the iteration input.
 * @param y_local    Scratch vector (size local_rows) for this rank's output.
 * @param out        Output: final result vector (size n).
 * @param counts     Row count per rank.
 * @param displs     First row index per rank.
 * @param n          Matrix dimension.
 * @param local_rows Rows owned by this rank.
 * @param num_procs  Number of MPI ranks.
 * @param iterations Number of multiplications to perform.
 * @return Elapsed time including the scatter, in seconds.
 */
static double run_dense_parallel(const int *mat, const long long *x0,
                                 long long *x, long long *y_local,
                                 long long *out, const int *counts,
                                 const int *displs, int n, int local_rows,
                                 int num_procs, int iterations)
{
    int *sendcounts = xmalloc((size_t)num_procs, sizeof(int));
    int *senddispls = xmalloc((size_t)num_procs, sizeof(int));

    for (int r = 0; r < num_procs; r++) {
        sendcounts[r] = counts[r] * n;
        senddispls[r] = displs[r] * n;
    }

    int *mat_local = xmalloc((size_t)local_rows * n, sizeof(int));

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    MPI_Scatterv(mat, sendcounts, senddispls, MPI_INT,
                 mat_local, local_rows * n, MPI_INT, ROOT, MPI_COMM_WORLD);

    memcpy(x, x0, (size_t)n * sizeof(long long));

    for (int it = 0; it < iterations; it++) {
        spmv_dense(mat_local, x, y_local, local_rows, n);
        MPI_Allgatherv(y_local, local_rows, MPI_LONG_LONG,
                       x, counts, displs, MPI_LONG_LONG, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double elapsed = MPI_Wtime() - start;

    memcpy(out, x, (size_t)n * sizeof(long long));

    free(sendcounts);
    free(senddispls);
    free(mat_local);
    return elapsed;
}

/**
 * @brief Run the iterative CSR multiplication serially on rank 0.
 *
 * Provides the correctness reference and the baseline for the speedup. The
 * input and output vectors are swapped between iterations rather than copied.
 *
 * @param full       Full CSR matrix.
 * @param x0         Initial input vector (size n).
 * @param out        Output: final result vector (size n).
 * @param n          Matrix dimension.
 * @param iterations Number of multiplications to perform.
 * @return Elapsed time, in seconds.
 */
static double run_csr_serial(const csr_matrix_t *full, const long long *x0,
                             long long *out, int n, int iterations)
{
    long long *x = xmalloc((size_t)n, sizeof(long long));
    long long *y = xmalloc((size_t)n, sizeof(long long));

    memcpy(x, x0, (size_t)n * sizeof(long long));

    double start = MPI_Wtime();

    for (int it = 0; it < iterations; it++) {
        spmv_csr(full, x, y);
        long long *swap = x;
        x = y;
        y = swap;
    }

    double elapsed = MPI_Wtime() - start;

    memcpy(out, x, (size_t)n * sizeof(long long));

    free(x);
    free(y);
    return elapsed;
}

/**
 * @brief Print the configuration and all required timings.
 *
 * @param t          Collected timings.
 * @param n          Matrix dimension.
 * @param sparsity   Percentage of zero elements requested.
 * @param nnz        Number of non-zero elements actually generated.
 * @param num_procs  Number of MPI ranks.
 * @param iterations Number of multiplications performed.
 */
static void print_report(const timings_t *t, int n, int sparsity, int nnz,
                         int num_procs, int iterations)
{
    printf("Matrix size:        %d x %d\n", n, n);
    printf("Sparsity:           %d%%\n", sparsity);
    printf("NNZ:                %d (%.1f%%)\n", nnz,
           100.0 * nnz / ((double)n * n));
    printf("Processes:          %d\n", num_procs);
    printf("Iterations:         %d\n", iterations);
    printf("\n");
    printf("CSR build time:     %.6f s\n", t->csr_build);
    printf("CSR send time:      %.6f s\n", t->csr_send);
    printf("CSR compute time:   %.6f s\n", t->csr_compute);
    printf("CSR total time:     %.6f s\n", t->csr_total);
    printf("\n");
    printf("Dense total time:   %.6f s\n", t->dense_total);
    printf("CSR vs Dense:       %.2fx\n", t->dense_total / t->csr_total);
    printf("\n");
    printf("CSR serial time:    %.6f s\n", t->csr_serial);
    printf("CSR speedup:        %.2fx\n", t->csr_serial / t->csr_total);
}

/**
 * @brief Program entry point.
 *
 * @param argc Argument count.
 * @param argv matrix_size, sparsity_pct, iterations.
 * @return 0 on success, 1 on argument error, 2 on correctness failure.
 */
int main(int argc, char *argv[])
{
    int rank;
    int num_procs;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int n;
    int sparsity_pct;
    int iterations;

    if (!parse_args(argc, argv, &n, &sparsity_pct, &iterations)) {
        MPI_Finalize();
        return 1;
    }

    if (rank == ROOT) {
        fprintf(stderr,
                "[ex2] size=%dx%d sparsity=%d%% iterations=%d procs=%d\n",
                n, n, sparsity_pct, iterations, num_procs);
    }

    int *counts = xmalloc((size_t)num_procs, sizeof(int));
    int *displs = xmalloc((size_t)num_procs, sizeof(int));
    compute_distribution(n, num_procs, counts, displs);

    int local_rows = counts[rank];

    int *mat = NULL;
    long long *x0 = xmalloc((size_t)n, sizeof(long long));
    csr_matrix_t full = { NULL, NULL, NULL, 0, 0 };
    timings_t t = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

    /* Rank 0 generates the matrix and vector, then builds the CSR form. The
     * generation is deliberately outside the timed region. */
    if (rank == ROOT) {
        fprintf(stderr, "[ex2] generating matrix and vector...\n");
        mat = generate_matrix(n, sparsity_pct);
        long long *generated = generate_vector(n);
        memcpy(x0, generated, (size_t)n * sizeof(long long));
        free(generated);

        fprintf(stderr, "[ex2] building CSR...\n");
        double start = MPI_Wtime();
        full = csr_build(mat, n);
        t.csr_build = MPI_Wtime() - start;
        fprintf(stderr, "[ex2] CSR built, NNZ=%d (%.1f%%) in %.6f s\n",
                full.nnz, 100.0 * full.nnz / ((double)n * n), t.csr_build);
    }

    MPI_Bcast(&t.csr_build, 1, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);

    /* Distribute the CSR rows and the initial vector. */
    MPI_Barrier(MPI_COMM_WORLD);
    double send_start = MPI_Wtime();

    csr_matrix_t local;
    csr_scatter(&full, &local, counts, displs, rank, num_procs);
    MPI_Bcast(x0, n, MPI_LONG_LONG, ROOT, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    t.csr_send = MPI_Wtime() - send_start;

    if (rank == ROOT) {
        fprintf(stderr, "[ex2] CSR distributed (%.6f s)\n", t.csr_send);
    }

    long long *x = xmalloc((size_t)n, sizeof(long long));
    long long *y_local = xmalloc((size_t)local_rows, sizeof(long long));
    long long *result_csr = xmalloc((size_t)n, sizeof(long long));
    long long *result_dense = xmalloc((size_t)n, sizeof(long long));

    if (rank == ROOT) {
        fprintf(stderr, "[ex2] running parallel CSR SpMV (%d iterations)...\n",
                iterations);
    }
    t.csr_compute = run_csr_parallel(&local, x0, x, y_local, result_csr,
                                     counts, displs, n, local_rows, iterations);
    t.csr_total = t.csr_build + t.csr_send + t.csr_compute;

    if (rank == ROOT) {
        fprintf(stderr,
                "[ex2] running parallel dense SpMV (%d iterations)...\n",
                iterations);
    }
    t.dense_total = run_dense_parallel(mat, x0, x, y_local, result_dense,
                                       counts, displs, n, local_rows,
                                       num_procs, iterations);

    int ok = 1;

    if (rank == ROOT) {
        fprintf(stderr, "[ex2] running serial CSR SpMV (%d iterations)...\n",
                iterations);

        long long *reference = xmalloc((size_t)n, sizeof(long long));
        t.csr_serial = run_csr_serial(&full, x0, reference, n, iterations);

        print_report(&t, n, sparsity_pct, full.nnz, num_procs, iterations);

        fprintf(stderr, "[ex2] verifying results...\n");
        ok = vectors_match(reference, result_csr, n) &&
             vectors_match(reference, result_dense, n);
        printf("\nCorrectness:        %s\n", ok ? "[OK]" : "[FAIL]");
        fprintf(stderr, "[ex2] done\n");

        free(reference);
        csr_free(&full);
        free(mat);
    }

    MPI_Bcast(&ok, 1, MPI_INT, ROOT, MPI_COMM_WORLD);

    csr_free(&local);
    free(counts);
    free(displs);
    free(x0);
    free(x);
    free(y_local);
    free(result_csr);
    free(result_dense);

    MPI_Finalize();
    return ok ? 0 : 2;
}
