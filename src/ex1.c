/**
 * @file ex1.c
 * @brief Exercise 2.1 — Polynomial multiplication with MPI.
 *
 * Rank 0 generates two random dense polynomials of degree n. It scatters the
 * coefficients of the first polynomial across the ranks and broadcasts the
 * second in full, since every coefficient of A must be convolved against all
 * of B. Each rank computes a partial product over the full output range, and
 * the partials are summed onto rank 0 with a reduction.
 *
 * The three communication and computation phases are timed separately, as
 * required: the scatter/broadcast (send), the local convolution (compute),
 * and the reduction (receive). The total excludes allocation and generation.
 *
 * Rank 0 also runs the serial O(n^2) algorithm for verification and speedup,
 * and the Pthreads implementation from Exercise 1.1 with the same worker
 * count. Pthreads was the faster of the two shared-memory versions measured
 * in Exercise 1.1, so it is the reference the assignment asks to compare the
 * MPI implementation against on a single node.
 *
 * Usage:
 *   mpirun -np <procs> ./ex1 <degree> [num_threads]
 *
 * num_threads defaults to the number of MPI processes.
 *
 * Example:
 *   mpirun -np 4 ./ex1 100000
 */

#include <errno.h>
#include <limits.h>
#include <mpi.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SEED 42
#define MAX_DEGREE 2000000
#define MAX_THREADS 256
#define ROOT 0

/**
 * @brief Arguments passed to each Pthreads worker thread.
 */
typedef struct {
    int *a;            /**< Coefficients of the first polynomial. */
    int *b;            /**< Coefficients of the second polynomial. */
    long long *result; /**< Output array (size 2*degree+1). */
    int degree;        /**< Degree of each input polynomial. */
    int thread_id;     /**< Zero-based index of this thread. */
    int num_threads;   /**< Total number of worker threads. */
} poly_args_t;

/**
 * @brief The timings the assignment requires the program to report.
 */
typedef struct {
    double init;     /**< Allocating and generating the polynomials. */
    double serial;   /**< The serial reference multiplication. */
    double send;     /**< Scattering A and broadcasting B. */
    double compute;  /**< The local convolution on every rank. */
    double receive;  /**< Reducing the partial products onto rank 0. */
    double mpi;      /**< Send plus compute plus receive. */
    double pthreads; /**< The shared-memory reference from Exercise 1.1. */
} timings_t;

/**
 * @brief Parse a string as a positive integer using strtol.
 *
 * Rejects empty strings, non-numeric input, values with trailing garbage,
 * negative values, zero, and values exceeding max.
 *
 * @param str   Input string to parse.
 * @param out   Output: parsed integer value on success.
 * @param max   Maximum accepted value (inclusive).
 * @param label Name of the parameter, used in error messages.
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
 * @param argc        Argument count from main.
 * @param argv        Argument vector from main.
 * @param degree      Output: parsed polynomial degree.
 * @param num_threads Output: Pthreads thread count for the comparison run.
 * @param num_procs   Number of MPI processes, used as the thread default.
 * @return 1 if all arguments are valid, 0 otherwise.
 */
static int parse_args(int argc, char *argv[], int *degree, int *num_threads,
                      int num_procs)
{
    if (argc != 2 && argc != 3) {
        fprintf(stderr, "Usage: mpirun -np <procs> %s <degree> [num_threads]\n",
                argv[0]);
        return 0;
    }

    if (!parse_positive_int(argv[1], degree, MAX_DEGREE, "degree")) {
        return 0;
    }

    if (argc == 3) {
        if (!parse_positive_int(argv[2], num_threads, MAX_THREADS,
                                "num_threads")) {
            return 0;
        }
    } else {
        *num_threads = num_procs;
    }

    return 1;
}

/**
 * @brief Allocate memory and exit on failure.
 *
 * @param n    Number of elements.
 * @param size Size of each element in bytes.
 * @return Pointer to allocated memory, never NULL.
 */
static void *xmalloc(size_t n, size_t size)
{
    void *ptr = malloc(n * size);
    if (!ptr) {
        fprintf(stderr, "Error: malloc failed for %zu bytes\n", n * size);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return ptr;
}

/**
 * @brief Allocate zero-initialised memory and exit on failure.
 *
 * A count of zero still allocates one element so the pointer stays valid,
 * which matters for ranks that receive an empty slice.
 *
 * @param n    Number of elements.
 * @param size Size of each element in bytes.
 * @return Pointer to zeroed memory, never NULL.
 */
static void *xcalloc(size_t n, size_t size)
{
    void *ptr = calloc(n > 0 ? n : 1, size);
    if (!ptr) {
        fprintf(stderr, "Error: calloc failed for %zu bytes\n", n * size);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return ptr;
}

/**
 * @brief Multiply two polynomials using the serial O(n^2) algorithm.
 *
 * Computes result[k] = sum_{i+j=k} a[i] * b[j] for each k. The caller must
 * allocate result with at least 2*degree+1 elements, pre-initialised to zero.
 *
 * @param a      Coefficients of the first polynomial (size degree+1).
 * @param b      Coefficients of the second polynomial (size degree+1).
 * @param result Output coefficient array (size 2*degree+1).
 * @param degree Degree of each input polynomial.
 */
static void poly_multiply_serial(int *a, int *b, long long *result, int degree)
{
    int size = degree + 1;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            result[i + j] += (long long)a[i] * b[j];
        }
    }
}

/**
 * @brief Worker function executed by each Pthreads thread.
 *
 * Uses cyclic (interleaved) distribution over output coefficients rather
 * than block distribution. Thread t computes k = t, t+T, t+2T, ... where
 * T is the total thread count. This balances load naturally: the work per
 * coefficient k peaks at the middle of the result array (degree+1 terms)
 * and tapers toward the edges. Cyclic distribution gives every thread a
 * mix of cheap and expensive coefficients, avoiding the imbalance that
 * block distribution creates when the middle threads receive all the heavy
 * coefficients.
 *
 * No synchronisation is required because each k is owned by exactly one
 * thread.
 *
 * @param arg Pointer to a poly_args_t struct for this thread.
 * @return NULL always.
 */
static void *pthread_worker(void *arg)
{
    poly_args_t *args = (poly_args_t *)arg;
    int size = args->degree + 1;
    int result_size = 2 * args->degree + 1;

    for (int k = args->thread_id; k < result_size; k += args->num_threads) {
        long long sum = 0;
        int i_start = k - args->degree;
        if (i_start < 0) {
            i_start = 0;
        }
        int i_end = k;
        if (i_end >= size) {
            i_end = size - 1;
        }
        for (int i = i_start; i <= i_end; i++) {
            sum += (long long)args->a[i] * args->b[k - i];
        }
        args->result[k] = sum;
    }
    return NULL;
}

/**
 * @brief Multiply two polynomials in parallel using Pthreads.
 *
 * Spawns num_threads threads with cyclic distribution over output
 * coefficients. Each thread computes its assigned indices independently.
 *
 * @param a           Coefficients of the first polynomial (size degree+1).
 * @param b           Coefficients of the second polynomial (size degree+1).
 * @param result      Output array (size 2*degree+1), pre-zeroed.
 * @param degree      Degree of each input polynomial.
 * @param num_threads Number of Pthreads worker threads to spawn.
 */
static void poly_multiply_pthreads(int *a, int *b, long long *result,
                                   int degree, int num_threads)
{
    pthread_t *threads = xmalloc((size_t)num_threads, sizeof(pthread_t));
    poly_args_t *args = xmalloc((size_t)num_threads, sizeof(poly_args_t));

    for (int t = 0; t < num_threads; t++) {
        args[t].a = a;
        args[t].b = b;
        args[t].result = result;
        args[t].degree = degree;
        args[t].thread_id = t;
        args[t].num_threads = num_threads;

        int rc = pthread_create(&threads[t], NULL, pthread_worker, &args[t]);
        if (rc != 0) {
            fprintf(stderr, "Error: pthread_create failed for thread %d: %s\n",
                    t, strerror(rc));
            exit(1);
        }
    }
    for (int t = 0; t < num_threads; t++) {
        int rc = pthread_join(threads[t], NULL);
        if (rc != 0) {
            fprintf(stderr, "Error: pthread_join failed for thread %d: %s\n",
                    t, strerror(rc));
            exit(1);
        }
    }

    free(threads);
    free(args);
}

/**
 * @brief Compute the block distribution of A's coefficients over the ranks.
 *
 * The first (size % num_procs) ranks receive one extra coefficient. Blocks are
 * contiguous, which keeps the scatter a single collective and gives every rank
 * an equal share of work: each coefficient of A performs exactly degree+1
 * multiply-adds against B, so equal counts mean equal work.
 *
 * @param size      Total number of coefficients to distribute.
 * @param num_procs Number of MPI ranks.
 * @param counts    Output: coefficient count per rank (size num_procs).
 * @param displs    Output: start offset per rank (size num_procs).
 */
static void compute_distribution(int size, int num_procs, int *counts,
                                 int *displs)
{
    int base = size / num_procs;
    int remainder = size % num_procs;
    int offset = 0;

    for (int r = 0; r < num_procs; r++) {
        counts[r] = base + (r < remainder ? 1 : 0);
        displs[r] = offset;
        offset += counts[r];
    }
}

/**
 * @brief Convolve this rank's slice of A against the whole of B.
 *
 * The local coefficient at index i sits at global index displ+i, so its
 * contributions land in the output band starting at displ+i. Because the bands
 * of different ranks overlap, the partial arrays must be combined with a sum
 * reduction rather than concatenated.
 *
 * @param a_local Local slice of A's coefficients.
 * @param count   Number of coefficients in the local slice.
 * @param displ   Global index of the first local coefficient.
 * @param b       Full second polynomial (size degree+1).
 * @param partial Output array (size 2*degree+1), pre-zeroed.
 * @param degree  Degree of each input polynomial.
 */
static void poly_multiply_local(int *a_local, int count, int displ, int *b,
                                long long *partial, int degree)
{
    int size = degree + 1;

    for (int i = 0; i < count; i++) {
        long long a_val = a_local[i];
        int base = displ + i;
        for (int j = 0; j < size; j++) {
            partial[base + j] += a_val * b[j];
        }
    }
}

/**
 * @brief Compare two result arrays element-by-element.
 *
 * @param a    First array.
 * @param b    Second array.
 * @param size Number of elements.
 * @return 1 if all elements are equal, 0 otherwise.
 */
static int results_match(long long *a, long long *b, int size)
{
    for (int i = 0; i < size; i++) {
        if (a[i] != b[i]) {
            return 0;
        }
    }
    return 1;
}


/**
 * @brief Run the MPI multiplication in its three timed phases.
 *
 * Rank 0 scatters the coefficients of A and broadcasts B in full, every rank
 * convolves its slice into a full-length partial product, and the partials are
 * summed onto rank 0. The phases are separated by barriers so that each
 * reported time is the time of the slowest rank rather than rank 0's local
 * view, which is what makes the numbers meaningful.
 *
 * @param a          Full first polynomial on rank 0; ignored elsewhere.
 * @param b          Second polynomial buffer (size degree+1) on every rank.
 * @param a_local    Scratch buffer for this rank's slice of A.
 * @param partial    Scratch buffer (size 2*degree+1), pre-zeroed.
 * @param result     Output on rank 0 (size 2*degree+1); ignored elsewhere.
 * @param counts     Coefficient count per rank.
 * @param displs     First coefficient index per rank.
 * @param rank       This rank.
 * @param degree     Degree of each input polynomial.
 * @param t          Output: the send, compute, receive, and mpi timings.
 */
static void run_mpi_multiply(int *a, int *b, int *a_local, long long *partial,
                             long long *result, const int *counts,
                             const int *displs, int rank, int degree,
                             timings_t *t)
{
    int size = degree + 1;
    int result_size = 2 * degree + 1;
    int local_count = counts[rank];
    int local_displ = displs[rank];

    /* Phase 1: distribute the data. A is scattered, B is broadcast in full. */
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    MPI_Scatterv(a, counts, displs, MPI_INT,
                 a_local, local_count, MPI_INT, ROOT, MPI_COMM_WORLD);
    MPI_Bcast(b, size, MPI_INT, ROOT, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    t->send = MPI_Wtime() - start;

    /* Phase 2: local convolution of this rank's slice against all of B. */
    start = MPI_Wtime();

    poly_multiply_local(a_local, local_count, local_displ, b, partial, degree);

    MPI_Barrier(MPI_COMM_WORLD);
    t->compute = MPI_Wtime() - start;

    /* Phase 3: the overlapping partials are summed onto rank 0. */
    start = MPI_Wtime();

    MPI_Reduce(partial, result, result_size, MPI_LONG_LONG, MPI_SUM,
               ROOT, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    t->receive = MPI_Wtime() - start;

    t->mpi = t->send + t->compute + t->receive;
}

/**
 * @brief Print the configuration and all required timings.
 *
 * @param t           Collected timings.
 * @param degree      Degree of each input polynomial.
 * @param num_procs   Number of MPI ranks.
 * @param num_threads Threads used by the shared-memory reference.
 */
static void print_report(const timings_t *t, int degree, int num_procs,
                         int num_threads)
{
    printf("Degree:           %d\n", degree);
    printf("Init time:        %.6f s\n", t->init);
    printf("Serial time:      %.6f s\n", t->serial);
    printf("\n");
    printf("MPI send time:    %.6f s\n", t->send);
    printf("MPI compute:      %.6f s\n", t->compute);
    printf("MPI receive:      %.6f s\n", t->receive);
    printf("MPI total:        %.6f s  [%d procs]\n", t->mpi, num_procs);
    printf("MPI speedup:      %.2fx\n", t->serial / t->mpi);
    printf("\n");
    printf("Pthreads time:    %.6f s  [%d threads]\n", t->pthreads,
           num_threads);
    printf("Pthreads speedup: %.2fx\n", t->serial / t->pthreads);
    printf("MPI vs Pthreads:  %.2fx\n", t->pthreads / t->mpi);
}

/**
 * @brief Program entry point.
 *
 * Parses arguments, generates the polynomials on rank 0, runs the MPI
 * multiplication in three timed phases, runs the serial and Pthreads versions
 * on rank 0 for comparison, prints timings, and verifies correctness.
 *
 * @param argc Argument count.
 * @param argv Argument vector: degree, optional num_threads.
 * @return 0 on success, 1 on argument error, 2 on correctness failure.
 */
int main(int argc, char *argv[])
{
    int rank;
    int num_procs;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int degree;
    int num_threads;

    if (!parse_args(argc, argv, &degree, &num_threads, num_procs)) {
        MPI_Finalize();
        return 1;
    }

    int size = degree + 1;
    int result_size = 2 * degree + 1;

    if (rank == ROOT) {
        fprintf(stderr, "[ex1] degree=%d procs=%d threads=%d\n",
                degree, num_procs, num_threads);
    }

    int *counts = xmalloc((size_t)num_procs, sizeof(int));
    int *displs = xmalloc((size_t)num_procs, sizeof(int));
    compute_distribution(size, num_procs, counts, displs);

    timings_t t = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

    int *a = NULL;
    long long *result_serial = NULL;
    long long *result_pthreads = NULL;
    long long *result_mpi = NULL;

    int *b = xmalloc((size_t)size, sizeof(int));
    int *a_local = xcalloc((size_t)counts[rank], sizeof(int));
    long long *partial = xcalloc((size_t)result_size, sizeof(long long));

    /* Rank 0 generates both polynomials. The generation is deliberately
     * outside the timed region of the parallel computation. */
    if (rank == ROOT) {
        fprintf(stderr, "[ex1] allocating and generating polynomials...\n");
        double start = MPI_Wtime();

        a = xmalloc((size_t)size, sizeof(int));
        result_serial = xcalloc((size_t)result_size, sizeof(long long));
        result_pthreads = xcalloc((size_t)result_size, sizeof(long long));
        result_mpi = xcalloc((size_t)result_size, sizeof(long long));

        srand(SEED);
        for (int i = 0; i < size; i++) {
            a[i] = (rand() % 100) + 1;
            b[i] = (rand() % 100) + 1;
        }

        t.init = MPI_Wtime() - start;
        fprintf(stderr, "[ex1] init done (%.6f s)\n", t.init);
        fprintf(stderr, "[ex1] running MPI multiply...\n");
    }

    run_mpi_multiply(a, b, a_local, partial, result_mpi, counts, displs,
                     rank, degree, &t);

    int ok = 1;

    if (rank == ROOT) {
        fprintf(stderr, "[ex1] MPI done (%.6f s)\n", t.mpi);

        fprintf(stderr, "[ex1] running serial multiply...\n");
        double start = MPI_Wtime();
        poly_multiply_serial(a, b, result_serial, degree);
        t.serial = MPI_Wtime() - start;
        fprintf(stderr, "[ex1] serial done (%.6f s)\n", t.serial);

        fprintf(stderr, "[ex1] running Pthreads multiply...\n");
        start = MPI_Wtime();
        poly_multiply_pthreads(a, b, result_pthreads, degree, num_threads);
        t.pthreads = MPI_Wtime() - start;
        fprintf(stderr, "[ex1] Pthreads done (%.6f s)\n", t.pthreads);

        print_report(&t, degree, num_procs, num_threads);

        fprintf(stderr, "[ex1] verifying results...\n");
        ok = results_match(result_serial, result_mpi, result_size) &&
             results_match(result_serial, result_pthreads, result_size);
        printf("\nCorrectness:      %s\n", ok ? "[OK]" : "[FAIL]");
        fprintf(stderr, "[ex1] done\n");
    }

    MPI_Bcast(&ok, 1, MPI_INT, ROOT, MPI_COMM_WORLD);

    free(counts);
    free(displs);
    free(b);
    free(a_local);
    free(partial);
    free(a);
    free(result_serial);
    free(result_pthreads);
    free(result_mpi);

    MPI_Finalize();
    return ok ? 0 : 2;
}
