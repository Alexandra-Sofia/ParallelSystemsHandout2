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
 * and an OpenMP version with the same worker count, so the MPI and shared
 * memory implementations can be compared on a single node.
 *
 * Usage:
 *   mpirun -np <procs> ./ex1 <degree> [omp_threads]
 *
 * omp_threads defaults to the number of MPI processes.
 *
 * Example:
 *   mpirun -np 4 ./ex1 100000
 */

#include <errno.h>
#include <limits.h>
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SEED 42
#define MAX_DEGREE 2000000
#define MAX_THREADS 256
#define ROOT 0

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
 * @param omp_threads Output: OpenMP thread count for the comparison run.
 * @param num_procs   Number of MPI processes, used as the thread default.
 * @return 1 if all arguments are valid, 0 otherwise.
 */
static int parse_args(int argc, char *argv[], int *degree, int *omp_threads,
                      int num_procs)
{
    if (argc != 2 && argc != 3) {
        fprintf(stderr, "Usage: mpirun -np <procs> %s <degree> [omp_threads]\n",
                argv[0]);
        return 0;
    }

    if (!parse_positive_int(argv[1], degree, MAX_DEGREE, "degree")) {
        return 0;
    }

    if (argc == 3) {
        if (!parse_positive_int(argv[2], omp_threads, MAX_THREADS,
                                "omp_threads")) {
            return 0;
        }
    } else {
        *omp_threads = num_procs;
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
 * @brief Multiply two polynomials in parallel using OpenMP.
 *
 * Parallelises over output coefficients, so each iteration writes a distinct
 * index and no reduction or critical section is needed. This mirrors the
 * fastest shared-memory version from Exercise 1.1 and exists so the MPI
 * implementation can be compared against it on a single node.
 *
 * @param a           Coefficients of the first polynomial (size degree+1).
 * @param b           Coefficients of the second polynomial (size degree+1).
 * @param result      Output array (size 2*degree+1), pre-zeroed.
 * @param degree      Degree of each input polynomial.
 * @param num_threads Number of OpenMP threads to use.
 */
static void poly_multiply_openmp(int *a, int *b, long long *result, int degree,
                                 int num_threads)
{
    int size = degree + 1;
    int result_size = 2 * degree + 1;

    omp_set_num_threads(num_threads);

    #pragma omp parallel for schedule(static, 1)
    for (int k = 0; k < result_size; k++) {
        long long sum = 0;
        int i_start = k - degree;
        if (i_start < 0) {
            i_start = 0;
        }
        int i_end = k;
        if (i_end >= size) {
            i_end = size - 1;
        }
        for (int i = i_start; i <= i_end; i++) {
            sum += (long long)a[i] * b[k - i];
        }
        result[k] = sum;
    }
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
 * @brief Program entry point.
 *
 * Parses arguments, generates the polynomials on rank 0, runs the MPI
 * multiplication in three timed phases, runs the serial and OpenMP versions on
 * rank 0 for comparison, prints timings, and verifies correctness.
 *
 * @param argc Argument count.
 * @param argv Argument vector: degree, optional omp_threads.
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
    int omp_threads;

    if (!parse_args(argc, argv, &degree, &omp_threads, num_procs)) {
        MPI_Finalize();
        return 1;
    }

    int size = degree + 1;
    int result_size = 2 * degree + 1;

    if (rank == ROOT) {
        fprintf(stderr, "[ex1] degree=%d procs=%d omp_threads=%d\n",
                degree, num_procs, omp_threads);
        fprintf(stderr, "[ex1] allocating and generating polynomials...\n");
    }

    int *counts = xmalloc((size_t)num_procs, sizeof(int));
    int *displs = xmalloc((size_t)num_procs, sizeof(int));
    compute_distribution(size, num_procs, counts, displs);

    int local_count = counts[rank];
    int local_displ = displs[rank];

    double t_start = MPI_Wtime();

    int *a = NULL;
    long long *result_serial = NULL;
    long long *result_openmp = NULL;
    long long *result_mpi = NULL;

    int *b = xmalloc((size_t)size, sizeof(int));
    int *a_local = xcalloc((size_t)local_count, sizeof(int));
    long long *partial = xcalloc((size_t)result_size, sizeof(long long));

    if (rank == ROOT) {
        a = xmalloc((size_t)size, sizeof(int));
        result_serial = xcalloc((size_t)result_size, sizeof(long long));
        result_openmp = xcalloc((size_t)result_size, sizeof(long long));
        result_mpi = xcalloc((size_t)result_size, sizeof(long long));

        srand(SEED);
        for (int i = 0; i < size; i++) {
            a[i] = (rand() % 100) + 1;
            b[i] = (rand() % 100) + 1;
        }
    }

    double t_init = MPI_Wtime() - t_start;

    if (rank == ROOT) {
        printf("Init time:      %.6f s\n", t_init);
        fprintf(stderr, "[ex1] init done (%.6f s)\n", t_init);
        fprintf(stderr, "[ex1] running MPI multiply...\n");
    }

    /* Phase 1: distribute the data. A is scattered, B is broadcast in full. */
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    MPI_Scatterv(a, counts, displs, MPI_INT,
                 a_local, local_count, MPI_INT, ROOT, MPI_COMM_WORLD);
    MPI_Bcast(b, size, MPI_INT, ROOT, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double t_send = MPI_Wtime() - t0;

    /* Phase 2: local convolution of this rank's slice against all of B. */
    double t1 = MPI_Wtime();

    poly_multiply_local(a_local, local_count, local_displ, b, partial, degree);

    MPI_Barrier(MPI_COMM_WORLD);
    double t_compute = MPI_Wtime() - t1;

    /* Phase 3: overlapping partials are summed onto rank 0. */
    double t2 = MPI_Wtime();

    MPI_Reduce(partial, result_mpi, result_size, MPI_LONG_LONG, MPI_SUM,
               ROOT, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double t_receive = MPI_Wtime() - t2;

    double t_mpi_total = t_send + t_compute + t_receive;

    int ok = 1;

    if (rank == ROOT) {
        fprintf(stderr, "[ex1] MPI done (%.6f s)\n", t_mpi_total);

        fprintf(stderr, "[ex1] running serial multiply...\n");
        double t3 = MPI_Wtime();
        poly_multiply_serial(a, b, result_serial, degree);
        double t_serial = MPI_Wtime() - t3;
        fprintf(stderr, "[ex1] serial done (%.6f s)\n", t_serial);

        fprintf(stderr, "[ex1] running OpenMP multiply...\n");
        double t4 = MPI_Wtime();
        poly_multiply_openmp(a, b, result_openmp, degree, omp_threads);
        double t_openmp = MPI_Wtime() - t4;
        fprintf(stderr, "[ex1] OpenMP done (%.6f s)\n", t_openmp);

        printf("Serial time:    %.6f s\n", t_serial);
        printf("\n");
        printf("MPI send time:  %.6f s\n", t_send);
        printf("MPI compute:    %.6f s\n", t_compute);
        printf("MPI receive:    %.6f s\n", t_receive);
        printf("MPI total:      %.6f s  [%d procs]\n", t_mpi_total, num_procs);
        printf("MPI speedup:    %.2fx\n", t_serial / t_mpi_total);
        printf("\n");
        printf("OpenMP time:    %.6f s  [%d threads]\n", t_openmp, omp_threads);
        printf("OpenMP speedup: %.2fx\n", t_serial / t_openmp);
        printf("MPI vs OpenMP:  %.2fx\n", t_openmp / t_mpi_total);

        fprintf(stderr, "[ex1] verifying results...\n");
        ok = results_match(result_serial, result_mpi, result_size) &&
             results_match(result_serial, result_openmp, result_size);
        printf("\nCorrectness:    %s\n", ok ? "[OK]" : "[FAIL]");
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
    free(result_openmp);
    free(result_mpi);

    MPI_Finalize();
    return ok ? 0 : 2;
}
