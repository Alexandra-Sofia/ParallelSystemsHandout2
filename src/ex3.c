/**
 * @file ex3.c
 * @brief Exercise 2.3 — Polynomial multiplication with SIMD extensions.
 *
 * Generates two random dense polynomials of degree n, multiplies them with the
 * original scalar O(n^2) algorithm, then repeats the multiplication with a
 * vectorised algorithm that uses SIMD instructions and no threads. The two
 * results are compared element by element to confirm the vector path is exact.
 *
 * The vector kernel broadcasts one coefficient of A into a vector register and
 * multiplies it against eight coefficients of B per instruction, accumulating
 * into the shifted output band. The trailing coefficients that do not fill a
 * full vector are handled with scalar code.
 *
 * Coefficients and accumulators are 32-bit. AVX2 provides a packed 32-bit
 * integer multiply (_mm256_mullo_epi32) but no efficient packed 64-bit
 * multiply, so 32-bit storage is what makes the vector path worthwhile. The
 * coefficients are kept small so that the products stay well inside the 32-bit
 * range, and the scalar and vector paths use the same type, so they agree
 * exactly rather than approximately.
 *
 * Usage:
 *   ./ex3 <degree>
 *
 * Example:
 *   ./ex3 100000
 */

#include <errno.h>
#include <immintrin.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SEED 42
#define MAX_DEGREE 2000000
#define MAX_COEFF 9
#define SIMD_WIDTH 8

/**
 * @brief Return the current monotonic time in seconds.
 *
 * @return Wall-clock time as a double, in seconds.
 */
static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

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
 * @param argc   Argument count from main.
 * @param argv   Argument vector from main.
 * @param degree Output: parsed polynomial degree.
 * @return 1 if all arguments are valid, 0 otherwise.
 */
static int parse_args(int argc, char *argv[], int *degree)
{
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <degree>\n", argv[0]);
        return 0;
    }

    return parse_positive_int(argv[1], degree, MAX_DEGREE, "degree");
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
        exit(1);
    }
    return ptr;
}

/**
 * @brief Allocate zero-initialised memory and exit on failure.
 *
 * @param n    Number of elements.
 * @param size Size of each element in bytes.
 * @return Pointer to zeroed memory, never NULL.
 */
static void *xcalloc(size_t n, size_t size)
{
    void *ptr = calloc(n, size);
    if (!ptr) {
        fprintf(stderr, "Error: calloc failed for %zu bytes\n", n * size);
        exit(1);
    }
    return ptr;
}

/**
 * @brief Multiply two polynomials using the original scalar algorithm.
 *
 * Computes result[i+j] += a[i] * b[j] for every pair, which is the same
 * traversal order the vector kernel uses, so the two are directly comparable.
 *
 * @param a      Coefficients of the first polynomial (size degree+1).
 * @param b      Coefficients of the second polynomial (size degree+1).
 * @param result Output array (size 2*degree+1), pre-zeroed.
 * @param degree Degree of each input polynomial.
 */
static void poly_multiply_scalar(int *a, int *b, int *result, int degree)
{
    int size = degree + 1;

    for (int i = 0; i < size; i++) {
        int a_val = a[i];
        for (int j = 0; j < size; j++) {
            result[i + j] += a_val * b[j];
        }
    }
}

/**
 * @brief Multiply two polynomials using AVX2 SIMD instructions.
 *
 * For each coefficient of A the value is broadcast across a 256-bit register
 * and multiplied against eight coefficients of B at a time. The products are
 * added into the output band starting at index i, which is read back, updated,
 * and stored. Unaligned loads and stores are used because the output offset i
 * shifts by one element on every outer iteration, so the access cannot be kept
 * aligned without padding.
 *
 * @param a      Coefficients of the first polynomial (size degree+1).
 * @param b      Coefficients of the second polynomial (size degree+1).
 * @param result Output array (size 2*degree+1), pre-zeroed.
 * @param degree Degree of each input polynomial.
 */
static void poly_multiply_simd(int *a, int *b, int *result, int degree)
{
    int size = degree + 1;

    for (int i = 0; i < size; i++) {
        __m256i a_vec = _mm256_set1_epi32(a[i]);
        int j = 0;

        for (; j + SIMD_WIDTH <= size; j += SIMD_WIDTH) {
            __m256i b_vec = _mm256_loadu_si256((__m256i *)(b + j));
            __m256i r_vec = _mm256_loadu_si256((__m256i *)(result + i + j));
            __m256i prod = _mm256_mullo_epi32(a_vec, b_vec);
            r_vec = _mm256_add_epi32(r_vec, prod);
            _mm256_storeu_si256((__m256i *)(result + i + j), r_vec);
        }

        for (; j < size; j++) {
            result[i + j] += a[i] * b[j];
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
static int results_match(int *a, int *b, int size)
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
 * Parses arguments, generates the polynomials, runs the scalar and vector
 * multiplications, prints timing, and verifies that both agree.
 *
 * @param argc Argument count.
 * @param argv Argument vector: degree.
 * @return 0 on success, 1 on argument error, 2 on correctness failure.
 */
int main(int argc, char *argv[])
{
    int degree;

    if (!parse_args(argc, argv, &degree)) {
        return 1;
    }

    int size = degree + 1;
    int result_size = 2 * degree + 1;

    fprintf(stderr, "[ex3] degree=%d\n", degree);
    fprintf(stderr, "[ex3] allocating and generating polynomials...\n");

    double t_start = now();

    int *a = xmalloc((size_t)size, sizeof(int));
    int *b = xmalloc((size_t)size, sizeof(int));
    int *result_scalar = xcalloc((size_t)result_size, sizeof(int));
    int *result_simd = xcalloc((size_t)result_size, sizeof(int));

    srand(SEED);
    for (int i = 0; i < size; i++) {
        a[i] = (rand() % MAX_COEFF) + 1;
        b[i] = (rand() % MAX_COEFF) + 1;
    }

    double t_init = now() - t_start;
    printf("Init time:     %.6f s\n", t_init);
    fprintf(stderr, "[ex3] init done (%.6f s)\n", t_init);

    /* Scalar multiplication */
    fprintf(stderr, "[ex3] running scalar multiply...\n");
    double t0 = now();
    poly_multiply_scalar(a, b, result_scalar, degree);
    double t_scalar = now() - t0;
    printf("Scalar time:   %.6f s\n", t_scalar);
    fprintf(stderr, "[ex3] scalar done (%.6f s)\n", t_scalar);

    /* Vectorised multiplication */
    fprintf(stderr, "[ex3] running SIMD multiply...\n");
    double t1 = now();
    poly_multiply_simd(a, b, result_simd, degree);
    double t_simd = now() - t1;
    printf("SIMD time:     %.6f s\n", t_simd);
    printf("Speedup:       %.2fx\n", t_scalar / t_simd);
    fprintf(stderr, "[ex3] SIMD done (%.6f s)\n", t_simd);

    /* Verification */
    fprintf(stderr, "[ex3] verifying results...\n");
    int ok = results_match(result_scalar, result_simd, result_size);
    printf("Correctness:   %s\n", ok ? "[OK]" : "[FAIL]");
    fprintf(stderr, "[ex3] done\n");

    free(a);
    free(b);
    free(result_scalar);
    free(result_simd);
    return ok ? 0 : 2;
}
