#!/bin/bash
set -euo pipefail
# shellcheck source=lib.sh
source "$(dirname "$0")/lib.sh"

# sweep(1) degree(2) repeat(3) init(4) scalar(5) simd(6) speedup(7)
# correctness(8)
avg_csv \
    "results/ex3/bench_ex3_results.csv" \
    "results/ex3/bench_ex3_averages.csv" \
    "1,2" \
    "4,5,6,7" \
    "sweep,degree,avg_init,avg_scalar,avg_simd,avg_speedup"
