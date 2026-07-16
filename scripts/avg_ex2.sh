#!/bin/bash
set -euo pipefail
# shellcheck source=lib.sh
source "$(dirname "$0")/lib.sh"

# sweep(1) size(2) sparsity(3) iterations(4) procs(5) repeat(6)
# csr_build(7) csr_send(8) csr_compute(9) csr_total(10)
# dense_total(11) csr_vs_dense(12) serial_build(13) serial_compute(14)
# serial_total(15) compute_speedup(16) total_speedup(17) correctness(18)
avg_csv \
    "results/ex2/bench_ex2_results.csv" \
    "results/ex2/bench_ex2_averages.csv" \
    "1,2,3,4,5" \
    "7,8,9,10,11,12,13,14,15,16,17" \
    "sweep,size,sparsity,iterations,procs,avg_csr_build,avg_csr_send,avg_csr_compute,avg_csr_total,avg_dense_total,avg_csr_vs_dense,avg_serial_build,avg_serial_compute,avg_serial_total,avg_compute_speedup,avg_total_speedup"
