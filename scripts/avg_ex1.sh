#!/bin/bash
set -euo pipefail
# shellcheck source=lib.sh
source "$(dirname "$0")/lib.sh"

# sweep(1) degree(2) procs(3) repeat(4) serial(5) mpi_send(6) mpi_compute(7)
# mpi_receive(8) mpi_total(9) mpi_speedup(10) pthreads(11) pthreads_speedup(12)
# correctness(13)
avg_csv \
    "results/ex1/bench_ex1_results.csv" \
    "results/ex1/bench_ex1_averages.csv" \
    "1,2,3" \
    "5,6,7,8,9,10,11,12" \
    "sweep,degree,procs,avg_serial,avg_mpi_send,avg_mpi_compute,avg_mpi_receive,avg_mpi_total,avg_mpi_speedup,avg_pthreads,avg_pthreads_speedup"
