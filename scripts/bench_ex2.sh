#!/bin/bash
set -euo pipefail
# shellcheck source=lib.sh
source "$(dirname "$0")/lib.sh"

setup_trap
require_tools
require_mpi
require_program "ex2"

REPEATS=4
RESULTS_DIR="results/ex2"
RESULTS_FILE="$RESULTS_DIR/bench_ex2_results.csv"
SYSTEM_FILE="$RESULTS_DIR/bench_ex2_system.txt"

mkdir -p "$RESULTS_DIR"
collect_system_info "$SYSTEM_FILE"

echo "sweep,size,sparsity,iterations,procs,repeat,csr_build,csr_send,csr_compute,csr_total,dense_total,csr_vs_dense,serial_build,serial_compute,serial_total,compute_speedup,total_speedup,correctness" \
    > "$RESULTS_FILE"

run_ex2() {
    local sweep="$1" size="$2" sparsity="$3" iters="$4" procs="$5" repeat="$6"
    local output
    output=$(mpirun $MPIRUN_FLAGS -np "$procs" "$BINDIR/ex2" "$size" "$sparsity" "$iters" 2>/dev/null)

    local build send comp tot dense cvd sbuild scomp stot cspd tspd ok
    build=$(echo "$output" | awk '/CSR build time:/   {print $4}')
    send=$(echo  "$output" | awk '/CSR send time:/    {print $4}')
    comp=$(echo  "$output" | awk '/CSR compute time:/ {print $4}')
    tot=$(echo   "$output" | awk '/CSR total time:/   {print $4}')
    dense=$(echo "$output" | awk '/Dense total time:/ {print $4}')
    cvd=$(echo   "$output" | awk '/CSR vs Dense:/     {print $4}' | tr -d 'x')
    sbuild=$(echo "$output" | awk '/Serial build time:/ {print $4}')
    scomp=$(echo  "$output" | awk '/Serial compute:/    {print $3}')
    stot=$(echo   "$output" | awk '/Serial total time:/ {print $4}')
    cspd=$(echo   "$output" | awk '/Compute speedup:/   {print $3}' | tr -d 'x')
    tspd=$(echo   "$output" | awk '/Total speedup:/     {print $3}' | tr -d 'x')
    ok=$(echo    "$output" | awk '/Correctness:/      {print $2}' | tr -d '[]')

    if [ -z "$build" ] || [ -z "$send" ] || [ -z "$comp" ] || [ -z "$tot" ] || \
       [ -z "$dense" ] || [ -z "$scomp" ] || [ -z "$ok" ]; then
        echo "Error: failed to parse output for size=$size sparsity=$sparsity procs=$procs"
        echo "$output"
        exit 1
    fi

    echo "$sweep,$size,$sparsity,$iters,$procs,$repeat,$build,$send,$comp,$tot,$dense,$cvd,$sbuild,$scomp,$stot,$cspd,$tspd,$ok" \
        >> "$RESULTS_FILE"
    echo "[bench] sweep=$sweep size=$size sparsity=$sparsity iters=$iters procs=$procs repeat=$repeat csr_total=$tot dense=$dense"
}

echo "[bench] sweep 1: varying processes"
for procs in 1 2 4 8; do
    for repeat in $(seq 1 "$REPEATS"); do
        run_ex2 procs 4000 90 10 "$procs" "$repeat"
    done
done

echo "[bench] sweep 2: varying sparsity"
for sparsity in 0 50 75 90 99; do
    for repeat in $(seq 1 "$REPEATS"); do
        run_ex2 sparsity 4000 "$sparsity" 10 4 "$repeat"
    done
done

echo "[bench] sweep 3: varying size"
for size in 1000 2000 4000 8000; do
    for repeat in $(seq 1 "$REPEATS"); do
        run_ex2 size "$size" 90 10 4 "$repeat"
    done
done

echo "[bench] sweep 4: varying iterations"
for iters in 1 5 10 20; do
    for repeat in $(seq 1 "$REPEATS"); do
        run_ex2 iterations 4000 90 "$iters" 4 "$repeat"
    done
done

echo "[bench] results written to $RESULTS_FILE"
