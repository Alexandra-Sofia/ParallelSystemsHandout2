#!/bin/bash
set -euo pipefail
# shellcheck source=lib.sh
source "$(dirname "$0")/lib.sh"

setup_trap
require_tools
require_mpi
require_program "ex1"

REPEATS=4
RESULTS_DIR="results/ex1"
RESULTS_FILE="$RESULTS_DIR/bench_ex1_results.csv"
SYSTEM_FILE="$RESULTS_DIR/bench_ex1_system.txt"

mkdir -p "$RESULTS_DIR"
collect_system_info "$SYSTEM_FILE"

echo "sweep,degree,procs,repeat,serial,mpi_send,mpi_compute,mpi_receive,mpi_total,mpi_speedup,pthreads,pthreads_speedup,correctness" \
    > "$RESULTS_FILE"

run_ex1() {
    local sweep="$1" degree="$2" procs="$3" repeat="$4"
    local output
    output=$(mpirun $MPIRUN_FLAGS -np "$procs" "$BINDIR/ex1" "$degree" "$procs" 2>/dev/null)

    local ser send comp recv tot spd pth pspd ok
    ser=$(echo  "$output" | awk '/Serial time:/    {print $3}')
    send=$(echo "$output" | awk '/MPI send time:/  {print $4}')
    comp=$(echo "$output" | awk '/MPI compute:/    {print $3}')
    recv=$(echo "$output" | awk '/MPI receive:/    {print $3}')
    tot=$(echo  "$output" | awk '/MPI total:/      {print $3}')
    spd=$(echo  "$output" | awk '/MPI speedup:/    {print $3}' | tr -d 'x')
    pth=$(echo  "$output" | awk '/Pthreads time:/    {print $3}')
    pspd=$(echo "$output" | awk '/Pthreads speedup:/ {print $3}' | tr -d 'x')
    ok=$(echo   "$output" | awk '/Correctness:/    {print $2}' | tr -d '[]')

    if [ -z "$ser" ] || [ -z "$send" ] || [ -z "$comp" ] || [ -z "$recv" ] || \
       [ -z "$tot" ] || [ -z "$pth" ] || [ -z "$ok" ]; then
        echo "Error: failed to parse output for degree=$degree procs=$procs"
        echo "$output"
        exit 1
    fi

    echo "$sweep,$degree,$procs,$repeat,$ser,$send,$comp,$recv,$tot,$spd,$pth,$pspd,$ok" \
        >> "$RESULTS_FILE"
    echo "[bench] sweep=$sweep degree=$degree procs=$procs repeat=$repeat mpi_total=$tot pthreads=$pth"
}

echo "[bench] sweep 1: varying processes"
for procs in 1 2 4 8; do
    for repeat in $(seq 1 "$REPEATS"); do
        run_ex1 procs 100000 "$procs" "$repeat"
    done
done

echo "[bench] sweep 2: varying degree"
for degree in 10000 50000 100000 120000 130000 140000 160000 200000; do
    for repeat in $(seq 1 "$REPEATS"); do
        run_ex1 degree "$degree" 4 "$repeat"
    done
done

echo "[bench] results written to $RESULTS_FILE"
