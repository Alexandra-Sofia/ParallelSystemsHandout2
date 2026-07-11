#!/bin/bash
set -euo pipefail
# shellcheck source=lib.sh
source "$(dirname "$0")/lib.sh"

setup_trap
require_tools
require_program "ex3"

REPEATS=4
RESULTS_DIR="results/ex3"
RESULTS_FILE="$RESULTS_DIR/bench_ex3_results.csv"
SYSTEM_FILE="$RESULTS_DIR/bench_ex3_system.txt"

mkdir -p "$RESULTS_DIR"
collect_system_info "$SYSTEM_FILE"

echo "sweep,degree,repeat,init,scalar,simd,speedup,correctness" > "$RESULTS_FILE"

run_ex3() {
    local sweep="$1" degree="$2" repeat="$3"
    local output
    output=$("$BINDIR/ex3" "$degree" 2>/dev/null)

    local init scalar simd spd ok
    init=$(echo   "$output" | awk '/Init time:/   {print $3}')
    scalar=$(echo "$output" | awk '/Scalar time:/ {print $3}')
    simd=$(echo   "$output" | awk '/SIMD time:/   {print $3}')
    spd=$(echo    "$output" | awk '/Speedup:/     {print $2}' | tr -d 'x')
    ok=$(echo     "$output" | awk '/Correctness:/ {print $2}' | tr -d '[]')

    if [ -z "$init" ] || [ -z "$scalar" ] || [ -z "$simd" ] || [ -z "$ok" ]; then
        echo "Error: failed to parse output for degree=$degree"
        echo "$output"
        exit 1
    fi

    echo "$sweep,$degree,$repeat,$init,$scalar,$simd,$spd,$ok" >> "$RESULTS_FILE"
    echo "[bench] sweep=$sweep degree=$degree repeat=$repeat scalar=$scalar simd=$simd speedup=$spd"
}

echo "[bench] sweep 1: varying degree"
for degree in 10000 50000 100000 200000 500000; do
    for repeat in $(seq 1 "$REPEATS"); do
        run_ex3 degree "$degree" "$repeat"
    done
done

echo "[bench] results written to $RESULTS_FILE"
