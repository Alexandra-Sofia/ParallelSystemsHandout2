#!/bin/bash
set -uo pipefail
# shellcheck source=../scripts/lib.sh
source "$(dirname "$0")/../scripts/lib.sh"

FAILURES=0

check() {
    local name="$1" output="$2"
    local ok
    ok=$(echo "$output" | awk '/Correctness:/ {print $2}' | tr -d '[]')
    if [ "$ok" = "OK" ]; then
        echo "[PASS] $name"
    else
        echo "[FAIL] $name"
        FAILURES=$((FAILURES + 1))
    fi
}

echo "[smoke] building..."
make >/dev/null || exit 1

echo "[smoke] ex1 — MPI polynomial multiplication"
check "ex1 np=1" "$(mpirun $MPIRUN_FLAGS -np 1 "$BINDIR/ex1" 500 1 2>/dev/null)"
check "ex1 np=2" "$(mpirun $MPIRUN_FLAGS -np 2 "$BINDIR/ex1" 500 2 2>/dev/null)"
check "ex1 np=4" "$(mpirun $MPIRUN_FLAGS -np 4 "$BINDIR/ex1" 500 4 2>/dev/null)"

echo "[smoke] ex2 — MPI sparse matrix-vector multiplication"
check "ex2 np=1 sparsity=90" "$(mpirun $MPIRUN_FLAGS -np 1 "$BINDIR/ex2" 300 90 5 2>/dev/null)"
check "ex2 np=2 sparsity=90" "$(mpirun $MPIRUN_FLAGS -np 2 "$BINDIR/ex2" 300 90 5 2>/dev/null)"
check "ex2 np=4 sparsity=0"  "$(mpirun $MPIRUN_FLAGS -np 4 "$BINDIR/ex2" 300 0 5 2>/dev/null)"
check "ex2 np=4 sparsity=99" "$(mpirun $MPIRUN_FLAGS -np 4 "$BINDIR/ex2" 300 99 5 2>/dev/null)"
check "ex2 np=3 uneven rows" "$(mpirun $MPIRUN_FLAGS -np 3 "$BINDIR/ex2" 301 75 3 2>/dev/null)"

echo "[smoke] ex3 — SIMD polynomial multiplication"
check "ex3 degree=1000"  "$("$BINDIR/ex3" 1000 2>/dev/null)"
check "ex3 degree=10000" "$("$BINDIR/ex3" 10000 2>/dev/null)"

echo
if [ "$FAILURES" -eq 0 ]; then
    echo "[smoke] all tests passed"
    exit 0
fi
echo "[smoke] $FAILURES test(s) failed"
exit 2
