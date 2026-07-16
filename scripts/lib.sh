#!/bin/bash
# scripts/lib.sh
# Shared functions sourced by all bench_exN.sh and avg_exN.sh scripts.
# Do not execute directly.

# Path to compiled executables, relative to the project root.
BINDIR="bin"

# Extra flags passed to mpirun. Leave empty on the department network; set to
# "--allow-run-as-root --oversubscribe" when running inside a container.
MPIRUN_FLAGS="${MPIRUN_FLAGS:---oversubscribe}"

# ---------------------------------------------------------------------------
# Guard: abort if required tools are missing
# ---------------------------------------------------------------------------

require_tools() {
    for tool in awk sort; do
        if ! command -v "$tool" >/dev/null 2>&1; then
            echo "Error: required tool '$tool' is not installed"
            exit 1
        fi
    done
}

# ---------------------------------------------------------------------------
# Guard: abort if mpirun is missing
# ---------------------------------------------------------------------------

require_mpi() {
    if ! command -v mpirun >/dev/null 2>&1; then
        echo "Error: mpirun is not installed"
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# Verify the program exists and is executable
# Usage: require_program <name>   (looks up bin/<name>)
# ---------------------------------------------------------------------------

require_program() {
    local name="$1"
    local path="$BINDIR/$name"
    if [ ! -x "$path" ]; then
        echo "Error: $path not found or not executable — run 'make' first"
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# Standard trap setup — call at the top of each bench script
# ---------------------------------------------------------------------------

setup_trap() {
    trap 'echo "[interrupted] cleaning up..."; pkill -P $$ || true; exit 1' INT TERM
}

# ---------------------------------------------------------------------------
# Collect system information into a file
# Usage: collect_system_info <output_file>
# ---------------------------------------------------------------------------

collect_system_info() {
    local out="$1"
    {
        echo "Hostname:"
        hostname
        echo
        echo "CPU model:"
        lscpu | grep "Model name" | sed 's/^[ \t]*//'
        echo
        echo "CPU cores/threads:"
        lscpu | grep -E "CPU\(s\)|Core\(s\) per socket|Thread\(s\) per core|Socket\(s\)" \
              | sed 's/^[ \t]*//'
        echo
        echo "SIMD support:"
        grep -oE '(avx512[a-z]*|avx2|avx|sse4_[12]|ssse3|sse3|sse2|sse|fma)' /proc/cpuinfo \
              | sort -u | tr '\n' ' '
        echo
        echo
        echo "Operating system:"
        if [ -f /etc/os-release ]; then
            grep PRETTY_NAME /etc/os-release | cut -d= -f2 | tr -d '"'
        else
            uname -a
        fi
        echo
        echo "Kernel:"
        uname -r
        echo
        echo "Compiler:"
        gcc --version | head -n 1
        echo
        echo "MPI:"
        mpirun --version 2>/dev/null | head -n 1
    } > "$out"
    echo "[bench] system info written to $out"
}

# ---------------------------------------------------------------------------
# Generic awk averaging script used by all avg_exN.sh scripts.
# Groups rows by key columns, averages numeric value columns.
# Usage: avg_csv <input_file> <output_file> <key_cols> <val_cols> <header>
#
# key_cols : comma-separated 1-based column numbers for grouping
# val_cols : comma-separated 1-based column numbers to average
# header   : output CSV header line
# ---------------------------------------------------------------------------

avg_csv() {
    local input="$1"
    local output="$2"
    local key_cols="$3"
    local val_cols="$4"
    local header="$5"

    if [ ! -f "$input" ]; then
        echo "Error: $input not found"
        exit 1
    fi

    awk -F, -v key_cols="$key_cols" -v val_cols="$val_cols" '
    BEGIN {
        n_keys = split(key_cols, kc, ",")
        n_vals = split(val_cols, vc, ",")
    }
    NR > 1 {
        key = ""
        for (i = 1; i <= n_keys; i++) {
            key = key (i > 1 ? "," : "") $kc[i]
        }
        for (i = 1; i <= n_vals; i++) {
            val_sum[key SUBSEP i] += $vc[i]
        }
        count[key]++
    }
    END {
        for (key in count) {
            printf "%s", key
            for (i = 1; i <= n_vals; i++) {
                printf ",%.6f", val_sum[key SUBSEP i] / count[key]
            }
            printf "\n"
        }
    }
    ' "$input" | sort -t, -k1,1 -k2,2n -k3,3n > "$output.tmp"

    echo "$header" > "$output"
    cat "$output.tmp" >> "$output"
    rm -f "$output.tmp"

    echo "[avg] averages written to $output"
}
