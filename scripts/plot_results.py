"""Generate figures for M127 Assignment 2 from the averaged benchmark CSVs.

Every figure plots measured times only, exactly as the programs reported them.
No ratios, baselines, or reference lines are drawn.

Format example::

    python3 scripts/plot_results.py --results results --outdir results/figures
"""

import argparse
import csv
import os

import matplotlib
import matplotlib.pyplot as plt


def read_csv(path):
    """Read a CSV file into a list of dictionaries.

    :param path: Path to the CSV file.
    :return: A list of rows, or an empty list when the file is missing.
    """
    if not os.path.isfile(path):
        print("skip: %s not found" % path)
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sweep_rows(rows, sweep, key):
    """Select the rows of one sweep and sort them by a numeric column.

    :param rows: List of row dictionaries.
    :param sweep: The sweep label to keep.
    :param key: Column name used for sorting.
    :return: The matching rows in ascending order of that column.
    """
    selected = [row for row in rows if row.get("sweep") == sweep]
    return sorted(selected, key=lambda row: int(row[key]))


def column(rows, key):
    """Extract a column as a list of floats.

    :param rows: List of row dictionaries.
    :param key: The column name to extract.
    :return: The column values converted to float.
    """
    return [float(row[key]) for row in rows]


def save(figure, outdir, name):
    """Write a figure and close it.

    :param figure: The matplotlib figure.
    :param outdir: Directory where the figure is written.
    :param name: File name of the figure.
    :return: None.
    """
    figure.tight_layout()
    figure.savefig(os.path.join(outdir, name), dpi=150)
    plt.close(figure)
    print("wrote %s" % name)


def line_figure(x_values, series, xlabel, title, logy=False):
    """Build a line figure from one or more named series of times.

    :param x_values: Values of the horizontal axis.
    :param series: List of (label, values, marker) tuples.
    :param xlabel: Label of the horizontal axis.
    :param title: Title of the figure.
    :param logy: Whether the vertical axis should be logarithmic.
    :return: The matplotlib figure.
    """
    figure, axis = plt.subplots(figsize=(7, 4.5))
    for label, values, marker in series:
        axis.plot(x_values, values, marker=marker, label=label)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("time (s)")
    axis.set_title(title)
    if logy:
        axis.set_yscale("log")
    if len(series) > 1:
        axis.legend()
    return figure


def plot_ex1(results, outdir):
    """Plot the measured times of exercise 2.1.

    :param results: Directory holding the averaged CSV files.
    :param outdir: Directory where the figures are written.
    :return: None.
    """
    rows = read_csv(os.path.join(results, "ex1", "bench_ex1_averages.csv"))
    if not rows:
        return

    procs = sweep_rows(rows, "procs", "procs")
    if procs:
        workers = [int(row["procs"]) for row in procs]

        figure = line_figure(
            workers,
            [("MPI total", column(procs, "avg_mpi_total"), "o"),
             ("Pthreads", column(procs, "avg_pthreads"), "s")],
            "workers",
            "Exercise 2.1 time by worker count (degree 100000)")
        save(figure, outdir, "ex1_workers.png")

        figure = line_figure(
            workers,
            [("send", column(procs, "avg_mpi_send"), "o"),
             ("receive", column(procs, "avg_mpi_receive"), "s")],
            "MPI processes",
            "Exercise 2.1 communication time by process count")
        save(figure, outdir, "ex1_communication.png")

    degree = sweep_rows(rows, "degree", "degree")
    if degree:
        figure = line_figure(
            [int(row["degree"]) for row in degree],
            [("serial", column(degree, "avg_serial"), "^"),
             ("Pthreads", column(degree, "avg_pthreads"), "s"),
             ("MPI total", column(degree, "avg_mpi_total"), "o")],
            "degree",
            "Exercise 2.1 time by degree (4 workers)",
            logy=True)
        save(figure, outdir, "ex1_degree.png")


def plot_ex2(results, outdir):
    """Plot the measured times of exercise 2.2.

    :param results: Directory holding the averaged CSV files.
    :param outdir: Directory where the figures are written.
    :return: None.
    """
    rows = read_csv(os.path.join(results, "ex2", "bench_ex2_averages.csv"))
    if not rows:
        return

    sparsity = sweep_rows(rows, "sparsity", "sparsity")
    if sparsity:
        figure = line_figure(
            [int(row["sparsity"]) for row in sparsity],
            [("CSR total", column(sparsity, "avg_csr_total"), "o"),
             ("dense total", column(sparsity, "avg_dense_total"), "s")],
            "sparsity (percent zeros)",
            "Exercise 2.2 time by sparsity")
        save(figure, outdir, "ex2_sparsity.png")

    iterations = sweep_rows(rows, "iterations", "iterations")
    if iterations:
        figure = line_figure(
            [int(row["iterations"]) for row in iterations],
            [("CSR total", column(iterations, "avg_csr_total"), "o"),
             ("dense total", column(iterations, "avg_dense_total"), "s")],
            "iterations",
            "Exercise 2.2 time by iteration count")
        save(figure, outdir, "ex2_iterations.png")

    size = sweep_rows(rows, "size", "size")
    if size:
        figure = line_figure(
            [int(row["size"]) for row in size],
            [("CSR total", column(size, "avg_csr_total"), "o"),
             ("dense total", column(size, "avg_dense_total"), "s")],
            "matrix size",
            "Exercise 2.2 time by matrix size")
        save(figure, outdir, "ex2_size.png")

    procs = sweep_rows(rows, "procs", "procs")
    if procs:
        figure = line_figure(
            [int(row["procs"]) for row in procs],
            [("CSR build", column(procs, "avg_csr_build"), "^"),
             ("CSR send", column(procs, "avg_csr_send"), "s"),
             ("CSR compute", column(procs, "avg_csr_compute"), "o")],
            "MPI processes",
            "Exercise 2.2 phase times by process count")
        save(figure, outdir, "ex2_phases.png")


def plot_ex3(results, outdir):
    """Plot the measured times of exercise 2.3.

    :param results: Directory holding the averaged CSV files.
    :param outdir: Directory where the figures are written.
    :return: None.
    """
    rows = read_csv(os.path.join(results, "ex3", "bench_ex3_averages.csv"))
    if not rows:
        return

    degree = sweep_rows(rows, "degree", "degree")
    if not degree:
        return

    figure = line_figure(
        [int(row["degree"]) for row in degree],
        [("scalar", column(degree, "avg_scalar"), "^"),
         ("SIMD", column(degree, "avg_simd"), "o")],
        "degree",
        "Exercise 2.3 time by degree",
        logy=True)
    save(figure, outdir, "ex3_degree.png")


def main():
    """Parse arguments and generate every figure.

    :return: None.
    """
    matplotlib.use("Agg")
    parser = argparse.ArgumentParser(description="Plot assignment 2 results")
    parser.add_argument("--results", default="results")
    parser.add_argument("--outdir", default="results/figures")
    arguments = parser.parse_args()

    os.makedirs(arguments.outdir, exist_ok=True)
    plot_ex1(arguments.results, arguments.outdir)
    plot_ex2(arguments.results, arguments.outdir)
    plot_ex3(arguments.results, arguments.outdir)
    print("figures written to %s" % arguments.outdir)


if __name__ == "__main__":
    main()
