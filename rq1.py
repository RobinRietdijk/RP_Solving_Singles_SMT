from collections import defaultdict
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import scienceplots

plt.style.use(['science','ieee'])

# Custom settings for plots
plt.rcParams.update({
    "lines.linewidth": 0.8,
    "lines.markersize": 1,
    "legend.fontsize": 5,
    "legend.title_fontsize": 5,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "xtick.minor.visible": False,
    "axes.grid.which": "major",
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "ytick.minor.size": 1.5,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "ytick.minor.width": 0.5,
    "xtick.top": False,
    "ytick.right": False,
    "xtick.bottom": True,
    "ytick.left": True,
    "axes.grid": True,
    "axes.grid.which": "major",
    "grid.color": "0.85",
    "grid.alpha": 0.6,
    "grid.linewidth": 0.6,
    "grid.linestyle": "-", 
})

colors = ["#D62728", "#006BA4", "#8C564B", "#2CA02C", "#7B4EA3", "#FF800E", "#E43D96"]

# Set the colors for solvers to be used in plots
SOLVER_COLORS = {
    "qf_ia": "#D62728", 
    "qf_ia_external": "#006BA4", 
    "qf_bv": "#8C564B", 
    "qf_bool": "#2CA02C", 
    "qf_ia_alt_c": "#7B4EA3",
    "qf_ia_alt_u": "#FF800E", 
    "qf_ia_tree_c": "#E43D96"
}

SOLVER_LINE_STYLES = {
    "qf_ia": "-", 
    "qf_ia_external": "--", 
    "qf_bv": "-.", 
    "qf_bool": ":", 
    "qf_ia_alt_c": "-",
    "qf_ia_alt_u": "--", 
    "qf_ia_tree_c": "-."
}

def _summarize_runtime_scaling(results: list) -> list:
    """ Create a summary of the runtime statistics

    Args:
        results (list): Results to be analysed

    Returns:
        list: Summary of runtime statistics
    """
    # Sort by puzzle
    by_puzzle = {}
    for r in results:
        key = (r["solver"], r["size"], r["puzzle"])
        if key not in by_puzzle:
            by_puzzle[key] = []
        by_puzzle[key].append(r["statistics"]["runtime"])

    # Flatten the puzzles with multiple runs
    per_puzzle = []
    for (solver, size, _), times in by_puzzle.items():
        per_puzzle.append((solver, size, np.median(times)))

    # Sort by puzzle size
    by_size = {}
    for (solver, size, runtime) in per_puzzle:
        key = (solver, size)
        if key not in by_size:
            by_size[key] = []
        by_size[key].append(runtime)

    summary = []
    for (solver, size), runtimes in by_size.items():
        times = np.array(runtimes)
        summary.append({
            "solver": solver,
            "size": size,
            "runs": len(times),
            "median": float(np.median(times)),
            "mean": float(np.mean(times)),
            "std": float(np.std(times)),
            "q1": float(np.percentile(times, 25)),
            "q3": float(np.percentile(times, 75))
        })
    
    return sorted(summary, key=lambda x: (x["size"], x["solver"]))

def _summarize_encoding_scaling(results: list) -> list:
    """ Create a summary of the encoding size statistics

    Args:
        results (list): Results to be analysed

    Returns:
        list: A summary of the encoding size statistics
    """
    # Sort by puzzle
    by_puzzle = {}
    for r in results:
        key = (r["solver"], r["size"], r["puzzle"])
        if key not in by_puzzle:
            by_puzzle[key] = {"variables": [], "assertions": []}
        
        encoding_size = r["statistics"]["encoding_size"]
        total_variables = sum(v for k, v in encoding_size.items() if k != "assertions")
        by_puzzle[key]["variables"].append(total_variables)
        by_puzzle[key]["assertions"].append(encoding_size["assertions"])

    # Flatten multiple runs into single values
    per_puzzle = []
    for (solver, size, _), values in by_puzzle.items():
        per_puzzle.append((solver, size, float(np.median(values["variables"])), float(np.median(values["assertions"]))))

    # Sort by size
    by_size = {}
    for (solver, size, variables, assertions) in per_puzzle:
        key = (solver, size)
        if key not in by_size:
            by_size[key] = {"variables": [], "assertions": []}
        by_size[key]["variables"].append(variables)
        by_size[key]["assertions"].append(assertions)

    summary = []
    for (solver, size), values in by_size.items():
        summary.append({
            "solver": solver,
            "size": size,
            "variables": np.median(values["variables"]),
            "assertions": np.median(values["assertions"]),
        })

    return sorted(summary, key=lambda x: (x["size"], x["solver"]))

def plot_runtime_vs_size(results: list, solver_order: list) -> None:
    """ Plot the median runtime per puzzle size

    Args:
        results (list): Results with runtime satistics
        solver_order (list): Order of the solvers to be plotted to make sure the legend stays the same across experiments
    """
    summary = _summarize_runtime_scaling(results)

    fig, ax = plt.subplots()
    ax.set_prop_cycle(color=colors)

    # To make sure the legend and colors stay the same between experiments, we use a solver order argument
    for i, solver in enumerate(solver_order):
        z = 10+(len(solver_order)-i)
        x = []
        y = []
        for r in summary:
            if r["solver"] == solver:
                x.append(r["size"])
                y.append(r["median"])
        ax.plot(x, y, marker=".", linestyle=SOLVER_LINE_STYLES[solver], color=SOLVER_COLORS[solver], alpha=0.85, label=solver, zorder=z)
    
    ax.set_yscale("log")
    ax.set_xlabel("Puzzle size (n)")
    ax.set_ylabel("Median runtime (s)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    legend = ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        frameon=True,
        framealpha=0.85,
        fancybox=False
    )
    legend.set_zorder(20+len(solver_order))
    frame = legend.get_frame()
    frame.set_alpha(0.75)
    fig.tight_layout()
    out_path = os.path.join(os.path.abspath("plots"), f"runtime_v_size.png")
    fig.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")

def plot_encoding_scaling(results: list, solver_order: list) -> None:
    """ Plot the median runtime per puzzle size

    Args:
        results (list): Results with encoding size satistics
        solver_order (list): Order of the solvers to be plotted to make sure the legend stays the same across experiments
    """
    summary = _summarize_encoding_scaling(results)
    print(summary)
    fig, ax = plt.subplots()
    ax.set_prop_cycle(color=colors)

    # To make sure the legend and colors stay the same between experiments, we use a solver order argument
    for i, solver in enumerate(solver_order):
        z = 10 + (len(solver_order)-i)
        x = []
        y = []
        for r in summary:
            if r["solver"] == solver:
                x.append(r["size"])
                y.append(r["assertions"]+r["variables"])
        ax.plot(x, y, marker=".", linestyle=SOLVER_LINE_STYLES[solver], color=SOLVER_COLORS[solver], alpha=0.85, label=solver, zorder=z)
    
    ax.set_yscale("log")
    ax.set_xlabel("Puzzle size (n)")
    ax.set_ylabel("Encoding size (vars + assertions)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    legend = ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        frameon=True,
        framealpha=0.85,
        fancybox=False
    )
    legend.set_zorder(20+len(solver_order))
    frame = legend.get_frame()
    frame.set_alpha(0.75)
    fig.tight_layout()
    out_path = os.path.join(os.path.abspath("plots"), f"encoding_scaling.png")
    fig.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")

def print_rq1_text_stats(results: list, baseline_solver: str = "qf_ia", report_sizes: list|None = None) -> None:
    """ Prints the statistics used in RQ1

    Args:
        results (list): Results from the experiments
        baseline_solver (str, optional): Name of the solver to be used as baseline. Defaults to "qf_ia".
        report_sizes (list | None, optional): Sizes to report in the summary. Defaults to None.

    Raises:
        ValueError: If the baseline solver is not in the data
    """
    summary = _summarize_runtime_scaling(results)

    # Sort by solver
    by_solver_size = defaultdict(dict)
    solvers = sorted(set(r["solver"] for r in summary))
    sizes = sorted(set(r["size"] for r in summary))
    for r in summary:
        by_solver_size[r["solver"]][r["size"]] = r

    if baseline_solver not in by_solver_size:
        raise ValueError(f"baseline_solver '{baseline_solver}' not found in data.")

    baseline_sizes = sorted(by_solver_size[baseline_solver].keys())
    if report_sizes is None:
        if not baseline_sizes:
            report_sizes = []
        else:
            mid = baseline_sizes[len(baseline_sizes) // 2]
            report_sizes = sorted(set([baseline_sizes[0], mid, baseline_sizes[-1]]))

    print(f"Baseline solver for speedups: {baseline_solver}")
    print(f"Reported sizes: {report_sizes}")

    print("- Median runtime & speedup/slowdown vs baseline (median_baseline / median_solver) -")
    for n in report_sizes:
        if n not in by_solver_size[baseline_solver]:
            continue
        base_med = by_solver_size[baseline_solver][n]["median"]

        print(f"\nSize n={n}: baseline median={base_med:.6g}s")
        for s in solvers:
            if n not in by_solver_size[s]:
                continue
            med = by_solver_size[s][n]["median"]
            speedup = (base_med/med) if med > 0 else float("inf")
            slowdown = (med/base_med) if med > 0 else float("inf")
            print(f" {s:>12}: median={med:.6g}s | speedup={speedup:.3g}x | slowdown={slowdown:.3g}x")

    print("- Relative variability per size: (q3 - q1) -")
    for n in report_sizes:
        print(f"\nSize n={n}:")
        for s in solvers:
            if n not in by_solver_size[s]:
                continue
            row = by_solver_size[s][n]
            med, q1, q3 = row["median"], row["q1"], row["q3"]
            rel_iqr = ((q3-q1)/med) if med > 0 else float("nan")
            print(f" {s:>12}: rel_IQR={rel_iqr:.3g} (q1={q1:.6g}, q3={q3:.6g})")
    
    print("- Average relative variability: (q3 - q1) -")
    avgs = {}
    counts = {}
    for n in sizes:
        for s in solvers:
            if s not in avgs:
                avgs[s] = 0
                counts[s] = 0
            if n not in by_solver_size[s]:
                continue
            if by_solver_size[s][n]["median"] > 10:
                continue

            row = by_solver_size[s][n]
            med, q1, q3 = row["median"], row["q1"], row["q3"]
            rel_iqr = (q3-q1)/med if med > 0 else float("nan")
            avgs[s] += rel_iqr
            counts[s] += 1

    for s in avgs:
        if counts[s] > 0:
            avgs[s] /= counts[s]
        print(f"{s:>12}: avg_rel_IQR={avgs[s]:.3g} (n={counts[s]})")

    print("- Fit log(median_runtime) = a*n + b -")
    for s in solvers:
        xs, ys = [], []
        for n in sizes:
            row = by_solver_size[s].get(n)
            if not row:
                continue
            med = row["median"]
            if med <= 0:
                continue
            xs.append(n)
            ys.append(np.log(med))

        if len(xs) < 3:
            print(f" {s:>12}: not enough points to fit (have {len(xs)})")
            continue

        a, b = np.polyfit(np.array(xs, dtype=float), np.array(ys, dtype=float), 1)
        yhat = a*np.array(xs, dtype=float)+b
        ss_res = float(np.sum((np.array(ys)-yhat)**2))
        ss_tot = float(np.sum((np.array(ys)-np.mean(ys))**2))
        r2 = 1.0-(ss_res/ss_tot) if ss_tot > 0 else float("nan")
        print(f" {s:>12}: a={a:.4g} | b={b:.4g} | R^2={r2:.3g}")

def print_encoding_text_stats(results: list, report_sizes: list|None = None) -> None:
    """ Prints the encoding statistics used in RQ1

    Args:
        results (list): Results from the experiments
        report_sizes (list | None, optional): which sizes to print. If None, prints min/mid/max sizes in data.
    """
    enc_summary = _summarize_encoding_scaling(results)
    by_solver_size = defaultdict(dict)
    solvers = sorted(set(r["solver"] for r in enc_summary))
    sizes = sorted(set(r["size"] for r in enc_summary))
    for r in enc_summary:
        by_solver_size[r["solver"]][r["size"]] = r
    if report_sizes is None and sizes:
        mid = sizes[len(sizes)//2]
        report_sizes = sorted(set([sizes[0], mid, sizes[-1]]))

    for n in report_sizes or []:
        print(f"\nSize n={n}:")
        for s in solvers:
            row = by_solver_size[s].get(n)
            if not row:
                continue
            total = row["variables"] + row["assertions"]
            print(f" {s:>12}: total={total} | vars={row['variables']} | assertions={row['assertions']}")