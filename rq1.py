from collections import defaultdict
import numpy as np
from numpy.polynomial import Polynomial
import scipy
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import scienceplots

plt.style.use(['science','ieee'])

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

colors = [
    "#D62728", 
    "#006BA4", 
    "#8C564B", 
    "#2CA02C", 
    "#7B4EA3",
    "#FF800E", 
    "#E43D96",
    "#17BECF", 
    "#BCBD22",  
    "#7F7F7F"
]

LINE_STYLES = ["-", "--", "-.", ":"]

def _summarize_runtime_scaling(results) -> list:
    grouped = {}
    for r in results:
        key = (r["solver"], r["size"])
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(r["statistics"]["runtime"])

    summary = []
    for (solver, size), runtimes in grouped.items():
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

def _summarize_encoding_scaling(results):
    grouped = {}
    for r in results:
        key = (r["solver"], r["size"])
        if key not in grouped:
            grouped[key] = []
        vars = sum([r["statistics"]["encoding_size"][key] for key in r["statistics"]["encoding_size"].keys() if key != "assertions"])
        grouped[key].append((vars, r["statistics"]["encoding_size"]["assertions"]))

    summary = []
    for (solver, size), values in grouped.items():
        variables, assertions = zip(*values)
        summary.append({
            "solver": solver,
            "size": size,
            "variables": int(np.median(variables)),
            "assertions": int(np.median(assertions)),
        })

    return sorted(summary, key=lambda x: (x["size"], x["solver"]))

def plot_runtime_vs_size(results: list, solver_order: list) -> None:
    summary = _summarize_runtime_scaling(results)

    fig, ax = plt.subplots()
    ax.set_prop_cycle(color=colors)

    for i, solver in enumerate(solver_order):
        z = 10+(len(solver_order)-i)
        x = []
        y = []
        for r in summary:
            if r["solver"] == solver:
                x.append(r["size"])
                y.append(r["median"])
        ax.plot(x, y, marker=".", linestyle=LINE_STYLES[i % len(LINE_STYLES)], alpha=0.85, label=solver, zorder=z)
    
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
    summary = _summarize_encoding_scaling(results)

    fig, ax = plt.subplots()
    ax.set_prop_cycle(color=colors)

    for i, solver in enumerate(solver_order):
        z = 10 + (len(solver_order)-i)
        x = []
        y = []
        for r in summary:
            if r["solver"] == solver:
                x.append(r["size"])
                y.append(r["assertions"]+r["variables"])
        ax.plot(x, y, marker=".", linestyle=LINE_STYLES[i % len(LINE_STYLES)], alpha=0.85, label=solver, zorder=z)
    
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

def print_rq1_text_stats(results: list, baseline_solver: str = "qf_ia", report_sizes: list|None = None, slope_min_points: int = 3) -> None:
    """
    Prints several stats used to answer RQ1
    
    Args:
        results: list of result dicts
        baseline_solver: solver used as baseline for speedups
        report_sizes: which sizes to print. If None, prints min/mid/max sizes in data.
        slope_min_points: minimum points needed to fit a slope.
    """
    summary = _summarize_runtime_scaling(results)

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

    print("\n=== RQ1 descriptive statistics (runtime) ===")
    print(f"Baseline solver for speedups: {baseline_solver}")
    print(f"Reported sizes: {report_sizes}")

    print("\n-- Median runtime & speedup vs baseline (median_baseline / median_solver) --")
    for n in report_sizes:
        if n not in by_solver_size[baseline_solver]:
            continue
        base_med = by_solver_size[baseline_solver][n]["median"]

        print(f"\nSize n={n}: baseline median={base_med:.6g}s")
        for s in solvers:
            if n not in by_solver_size[s]:
                continue
            med = by_solver_size[s][n]["median"]
            speedup = (base_med / med) if med > 0 else float("inf")
            print(f"  {s:>12}: median={med:.6g}s  speedup={speedup:.3g}x")

    print("\n-- Relative variability: (q3 - q1) --")
    for n in report_sizes:
        print(f"\nSize n={n}:")
        for s in solvers:
            if n not in by_solver_size[s]:
                continue
            row = by_solver_size[s][n]
            med, q1, q3 = row["median"], row["q1"], row["q3"]
            rel_iqr = ((q3 - q1) / med) if med > 0 else float("nan")
            print(f"  {s:>12}: rel_IQR={rel_iqr:.3g}  (q1={q1:.6g}, q3={q3:.6g})")

    print("\n-- Scaling slope: fit log(median_runtime) = a*n + b --")
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

        if len(xs) < slope_min_points:
            print(f"  {s:>12}: not enough points to fit (have {len(xs)})")
            continue

        a, b = np.polyfit(np.array(xs, dtype=float), np.array(ys, dtype=float), 1)
        yhat = a * np.array(xs, dtype=float) + b
        ss_res = float(np.sum((np.array(ys) - yhat) ** 2))
        ss_tot = float(np.sum((np.array(ys) - np.mean(ys)) ** 2))
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")

        print(f"  {s:>12}: a={a:.4g}  b={b:.4g}  R^2={r2:.3g}")

def print_encoding_text_stats(results: list, report_sizes: list[int] | None = None) -> None:
    """
    Optional: prints encoding size (vars+assertions) at selected sizes, per solver.
    Since you sum them in the plot, this prints the same scalar, plus components.
    """
    enc_summary = _summarize_encoding_scaling(results)
    by_solver_size = defaultdict(dict)
    solvers = sorted(set(r["solver"] for r in enc_summary))
    sizes = sorted(set(r["size"] for r in enc_summary))
    for r in enc_summary:
        by_solver_size[r["solver"]][r["size"]] = r

    if report_sizes is None and sizes:
        mid = sizes[len(sizes) // 2]
        report_sizes = sorted(set([sizes[0], mid, sizes[-1]]))

    print("\n=== RQ1 descriptive statistics (encoding size) ===")
    print(f"Reported sizes: {report_sizes}")

    for n in report_sizes or []:
        print(f"\nSize n={n}:")
        for s in solvers:
            row = by_solver_size[s].get(n)
            if not row:
                continue
            total = row["variables"] + row["assertions"]
            print(f"  {s:>12}: total={total}  vars={row['variables']}  assertions={row['assertions']}")