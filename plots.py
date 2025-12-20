import os
import matplotlib.pyplot as plt
import scienceplots
import scipy
import numpy as np

plt.style.use(['science','ieee'])

plt.rcParams.update({
    "lines.linewidth": 0.8,
    "lines.markersize": 1,
    "legend.fontsize": 6,
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

plt.rcParams["axes.prop_cycle"] = plt.cycler(color=[ # type: ignore
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#a6cee3", "#fb9a99", "#fdbf6f", "#b2df8a", "#cab2d6",
    "#ffff99", "#33a02c", "#b15928", "#e31a1c", "#1f78b4"
])

LINE_STYLES = ["-", "--", "-.", ":"]

def _log_differences(results: list, statistic, solver1: str, solver2: str, size: int) -> list:
    a = {}
    b = {}

    for r in results:
        if r["size"] == size:
            if r["solver"] == solver1:
                a[r["puzzle"]] = r["statistics"][statistic]
            elif r["solver"] == solver2:
                b[r["puzzle"]] = r["statistics"][statistic]

    common = sorted(set(a) & set(b))
    differences = [a[puzzle]-b[puzzle] for puzzle in common]
    return differences

def plot_qq(results: list, statistic, solver1: str, solver2: str, size: int) -> None:
    differences = _log_differences(results, statistic, solver1, solver2, size)

    scipy.stats.probplot(differences, plot=plt)
    plt.title("Q–Q plot of paired log-runtime differences")
    plt.show()

def _plot_puzzlestatistic(results: dict, statistic: str, y_label: str) -> None:
    by_size = {}
    for r in results:
        size = r["size"]
        if size not in by_size:
            by_size[size] = []
        by_size[r["size"]].append(r)
    
    solver_order = []
    for r in results:
        s = r["solver"]
        if s not in solver_order:
            solver_order.append(s)

    for size, entries in by_size.items():
        puzzles = sorted({e["puzzle"] for e in entries})
        x = list(range(len(puzzles)))

        by_solver = {}
        for e in entries:
            solver = e["solver"]
            if solver not in by_solver:
                by_solver[solver] = {}
            stat = e["statistics"].get(statistic, 0)
            by_solver[solver][e["puzzle"]] = stat

        plt.figure()
        for index, solver in enumerate(solver_order):
            if solver not in by_solver:
                continue
            puzzle_map = by_solver[solver]

            ys = [puzzle_map.get(p, float("nan")) for p in puzzles]
            z = 10 + (len(solver_order)-index)
            plt.plot(x, ys, marker="." , linestyle=LINE_STYLES[index % len(LINE_STYLES)], alpha=0.85, label=solver, zorder=z)
        
        plt.legend(markerscale=0.5, handlelength=1.2, borderpad=0.2, labelspacing=0.2)
        plt.xticks(x)
        plt.xlabel("Puzzle index")
        plt.ylabel(y_label)
        plt.title(f"{y_label} per puzzle for size {size}x{size}")
        plt.legend()
        plt.tight_layout()

        out_path = os.path.join(os.path.abspath("plots"), f"{statistic}_n{size}.png")
        plt.savefig(out_path)
        plt.close()
        print(f"Saved {out_path}")

def _plot_avgstatistic(results: dict, statistic: str, y_label: str) -> None:
    sums = {}
    counts = {}

    for r in results:
        key = (r["size"], r["solver"])
        if key not in sums:
            sums[key] = 0.0
            counts[key] = 0
        sums[key] += r["statistics"][statistic]
        counts[key] += 1

    sizes = sorted({size for (size, _) in sums.keys()})
    solvers = sorted({solver for (_, solver) in sums.keys()})

    for size in sizes:
        avg_per_solver = []
        solver_labels = []
        for solver in solvers:
            key = (size, solver)
            if key in sums:
                avg = sums[key] / counts[key]
                avg_per_solver.append(avg)
                solver_labels.append(solver)
        
        if not avg_per_solver:
            continue

        plt.figure()
        x = list(range(len(solver_labels)))
        plt.bar(x, avg_per_solver)
        plt.legend(markerscale=0.5, handlelength=1.2, borderpad=0.2, labelspacing=0.2)
        plt.xticks(x, solver_labels, rotation=45, ha='right')
        plt.xlabel("Solver")
        plt.ylabel(y_label)
        plt.title(f"{y_label} per solver for size {size}x{size}")
        plt.tight_layout()
        
        out_path = os.path.join(os.path.abspath("plots"), f"avg_{statistic}_n{size}.png")
        plt.savefig(out_path)
        plt.close()
        print(f"Saved {out_path}")

def _plot_scalingavgstatistic(results: dict, statistic: str, y_label: str, plot_name: str) -> None:
    sums = {}
    counts = {}

    for r in results:
        key = (r["size"], r["solver"])
        if key not in sums:
            sums[key] = 0.0
            counts[key] = 0
        sums[key] += r["statistics"].get(statistic, 0)
        counts[key] += 1

    sizes = sorted({size for (size, _) in sums.keys()})
    solvers = sorted({solver for (_, solver) in sums.keys()})

    plt.figure()
    for index, solver in enumerate(solvers):
        z = 10 + (len(solvers)-index)
        x_sizes = []
        y_avgs = []
        for size in sizes:
            key = (size, solver)
            if key in sums:
                avg = sums[key] / counts[key]
                x_sizes.append(size)
                y_avgs.append(avg)
        if x_sizes:
            plt.plot(x_sizes, y_avgs, marker=".", linestyle=LINE_STYLES[index % len(LINE_STYLES)], alpha=0.85, label=solver, zorder=z)

    plt.legend(markerscale=0.5, handlelength=1.2, borderpad=0.2, labelspacing=0.2)
    plt.xlabel("Puzzle size n (n x n)")
    plt.ylabel(y_label)
    plt.title(plot_name)
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(os.path.abspath("plots"), f"scaling_avg_{statistic}.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")

PLOT_TYPES = {
    1: {
        "f": _plot_puzzlestatistic,
        "args": ["runtime", "Runtime"]
    },
    1_1: {
        "f": _plot_puzzlestatistic,
        "args": ["conflicts", "Conflicts"]
    },
    1_2: {
        "f": _plot_puzzlestatistic,
        "args": ["propagations", "Propagations"]
    },
    1_3: {
        "f": _plot_puzzlestatistic,
        "args": ["decisions", "Decisions"]
    },
    2: {
        "f": _plot_avgstatistic,
        "args": ["runtime", "Averge runtime"]
    },
    2_1: {
        "f": _plot_avgstatistic,
        "args": ["conflicts", "Averge conflicts"]
    },
    2_2: {
        "f": _plot_avgstatistic,
        "args": ["propagations", "Averge propagations"]
    },
    2_3: {
        "f": _plot_avgstatistic,
        "args": ["decisions", "Averge decisions"]
    },
    3: {
        "f": _plot_scalingavgstatistic,
        "args": ["runtime", "Average runtime", "Average runtime by puzzle size"]
    },
    3_1: {
        "f": _plot_scalingavgstatistic,
        "args": ["conflicts", "Average number of conflicts", "Average number of conflicts by puzzle size"]
    },
    3_2: {
        "f": _plot_scalingavgstatistic,
        "args": ["propagations", "Average number of propagations", "Average number of propagations by puzzle size"]
    },
    3_3: {
        "f": _plot_scalingavgstatistic,
        "args": ["decisions", "Average number of decisions", "Average number of decisions by puzzle size"]
    }
}

def plot_stat(id: int, results: list) -> None:
    if id not in PLOT_TYPES:
        print(f"Unknown plot id {id}")
        return
    
    meta = PLOT_TYPES[id]
    f = meta["f"]
    args = meta["args"]

    puzzle_set = {r["puzzle"] for r in results}
    solver_set = {r["solver"] for r in results}
    
    os.makedirs(os.path.abspath("plots"), exist_ok=True)
    print(f"Generating plot {id}: {meta['description']}")
    f(results, *args)