import os
import matplotlib.pyplot as plt
import scienceplots

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

plt.rcParams["axes.prop_cycle"] = plt.cycler(color=[
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#a6cee3", "#fb9a99", "#fdbf6f", "#b2df8a", "#cab2d6",
    "#ffff99", "#33a02c", "#b15928", "#e31a1c", "#1f78b4"
])

LINE_STYLES = ["-", "--", "-.", ":"]

def _plot_puzzleruntime(results):
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
            by_solver[solver][e["puzzle"]] = e["elapsed"]

        plt.figure()
        for index, solver in enumerate(solver_order):
            if solver not in by_solver:
                continue
            puzzle_map = by_solver[solver]

            ys = [puzzle_map.get(p, float("nan")) for p in puzzles]
            z = 10 + (len(solver_order)-index)
            plt.plot(x, ys, marker=".", linestyle=LINE_STYLES[index % len(LINE_STYLES)], alpha=0.85, label=solver, zorder=z)

        plt.legend(markerscale=0.5, handlelength=1.2, borderpad=0.2, labelspacing=0.2)
        plt.xticks(x)
        plt.xlabel("Puzzle index")
        plt.ylabel("Runtime (s)")
        plt.title(f"Runtime per puzzle for size {size}x{size}")
        plt.legend()
        plt.tight_layout()

        out_path = os.path.join(os.path.abspath("plots"), f"runtime_n{size}.png")
        plt.savefig(out_path)
        plt.close()
        print(f"Saved {out_path}")

def _plot_puzzlestatistic(results, statistic, y_label):
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

def _plot_puzzleconflicts(results):
    _plot_puzzlestatistic(results, "conflicts", "Conflicts")
def _plot_puzzlepropagations(results):
    _plot_puzzlestatistic(results, "propagations", "Propagations")
def _plot_puzzledecisions(results):
    _plot_puzzlestatistic(results, "decisions", "Decisions")
def _plot_puzzleboolvars(results):
    _plot_puzzlestatistic(results, "bool_vars", "Boolean variables")
def _plot_puzzleclauses(results):
    _plot_puzzlestatistic(results, "clauses", "Clauses")
def _plot_puzzlebinclauses(results):
    _plot_puzzlestatistic(results, "bin_clauses", "Binary clauses")
    
def _plot_avgruntime(results):
    sums = {}
    counts = {}

    for r in results:
        key = (r["size"], r["solver"])
        if key not in sums:
            sums[key] = 0.0
            counts[key] = 0
        sums[key] += r["elapsed"]
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
        plt.ylabel("Average runtime (s)")
        plt.title(f"Average runtime per solver for size {size}x{size}")
        plt.tight_layout()
        
        out_path = os.path.join(os.path.abspath("plots"), f"avg_runtime_n{size}.png")
        plt.savefig(out_path)
        plt.close()
        print(f"Saved {out_path}")

def _plot_avgstatistic(results, statistic, y_label):
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

def _plot_avgconflicts(results):
    _plot_avgstatistic(results, "conflicts", "Averge conflicts")
def _plot_avgpropagations(results):
    _plot_avgstatistic(results, "propagations", "Averge propagations")
def _plot_avgdecisions(results):
    _plot_avgstatistic(results, "decisions", "Averge decisions")
def _plot_avgboolvars(results):
    _plot_avgstatistic(results, "bool_vars", "Averge Boolean variables")
def _plot_avgclauses(results):
    _plot_avgstatistic(results, "clauses", "Averge clauses")
def _plot_avgbinclauses(results):
    _plot_avgstatistic(results, "bin_clauses", "Averge binary clauses")

def _plot_scalingavgruntime(results):
    sums = {}
    counts = {}

    for r in results:
        key = (r["size"], r["solver"])
        if key not in sums:
            sums[key] = 0.0
            counts[key] = 0
        sums[key] += r["elapsed"]
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
    plt.ylabel("Average runtime (s)")
    plt.title("Average runtime by puzzle size")
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(os.path.abspath("plots"), f"scaling_avg_runtime.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")

def _plot_scalingavgstatistic(results, statistic, y_label, plot_name):
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
        z = 10 + (len(solver)-index)
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

def _plot_scalingavgconflicts(results):
    _plot_scalingavgstatistic(results, "conflicts", "Average number of conflicts", "Average number of conflicts by puzzle size")
def _plot_scalingavgpropagations(results):
    _plot_scalingavgstatistic(results, "propagations", "Average number of propagations", "Average number of propagations by puzzle size")
def _plot_scalingavgdecisions(results):
    _plot_scalingavgstatistic(results, "decisions", "Average number of decisions", "Average number of decisions by puzzle size")
def _plot_scalingavgboolvars(results):
    _plot_scalingavgstatistic(results, "bool_vars", "Average number of Boolean variables", "Average encoding size by puzzle size (bool_vars)")
def _plot_scalingavgclauses(results):
    _plot_scalingavgstatistic(results, "clauses", "Average number of clauses", "Average encoding size by puzzle size (clauses)")
def _plot_scalingavgbinclauses(results):
    _plot_scalingavgstatistic(results, "bin_clauses", "Average number of binary clauses", "Average encoding size by puzzle size (bin_clauses)")

PLOT_TYPES = {
    1: {
        "f": _plot_puzzleruntime,
        "description": "Runtime vs puzzle (per size, line plot, one line per solver)",
        "requires": {
            "min_puzzles": 1,
            "min_solvers": 1
        }
    },
    1_1: {
        "f": _plot_puzzleconflicts,
        "description": "Conflict vs puzzle (per size, line plot, one line per solver)",
        "requires": {
            "min_puzzles": 1,
            "min_solvers": 1
        }
    },
    1_2: {
        "f": _plot_puzzledecisions,
        "description": "Decisions vs puzzle (per size, line plot, one line per solver)",
        "requires": {
            "min_puzzles": 1,
            "min_solvers": 1
        }
    },
    1_3: {
        "f": _plot_puzzlepropagations,
        "description": "Propagations vs puzzle (per size, line plot, one line per solver)",
        "requires": {
            "min_puzzles": 1,
            "min_solvers": 1
        }
    },
    1_4: {
        "f": _plot_puzzleboolvars,
        "description": "Boolean variables vs puzzle (per size, line plot, one line per solver)",
        "requires": {
            "min_puzzles": 1,
            "min_solvers": 1
        }
    },
    1_5: {
        "f": _plot_puzzleclauses,
        "description": "Clauses vs puzzle (per size, line plot, one line per solver)",
        "requires": {
            "min_puzzles": 1,
            "min_solvers": 1
        }
    },
    1_6: {
        "f": _plot_puzzlebinclauses,
        "description": "Binary clauses vs puzzle (per size, line plot, one line per solver)",
        "requires": {
            "min_puzzles": 1,
            "min_solvers": 1
        }
    },
    2: {
        "f": _plot_avgruntime,
        "description": "Average runtime per solver and size (bar chart)",
        "requires": {
            "min_puzzles": 1,
            "min_solvers": 1
        }
    },
    2_1: {
        "f": _plot_avgconflicts,
        "description": "Average conflicts per solver and size (bar chart)",
        "requires": {
            "min_puzzles": 1,
            "min_solvers": 1
        }
    },
    2_2: {
        "f": _plot_avgpropagations,
        "description": "Average propagations per solver and size (bar chart)",
        "requires": {
            "min_puzzles": 1,
            "min_solvers": 1
        }
    },
    2_3: {
        "f": _plot_avgdecisions,
        "description": "Average decisions per solver and size (bar chart)",
        "requires": {
            "min_puzzles": 1,
            "min_solvers": 1
        }
    },
    2_4: {
        "f": _plot_avgboolvars,
        "description": "Average Boolean variables per solver and size (bar chart)",
        "requires": {
            "min_puzzles": 1,
            "min_solvers": 1
        }
    },
    2_5: {
        "f": _plot_avgclauses,
        "description": "Average clauses per solver and size (bar chart)",
        "requires": {
            "min_puzzles": 1,
            "min_solvers": 1
        }
    },
    2_6: {
        "f": _plot_avgbinclauses,
        "description": "Average binary clauses per solver and size (bar chart)",
        "requires": {
            "min_puzzles": 1,
            "min_solvers": 1
        }
    },
    3: {
        "f": _plot_scalingavgruntime,
        "description": "Average runtime by puzzle size",
        "requires": {
            "min_puzzles": 1,
            "min_solvers": 1
        }
    },
    3_1: {
        "f": _plot_scalingavgconflicts,
        "description": "Average conflicts by puzzle size",
        "requires": {
            "min_puzzles": 1,
            "min_solvers": 1
        }
    },
    3_2: {
        "f": _plot_scalingavgpropagations,
        "description": "Average propagations by puzzle size",
        "requires": {
            "min_puzzles": 1,
            "min_solvers": 1
        }
    },
    3_3: {
        "f": _plot_scalingavgdecisions,
        "description": "Average decisions by puzzle size",
        "requires": {
            "min_puzzles": 1,
            "min_solvers": 1
        }
    },
    3_4: {
        "f": _plot_scalingavgboolvars,
        "description": "Average Boolean variables by puzzle size",
        "requires": {
            "min_puzzles": 1,
            "min_solvers": 1
        }
    },
    3_5: {
        "f": _plot_scalingavgclauses,
        "description": "Average clauses by puzzle size",
        "requires": {
            "min_puzzles": 1,
            "min_solvers": 1
        }
    },
    3_6: {
        "f": _plot_scalingavgbinclauses,
        "description": "Average binary clauses by puzzle size",
        "requires": {
            "min_puzzles": 1,
            "min_solvers": 1
        }
    }
}

def plot(id: int, results: list[dict]) -> None:
    if id not in PLOT_TYPES:
        print(f"Unknown plot id {id}")
        return
    
    meta = PLOT_TYPES[id]
    f = meta["f"]
    req = meta["requires"]

    puzzle_set = {r["puzzle"] for r in results}
    solver_set = {r["solver"] for r in results}

    min_puzzles = req.get("min_puzzles", 1)
    min_solvers = req.get("min_solvers", 1)
    if len(puzzle_set) < min_puzzles:
        print(f"Plot {id} requires at least {min_puzzles} puzzle(s)")
        return
    if len(solver_set) < min_solvers:
        print(f"Plot {id} requires at least {min_solvers} solver(s)")
        return
    
    os.makedirs(os.path.abspath("plots"), exist_ok=True)
    print(f"Generating plot {id}: {meta['description']}")
    f(results)