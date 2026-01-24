import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from copy import deepcopy
import os
from matplotlib.colors import ListedColormap, BoundaryNorm, TwoSlopeNorm
import matplotlib.patches as patches
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

LINE_STYLES = ["-", "--", "-.", ":"]

def _flatten_results(results: list) -> list:
    """ Flattens the results from multiple runs into a single value

    Args:
        results (list): Results from the experiments

    Returns:
        list: A flattened list of the results
    """
    grouped = {}
    for r in results:
        key = (r["solver"], r["size"], r["puzzle"])
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(r)
    
    flattened = []
    for (_, _, _), rs in grouped.items():
        base = deepcopy(rs[0])
        runtimes = [float(r["statistics"]["runtime"]) for r in rs]
        base["statistics"]["runtime"] = float(np.median(runtimes))

        encoding_keys = rs[0]["statistics"]["encoding_size"]
        for key in encoding_keys.keys():
            values = [float(r["statistics"]["encoding_size"][key]) for r in rs]
            if values:
                base["statistics"]["encoding_size"][key] = float(np.median(values))
        flattened.append(base)
    return flattened

def _encoding_total(result: dict) -> int:
    """ Counts the total encoding size

    Args:
        result (dict): Results for which we want to count the encoding size.

    Returns:
        int: The number for the encoding size
    """
    encoding = result["statistics"]["encoding_size"]
    vars = sum(encoding[k] for k in encoding.keys() if k != "assertions")
    return int(vars+encoding["assertions"])

def _short_label(name: str, baseline: str) -> str:
    """ Get the name of the solver without the baseline part

    Args:
        name (str): Full name of the solver
        baseline (str): Baseline that is used for this solver

    Returns:
        str: Name of the solver without the baseline
    """
    prefix = baseline+"+"
    return name[len(prefix):] if name.startswith(prefix) else name

def _order_constraints(matrix: dict, relevance: dict, baseline: str) -> list:
    """ Order the constraints in the significance grid by their significance, used for plotting

    Args:
        matrix (dict): Significance grid
        relevance (dict): Statistics related to statistical relevance for each constraint
        baseline (str): Baseline solver used

    Returns:
        list: An ordering of constraints based on the amount and direction of significant values
    """
    def stats(constraint: str):
        significant_sizes = relevance[constraint]["sizes"]
        directions = [matrix[constraint][n][1] for n in significant_sizes]
        faster = sum(1 for dir in directions if dir==1)
        slower = sum(1 for dir in directions if dir==-1)
        mixed = (faster > 0 and slower > 0)
        relevant = relevance[constraint]["relevant"]

        return relevant, faster, slower, mixed, len(significant_sizes)
    def key(constraint: str):
        relevant, faster, slower, mixed, k = stats(constraint)
        # Lower rank number is higher sort order
        if faster > 0:
            rank = 0
        elif mixed:
            rank = 1
        elif relevant:
            rank = 2
        else:
            rank = 3
        
        return (rank, -k, -faster, -slower, _short_label(constraint, baseline))
    return sorted(matrix.keys(), key=key)

def _pair_by_size(results: list, size: int, this_name: str, that_name: str) -> tuple:
    """ Gather all runtimes for the same puzzle ran by two solvers, within a single puzzle size

    Args:
        results (list): Results from the experiments
        size (int): Puzzle size to pair
        this_name (str): One of the solvers to pair
        that_name (str): The other solver to pair

    Returns:
        tuple: Tuple containing two lists of runtimes from each of the solvers
    """
    this = {}
    that = {}

    for r in results:
        if r["size"] != size:
            continue
        puzzle = r["puzzle"]
        solver = r["solver"]
        if solver == this_name:
            this[puzzle] = r["statistics"]["runtime"]
        elif solver == that_name:
            that[puzzle] = r["statistics"]["runtime"]
    
    # Filter out puzzles that are not in both sets
    common = sorted(set(this.keys()) & set(that.keys()))
    if not common:
        return np.array([], dtype=float), np.array([], dtype=float)
    
    this_common = np.array([this[p] for p in common], dtype=float)
    that_common = np.array([that[p] for p in common], dtype=float)
    return this_common, that_common

def _pair_median_runtime_by_size(results: list, size: int, baseline: str, solver: str) -> tuple[float, float, int]:
    """ Get the median runtimes from a paired group of solvers

    Args:
        results (list): Results from the experiments
        size (int): Puzzle size to pair
        baseline (str): Baseline solver to be paired with
        solver (str): Other solver to pair with the baseline

    Returns:
        tuple[float, float, int]: _description_
    """
    base, other = _pair_by_size(results, size, baseline, solver)
    # If the base list is empty, there were no equal puzzles between the sets
    if len(base) == 0:
        return (float("nan"), float("nan"), 0)
    return (float(np.median(base)), float(np.median(other)), int(len(base)))

def _holm_correction(p_values: dict, alpha: float) -> dict:
    """ Use holm correction to filter out significant values per size within a constraint

    Args:
        p_values (dict): Dict of p_values
        alpha (float): Alpha to use as threshold

    Returns:
        dict: list of solvers that holds True if not significant
    """
    keys = list(p_values.keys())
    ps = np.array([p_values[k] for k in keys], dtype=float)

    ps[np.isnan(ps)] = 1.0
    order = np.argsort(ps)
    n = len(ps)

    rejected = {k: False for k in keys}
    for i, index in enumerate(order):
        if ps[index] <= alpha/(n-i):
            rejected[keys[index]] = True
        else:
            break
    
    return rejected

def _classify_constraints(matrix: dict, alpha: float) -> dict:
    """ Calculate the relevance of a constraint based on the p-values

    Args:
        matrix (dict): Siginicance matrix
        alpha (float): Alpha to used as a threshold for the Holm-correction

    Returns:
        dict: A Dict of the relevance data
    """
    relevance = {}

    for c, sizes in matrix.items():
        p_values = {n: sizes[n][0] for n in sizes}
        rejected = _holm_correction(p_values, alpha)
        significant_sizes = [k for k, v in rejected.items() if v]

        relevance[c] = {
            "relevant": len(significant_sizes) > 0,
            "sizes": significant_sizes,
            "directions": {n: sizes[n][1] for n in significant_sizes}
        }
    
    return relevance

def _significance_grid(matrix: dict, relevance: dict, baseline: str) -> tuple:
    """ Create a grid of only the significant values

    Args:
        matrix (dict): Significance matrix
        relevance (dict): Statistics related to statistical relevance for each constraint
        baseline (str): Baseline solver used

    Returns:
        tuple: A grid with only the significant values
    """
    constraint_order = _order_constraints(matrix, relevance, baseline) 
    size_order = sorted({n for c in matrix for n in matrix[c].keys()})
    grid = np.zeros((len(constraint_order), len(size_order)), dtype=int)

    for i, c in enumerate(constraint_order):
        significant = set(relevance[c]["sizes"])
        for j, n in enumerate(size_order):
            if n in significant:
                grid[i, j] = int(matrix[c][n][1])

    constraint_labels = [_short_label(c, baseline) for c in constraint_order]
    return constraint_order, constraint_labels, size_order, grid

def _wilcoxon_by_constraint(results: list, baseline: str, constraints: list|None, sizes: list|None) -> dict:
    """ Calculate the Wilcoxon signed-rank test per constraint per puzzle size

    Args:
        results (list): Results from the experiments
        baseline (str): Baseline used in the experiments
        constraints (list | None): List of constraints added on top of the baseline
        sizes (list | None): Puzzle sizes used in the experiments

    Returns:
        dict: A matrix of p-values per constraint per puzzle size
    """
    if constraints is None:
        constraints = sorted({r["solver"] for r in results if r["solver"] != baseline})
    if sizes is None:
        sizes = sorted({r["size"] for r in results})
    
    matrix = {c: {} for c in constraints}
    for c in constraints:
        for n in sizes:
            s1, s2 = _pair_by_size(results, n, baseline, c)
            if len(s1) == 0 or len(s2) == 0:
                matrix[c][n] = (np.nan, 0)
                continue

            difference  = s2-s1
            median_difference = float(np.median(difference))
            direction = 0
            if median_difference < 0:
                direction = 1
            elif median_difference > 0:
                direction = -1
            
            try:
                stat = scipy.stats.wilcoxon(s1, s2)
                p = float(stat.pvalue)
            except ValueError:
                p = 1.0
                direction = 0
            
            matrix[c][n] = (p, direction)
    return matrix

def _print_wolcoxon(relevance: dict) -> None:
    """ Print a summary of the Wilcoxon test

    Args:
        relevance (dict): Statistics related to statistical relevance for each constraint
    """
    relevant = [(c, r) for c, r in relevance.items() if r["relevant"]]
    not_relevant = [(c, r) for c, r in relevance.items() if not r["relevant"]]

    relevant.sort(key=lambda x: (-len(x[1]["sizes"]), x[0]))
    not_relevant.sort(key=lambda x: x[0])

    print(f"Relevant constraints: {len(relevant)} / {len(relevance)}")
    print(f"Irrelevant constraints: {len(not_relevant)} / {len(relevance)}")

    if not_relevant:
        print("- Irrelevant (no significant size) -")
        for c, _ in not_relevant:
            print(f"{c}")

    if relevant:
        print("- Relevant (significant at >0 size) -")
        for c, r in relevant:
            parts = []
            for n in r["sizes"]:
                d = r["directions"].get(n, 0)
                parts.append(f"{n}:{'faster' if d == 1 else 'slower' if d == -1 else 'n/a'}")
            print(f" {c}: {', '.join(parts)}")

def _plot_significance_heatmap(constraint_labels: list, size_order: list, significance_grid: np.ndarray) -> None:
    """ Plot a heatmap to showcase significance puzzle sizes per constraint

    Args:
        constraint_labels (list): Labels to be used for the constraint names
        size_order (list): Order of puzzle sizes to be used
        significance_grid (np.ndarray): Significance grid of only values that are significant, else 0
    """
    n_rows, n_cols = significance_grid.shape
    fig_h = max(2.6, 0.32*n_rows+1.2)
    fig_w = max(6.0, 0.38*n_cols+2.2)
    cmap = ListedColormap(["#2B59C3", "#FFFFFF", "#D1495B"]) 
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.grid(False)
    ax.minorticks_off()
    x = np.arange(n_cols+1)
    y = np.arange(n_rows+1)
    m = ax.pcolormesh(x, y, significance_grid, cmap=cmap, norm=norm, edgecolors="0.85", linewidth=0.6)

    ax.invert_yaxis()
    ax.set_yticks(np.arange(n_rows)+0.5)
    ax.set_xticks(np.arange(n_cols)+0.5)
    ax.set_xticklabels([str(n) for n in size_order], fontsize=7)
    ax.set_yticklabels(constraint_labels, fontsize=7)
    ax.tick_params(axis="x", bottom=True, labelbottom=True, length=2.5, width=0.6)
    ax.tick_params(axis="y", left=True, labelleft=True)

    ax.set_xlabel("Puzzle size (n)")
    ax.set_ylabel("Redundant constraint")
    ax.set_title("Wilcoxon vs baseline: significant slower / n.s. / faster", fontsize=8)

    cbar = fig.colorbar(m, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_ticks([-1, 0, 1])
    cbar.set_ticklabels(["slower", "n.s.", "faster"])
    cbar.ax.tick_params(labelsize=6)

    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.14, top=0.92)
    out_path = os.path.join(os.path.abspath("plots"), f"heatmap.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved {out_path}")

def _summarize_speedup_by_constraint(results: list, baseline: str, constraints: list|None = None, sizes: list|None = None) -> dict:
    """ Summarize the total speedup gained per constraint against the baseline solver

    Args:
        results (list): Results from the experiments
        baseline (str): Baseline solver used in the experiments
        constraints (list | None, optional): List of constraints used in the experiments. Defaults to None.
        sizes (list | None, optional): Puzzle sizes used in the experiments. Defaults to None.

    Returns:
        dict: A dictionary containing speedup statistics compared to the baseline for each constraint per puzzle size
    """
    if constraints is None:
        constraints = sorted({r["solver"] for r in results if r["solver"] != baseline})
    if sizes is None:
        sizes = sorted({r["size"] for r in results})

    out = {c: {} for c in constraints}

    for c in constraints:
        for n in sizes:
            base_med, constraint_med, num_pairs = _pair_median_runtime_by_size(results, n, baseline, c)
            if num_pairs == 0 or constraint_med <= 0:
                out[c][n] = {
                    "pairs": num_pairs,
                    "baseline_median": base_med,
                    "solver_median": constraint_med,
                    "speedup": float("nan"),
                    "delta": float("nan"),
                }
                continue

            out[c][n] = {
                "pairs": num_pairs,
                "baseline_median": base_med,
                "solver_median": constraint_med,
                "speedup": float(base_med/constraint_med),
                "delta": float(constraint_med-base_med),
            }

    return out

def _summarize_encoding_ratio_by_constraint(results: list, baseline: str, constraints: list|None = None, sizes: list|None = None) -> dict:
    """ Summarize the encoding size ratios based on the baseline for each redundant constraint added

    Args:
        results (list): Results from the experiments
        baseline (str): Baseline solver used in the experiments
        constraints (list | None, optional): List of constraints used in the experiments. Defaults to None.
        sizes (list | None, optional): Puzzle sizes used in the experiments. Defaults to None.

    Returns:
        dict: Dictionary containing the ratios of encoding sizes for all constraint compared to the baseline
    """
    if constraints is None:
        constraints = sorted({r["solver"] for r in results if r["solver"] != baseline})
    if sizes is None:
        sizes = sorted({r["size"] for r in results})

    totals = {}
    for result in results:
        solver = result["solver"]
        n = result["size"]
        if solver != baseline and solver not in constraints:
            continue
        if n not in sizes:
            continue
        totals.setdefault(solver, {}).setdefault(n, []).append(_encoding_total(result))

    out = {constraint: {} for constraint in constraints}
    for constraint in constraints:
        for n in sizes:
            base_list = totals[baseline][n]
            constraint_list = totals[constraint][n]
            if not base_list or not constraint_list:
                out[constraint][n] = {"baseline_total": None, "solver_total": None, "ratio": float("nan")}
                continue

            base_med = int(np.median(np.array(base_list, dtype=float)))
            constraint_med = int(np.median(np.array(constraint_list, dtype=float)))
            ratio = float(constraint_med/base_med) if base_med > 0 else float("nan")

            out[constraint][n] = {"baseline_total": base_med, "solver_total": constraint_med, "ratio": ratio}
    return out

def print_constraint_summary(speedups: dict, encoding_size_stats: dict, *, sizes: list|None = None, baseline: str = "qf_ia") -> None:
    """ Print a summary for each constraint

    Args:
        speedup (dict): Speedup statistics compared to baseline
        encoding_size_stats (dict): Satistics from encoding sizes compared to the baseline
        sizes (list[int] | None, optional): Sizes to report on. Defaults to None.
        baseline (str, optional): Baseline used in the experiments. Defaults to "qf_ia".
    """
    if sizes is None:
        sizes = sorted({n for constraint in speedups for n in speedups[constraint].keys()})

    n_max = max(sizes)

    print(f"Baseline: {baseline}")
    print(f"Sizes: {sizes} (max n={n_max})")

    for constraint in sorted(speedups.keys()):
        values = []
        for n in sizes:
            value = speedups[constraint][n].get("speedup", float("nan"))
            if np.isfinite(value):
                values.append((n, value))

        best = max(values, key=lambda t: t[1]) if values else (None, float("nan"))
        worst = min(values, key=lambda t: t[1]) if values else (None, float("nan"))

        speedup_at_max = speedups[constraint][n_max].get("speedup", float("nan"))
        size_at_max = encoding_size_stats[constraint][n_max].get("ratio", float("nan"))

        print(f"{constraint}:")
        print(f" best speedup: n={best[0]} | {best[1]:.3g}x")
        print(f" worst speedup: n={worst[0]} | {worst[1]:.3g}x")
        print(f" speedup at {n_max}: {speedup_at_max:.3g}x")
        print(f" encoding_ratio at {n_max}: {size_at_max:.3g}x")

def _speedup_grid(constraint_order: list, size_order: list, matrix: dict, relevance: dict, baseline: str):
    """ Create a grid with all speedup amounts for all constraints and sizes that are significant
        Non-significant values are set to NaN

    Args:
        constraint_order (list): All constraints to be checked
        size_order (list): All sizes to be checked
        matrix (dict): Significance matrix
        relevance (dict): Relevance data for each constraint and size
        baseline (str): Baseline used to compare to in the experiment

    Returns:
        _type_: _description_
    """
    grid = np.full((len(constraint_order), len(size_order)), np.nan, dtype=float)

    for i, c in enumerate(constraint_order):
        significant = set(relevance[c]["sizes"])
        for j, n in enumerate(size_order):
            speedup = matrix[c][n]["speedup"]
            if not np.isfinite(speedup) or speedup <= 0:
                continue
            if n not in significant:
                continue

            grid[i, j] = speedup

    constraint_labels = [_short_label(c, baseline) for c in constraint_order]
    return constraint_labels, size_order, grid

def _plot_speedup_heatmap(constraint_labels: list, size_order: list, speedup_grid: np.ndarray) -> None:
    """ Plot a heatmap of the speedup values for all significant constraints and sizes

    Args:
        constraint_labels (list): Labels to be used to represent the constraints
        size_order (list): Order of the sizes in the heatmap
        speedup_grid (np.ndarray): Grid with all speedup values for significant constraints and sizes
    """
    n_rows, n_cols = speedup_grid.shape
    fig_h = max(2.6, 0.32*n_rows+1.2)
    fig_w = max(6.0, 0.38*n_cols+2.2)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="white")
    vmin = np.min(speedup_grid[np.isfinite(speedup_grid)])
    difference = 1-vmin
    print(vmin)
    norm = TwoSlopeNorm(vcenter=1.0, vmin=vmin, vmax=1+difference)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.grid(False)
    ax.minorticks_off()
    x = np.arange(n_cols+1)
    y = np.arange(n_rows+1)
    m = ax.pcolormesh(x, y, speedup_grid, cmap=cmap, norm=norm, edgecolors="0.85", linewidth=0.6)
    
    mask = ~np.isfinite(speedup_grid)
    for i in range(n_rows):
        for j in range(n_cols):
            if mask[i, j]:
                ax.add_patch(patches.Rectangle((j, i), 1, 1, facecolor="none", hatch="///", edgecolor="0.6", linewidth=0.0))

    ax.invert_yaxis()
    ax.set_yticks(np.arange(n_rows)+0.5)
    ax.set_xticks(np.arange(n_cols)+0.5)
    ax.set_xticklabels([str(n) for n in size_order], fontsize=7)
    ax.set_yticklabels(constraint_labels, fontsize=7)

    ax.set_xlabel("Puzzle size (n)")
    ax.set_ylabel("Redundant constraint")
    ax.set_title("Wilcoxon vs baseline: speedup on significant sizes", fontsize=8)

    cbar = fig.colorbar(m, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("speedup = median(baseline)/median(solver)", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.14, top=0.92)
    out_path = os.path.join(os.path.abspath("plots"), f"heatmap_speedups.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved {out_path}")

def _summarize_runtime(results: list) -> list:
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
        })
    
    return sorted(summary, key=lambda x: (x["size"], x["solver"]))

def _plot_runtime_per_size(results: list, baseline: str, constraints: list, sizes: list) -> None:
    """ Create a plot of a selection of constraints for a selection of sizes

    Args:
        results (list): Results from the experiment
        baseline (str): Baseline that was used to compare to
        constraints (list): Constraints to be plotted
        sizes (list): Puzzle sizes to be plotted
    """
    summary = _summarize_runtime(results)

    fig, ax = plt.subplots()
    ax.set_prop_cycle(color=colors)

    # To make sure the legend and colors stay the same between experiments, we use a solver order argument
    for i, solver in enumerate([baseline] + constraints):
        z = 10+(len(constraints)-i)
        x = []
        y = []
        for r in summary:
            if r["solver"] == solver:
                if r["size"] in sizes:
                    x.append(r["size"])
                    y.append(r["median"])
        ax.plot(x, y, marker=".", linestyle=LINE_STYLES[i%len(LINE_STYLES)], alpha=0.85, label=solver, zorder=z)
    
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
    legend.set_zorder(20+len(constraints))
    frame = legend.get_frame()
    frame.set_alpha(0.75)
    fig.tight_layout()
    out_path = os.path.join(os.path.abspath("plots"), f"redundant_runtime_v_size.png")
    fig.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")

def run_wilcoxon(results: list, baseline: str, constraints: list|None = None, sizes: list|None = None, alpha: float = 0.05, print_results: bool = True) -> None:
    """ Run the RQ2 analysis

    Args:
        results (list): Results from the experiments
        baseline (str): Baseline solver used in the experiments
        constraints (list | None, optional): Constraints used in the experiments. Defaults to None.
        sizes (list | None, optional): Puzzle sizes to be analysed. Defaults to None.
        alpha (float, optional): Alpha value to use for filtering significance. Defaults to 0.05.
        print_results (bool, optional): Flag to indicate if we want to print the results to the CLI. Defaults to True.
    """
    flattened_results = _flatten_results(results)
    p_matrix = _wilcoxon_by_constraint(flattened_results, baseline, constraints, sizes)
    relevance_stats = _classify_constraints(p_matrix, alpha)

    constraint_order, constraint_labels, size_order, significance_grid = _significance_grid(p_matrix, relevance_stats, baseline)
    if print_results:
        _print_wolcoxon(relevance_stats)
    _plot_significance_heatmap(constraint_labels, size_order, significance_grid)
    speedup = _summarize_speedup_by_constraint(flattened_results, baseline, constraints, sizes)
    encoding_size_stats = _summarize_encoding_ratio_by_constraint(flattened_results, baseline, constraints, sizes)
    constraint_labels2, size_order2, speedup_grid = _speedup_grid(constraint_order, size_order, speedup, relevance_stats, baseline)
    _plot_speedup_heatmap(constraint_labels2, size_order2, speedup_grid)
    _plot_runtime_per_size(results, baseline, constraints, sizes)
    print_constraint_summary(speedup, encoding_size_stats, sizes=size_order, baseline=baseline)
