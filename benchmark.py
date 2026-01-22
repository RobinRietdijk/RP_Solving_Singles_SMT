from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
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

# Set the colors for solvers to be used in plots
SOLVER_COLORS = {
    "z3": "#D62728", 
    "asp": "#006BA4", 
    "prolog": "#8C564B", 
    "pumpkin": "#2CA02C", 
    "gurobi": "#7B4EA3",
}

SOLVER_LINE_STYLES = {
    "z3": "-", 
    "asp": "--", 
    "prolog": "-.", 
    "pumpkin": ":", 
    "gurobi": "-",
}

def _read_files(path: str) -> dict:
    benchmarks = {}
    folder = Path(path)
    for file in folder.iterdir():
        if file.suffix == ".csv":
            solver_name = file.stem
            df = pd.read_csv(file)
            benchmarks[solver_name] = df.to_dict(orient="records")
    return benchmarks

def _plot(results: dict, solver_order: list) -> None:
    fig, ax = plt.subplots()

    for i, solver in enumerate(solver_order):
        z = 10+(len(solver_order)-i)
        x = []
        y = []

        by_size = {}
        for r in results[solver]:
            key = r["grid size"]
            if key not in by_size:
                by_size[key] = []
            by_size[key].append(r["time in s"])

        xs = sorted(by_size.keys())
        ys = [float(np.median(by_size[n])) for n in xs]
        ax.plot(xs, ys, marker=".", linestyle=SOLVER_LINE_STYLES[solver], color=SOLVER_COLORS[solver], alpha=0.85, label=solver, zorder=z)

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
    out_path = os.path.join(os.path.abspath("plots"), f"benchmark.png")
    fig.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")

def main():
    results = _read_files("csvs/benchmark/internal")
    _plot(results, ["asp", "z3", "prolog", "pumpkin", "gurobi"])

if __name__ == "__main__":
    main()