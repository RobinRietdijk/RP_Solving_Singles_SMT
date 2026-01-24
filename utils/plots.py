import matplotlib.pyplot as plt
import scienceplots
from scipy.stats import probplot

plt.style.use(['science','ieee'])

def _log_differences(results: list, statistic, solver1: str, solver2: str, size: int) -> list:
    """ Calculate the difference per puzzle for a given statistic between 2 solvers

    Args:
        results (list): Results from the evaluation
        statistic (_type_): Statistic to be compared
        solver1 (str): First solver to be used in the comparison
        solver2 (str): Secon solver to be used in the comparison
        size (int): Puzzle size to be compared on

    Returns:
        list: A list of differences between the two solvers on the given statistic
    """
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

def plot_qq_differences(results: list, statistic, solver1: str, solver2: str, size: int) -> None:
    """ Plot a QQ graph of the differences between two solvers on a given statistic

    Args:
        results (list): Results from the evaluation
        statistic (_type_): Statistic to be plotted
        solver1 (str): First solver to be used in the comparison
        solver2 (str): Second solver to be used in the comparison
        size (int): Puzzle size to be plotted
    """
    differences = _log_differences(results, statistic, solver1, solver2, size)

    probplot(differences, plot=plt)
    plt.title("Q–Q plot of paired log-runtime differences")
    plt.show()

def plot_qq_runtime(results: list, solver: str, size: int) -> None:
    """ Plot a QQ graph for runtime of a given solver on a given size

    Args:
        results (list): Results from the evaluation
        solver (str): Solver to be plotted
        size (int): Puzzle size to be plotted
    """
    runtimes = []
    for r in results:
        if r["size"] == size:
            if r["solver"] == solver:
                runtimes.append(r["statistics"]["runtime"])
    
    probplot(runtimes, plot=plt)
    plt.title("Q-Q plot of runtimes for a single solver on a single size")
    plt.show()