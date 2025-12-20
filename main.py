import argparse
import time
import os
import z3solver
import hashlib
import shutil
import z3solver_locals
import z3solver_globals
import rq1
import plots
from datetime import datetime
from file_utils import read_puzzle, read_puzzle_dir, read_solution, read_solution_dir, write_file, append_comment, append_dict, write_csv,read_csv
from analysis import find_difficult, print_outlying_puzzles, analyze_puzzle_statistics, analyze_sets
from utils import format_elapsed
from checker import check_puzzle

# Default path to the puzzle folder
PUZZLES_FOLDER = os.path.abspath("puzzles")
# Default path to the solutions folder
SOLUTIONS_FOLDER = os.path.abspath("solutions")
# Default path to the csv folder
CSV_FOLDER = os.path.abspath("csvs")
# Options for the analysis command
ANALYSIS_OPTIONS = ["write_csv", "difficulty", "rq1", "rq2"]
# Multiplier for solver that triggered a timeout
PAR_MULTIPLIER = 2

# Available base solvers that can be called using the CLI
SOLVERS = {
    "qf_ia": z3solver.qf_ia,
    "qf_ia_alt_u": z3solver.qf_ia_alt_u,
    "qf_ia_alt_c": z3solver.qf_ia_alt_c,
    "qf_ia_tree_c": z3solver.qf_ia_tree_c,
    "qf_bv": z3solver.qf_bv,
    "boolean": z3solver.boolean,
    "lazy": z3solver.lazy,
}

# Available constraints that can be added using the CLI
CONSTRAINTS = {
    "wn": z3solver_locals.white_neighbours,
    "cc": z3solver_locals.corner_close,
    "st": z3solver_locals.sandwich_triple,
    "sp": z3solver_locals.sandwich_pair,
    "tc": z3solver_locals.triple_corner,
    "qc": z3solver_locals.quad_corner,
    "tep": z3solver_locals.triple_edge_pair,
    "dep": z3solver_locals.double_edge_pair,
    "ce": z3solver_locals.close_edge,
    "fde": z3solver_locals.force_double_edge,
    "bc": z3solver_locals.border_close,
    "lw": z3solver_globals.least_whites,
    "mb": z3solver_globals.most_blacks,
    "pi": z3solver_globals.pair_isolation,
    "ci": z3solver_globals.close_isolation,
    "wb": z3solver_globals.white_bridges
}


def _read_files(file: str|list, folder: str|list, recursive: bool, strict: bool, read_puzzles: bool) -> list:
    """ Read file(s) or folder(s) for puzzle(s) or solution(s)

    Args:
        file (str | list): A path or list of paths to the file(s) containing the puzzle(s) or solution(s)
        folder (str | list): A path or list of paths to the folder(s) containing the puzzle(s) or solution(s)
        recursive (bool): A flag to enable looking in subfolders for additional files
        strict (bool): A flag to enable strict file checking which throws an error on wrong files
        read_puzzles (bool): A flag to indicate looking for puzzles of solutions

    Returns:
        list: A list of puzzles of solutions found in the file(s) or folder(s)
    """
    results = []
    if file:
        # file can be either a String of list based on the amount of arguments given
        args_file = file if isinstance(file, list) else [file]   
        for file in args_file:
            if read_puzzles:
                results.append(read_puzzle(file, strict))
            else:
                results.append(read_solution(file, strict))
    if folder:
        # folder can be either a String of list based on the amount of arguments given
        args_folder = folder if isinstance(folder, list) else [folder]
        for folder in args_folder:
            if read_puzzles:
                results.extend(read_puzzle_dir(folder, recursive, strict))
            else:
                results.extend(read_solution_dir(folder, recursive, strict))
    return results


def _run_solver(solver: dict, puzzle: list, seed: int|None = None) -> tuple[list|None, dict, dict|None]:
    """ Helper function to run the solver on a puzzle

    Args:
        solver (dict): Solver and constraints to be used
        puzzle (list): Puzzle to be solved by the solver
        seed (int | None): Seed used in the solver

    Returns:
        tuple[list, dict, dict, float]: A collection of the solution to the puzzle, 
            the statistics from the solver, the puzzle statistics and the runtime
    """
    start = time.perf_counter()
    timed_out, solution, solver_statistics, puzzle_statistics = z3solver.solve(solver["base"], solver["constraints"], puzzle, seed)
    end = time.perf_counter()

    if timed_out:
        # Add a penalty to the runtime if the solver timed out
        elapsed = (z3solver.TIMEOUT/1000)*PAR_MULTIPLIER
    else:
        elapsed = end-start
    solver_statistics["runtime"] = elapsed
    return solution, solver_statistics, puzzle_statistics


def _check_command(args: dict) -> None:
    """ Command for checking the validity of solution files. Invoked through the CLI

    Args:
        args (dict): CLI arguments given for this command
    """
    if ["file"] not in args:
        return
    solutions = _read_files(args.file, args.folder, args.recursive, args.strict, False)

    for path, solution, _ in solutions:
        fname = os.path.splitext(os.path.basename(path))[0]
    
        correct = check_puzzle(solution)
        print(f"Solution {fname} is {'correct' if correct else 'wrong'}")
        
        if args.write:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if correct:
                append_comment(path, f"Solution checked to be CORRECT on {timestamp}")
            else:
                append_comment(path, f"Solution checked to be INCORRECT on {timestamp}")


def _solve_command(args: dict) -> None:
    """ Command for solving puzzles. Invoked through the CLI

    Args:
        args (dict): CLI arguments given for this command
    """
    puzzles = _read_files(args.file, args.folder, args.recursive, args.strict, True)

    results = []
    for path, puzzle, seed in puzzles:
        fname = os.path.splitext(os.path.basename(path))[0]
        n = len(puzzle)

        best_elapsed = None
        best_solution = None
        best_statistics = None
        best_solver = None

        for solver in args.solvers:
            solution, solver_statistics, puzzle_statistics = _run_solver(solver, puzzle)
            elapsed = solver_statistics["runtime"]
            print(f"{'Solved' if elapsed < z3solver.TIMEOUT*PAR_MULTIPLIER else 'Timed out'} {fname} ({n}x{n}): time= {format_elapsed(elapsed)}")
            results.append({
                "puzzle": fname,
                "size": n,
                "solver": solver["name"],
                "statistics": solver_statistics,
                "puzzle_statistics": puzzle_statistics
            })

            if not elapsed >= z3solver.TIMEOUT*PAR_MULTIPLIER and best_elapsed is None or elapsed < best_elapsed:
                best_elapsed = elapsed
                best_solution = solution
                best_statistics = solver_statistics
                best_solver = solver["name"]

        # Write the solution with statistics to a solution file
        if args.write and best_solver is not None:
            absolute_path = os.path.abspath(path)

            # Check if the puzzle is located in the default puzzles folder
            try:
                relative_path = os.path.relpath(absolute_path, PUZZLES_FOLDER)
                in_puzzles = not relative_path.startswith(os.pardir+os.sep) and relative_path != os.pardir
            except ValueError:
                in_puzzles = False
            
            if in_puzzles:
                # Follow the structure in the puzzles folder
                no_extension, _ = os.path.splitext(relative_path)
                solution_path = os.path.join(SOLUTIONS_FOLDER, no_extension + ".singlessol")
            else:
                # Add puzzle to the base of the solutions folder
                solution_fname = fname + ".singlessol"
                solution_path = os.path.join(SOLUTIONS_FOLDER, solution_fname)
            os.makedirs(os.path.dirname(solution_path), exist_ok=True)

            statistics_str = "\n".join(f"{k}: {v}" for k, v in best_statistics.items())
            comment = f"Solved in {format_elapsed(best_elapsed)} using {best_solver}\n{statistics_str}\n"
            write_file(solution_path, best_solution, seed, comment)

    # Plot statistics
    if args.plot is not None:
        if not results:
            print("No results to plot")
        else:
            args_plot = args.plot if isinstance(args.plot, list) else [args.plot]
            for plt in args_plot:
                plots.plot_stat(plt, results)


def _gather_satistics(run: int, path: str, puzzle: list, seed: int, solver: dict) -> dict:
    fname = os.path.splitext(os.path.basename(path))[0]
    n = len(puzzle)

    _, solver_statistics, puzzle_statistics = _run_solver(solver, puzzle, seed)
    print(f"[Run: {run+1}] {'Solved' if solver_statistics["runtime"] < z3solver.TIMEOUT*PAR_MULTIPLIER else 'Timed out'} {fname} ({n}x{n}): " \
            f"time= {format_elapsed(solver_statistics["runtime"])}")
    
    return {"run": run,
        "puzzle": fname,
        "path": path,
        "size": n,
        "solver": solver["name"],
        "seed": seed,
        "statistics": solver_statistics,
        "puzzle_statistics": puzzle_statistics
    }


def _analyze_difficulty(args: dict, results: list) -> None:
    """ Command to analyze the puzzle difficulty for the solver.

    Args:
        args (dict): CLI arguments given for this command
        results (list): Results from the solver runs
    """
    hard, easy = find_difficult(results, args.hard_threshold, args.easy_threshold, args.print)

    # Filter hard and easy puzzles to a seperate folder
    if args.copy:
        os.makedirs(args.copy, exist_ok=True)
        
        # Inner function to copy the list of easy and hard puzzles
        def copy_list(stats, destination):
            os.makedirs(destination, exist_ok=True)
            for stat in stats:
                try:
                    path = stat["path"]
                    if not os.path.isfile(path):
                        continue
                    
                    file_destination = os.path.join(destination, os.path.basename(path))
                    if os.path.exists(file_destination):
                        continue
                    shutil.copy2(path, file_destination)
                    
                    # Add the statistics for evaluating difficulty to the puzzle file
                    comment = f"# Solving time: {stat['time']} \n" \
                        f"hard_score: {stat['hard_score']:.2f} \n" \
                        f"easy_score: {stat['easy_score']:.2f} \n" \
                        f"conflicts: {int(stat['conflicts'])} \n" \
                        f"decisions: {int(stat['decisions'])} \n" \
                        f"propagations: {int(stat['propagations'])} \n" \
                        f"rlimit: {int(stat['rlimit'])} \n" \
                        f"max_memory: {stat['max_memory']:.2f} \n" \
                        f"zT: {stat['z']['elapsed']:.2f} \n" \
                        f"zC: {stat['z']['conflicts']:.2f} \n" \
                        f"zD: {stat['z']['decisions']:.2f} \n" \
                        f"zP: {stat['z']['propagations']:.2f} \n" \
                        f"zR: {stat['z']['rlimit_count']:.2f} \n" \
                        f"zM: {stat['z']['max_memory']:.2f} \n" \
                        f"conflicts/decisions: {stat['ratios']['cpd']:.3g} \n" \
                        f"propagations/decisions: {stat['ratios']['ppd']:.3g} \n" \
                        f"rlimit_count/decisions: {stat['ratios']['rpd']:.3g} \n" \
                        f"time/rlimit_count: {stat['ratios']['tpr']:.3g} \n" \
                        f"z_C/D: {stat['z_ratios']['z_rcd']:.2f} \n" \
                        f"z_P/D: {stat['z_ratios']['z_rpd']:.2f} \n" \
                        f"z_R/D: {stat['z_ratios']['z_rrd']:.2f} \n" \
                        f"z_T/R: {stat['z_ratios']['z_rtr']:.2f} \n"
                    append_comment(file_destination, comment)
                except:
                    continue
        
        solvers = set(hard) | set(easy)
        for solver in solvers:
            solver_dir = os.path.join(args.copy, solver)
            copy_list(hard.get(solver, []), os.path.join(solver_dir, "hard"))
            copy_list(easy.get(solver, []), os.path.join(solver_dir, "easy"))


def _analyze_command(args: dict) -> None:
    """ Command to analyze certain properties of the solver. Invoked through the CLI

    Args:
        args (dict): CLI arguments given for this command
    """
    if not args.csv:
        puzzles = _read_files(args.file, args.folder, args.recursive, args.strict, True)
    
    results = []
    if args.csv:
        if isinstance(args.csv, list):
            paths = args.csv
        else:
            paths = [args.csv]
        for path in paths:
            results.extend(read_csv(path))
    else:
        for run in range(args.runs):
            # Generate a reproducable seed to be used for all solvers in this run
            seed = int(hashlib.sha256("|".join(map(str, [run, args.runs, len(puzzles), len(args.solvers)])).encode()).hexdigest()[:8], 16)
            for path, puzzle, _ in puzzles:
                for solver in args.solvers:
                    # Generate a sub seed based on the general seed
                    sub_seed = int(hashlib.sha256("|".join(map(str, [seed, puzzle, solver])).encode()).hexdigest()[:8], 16)
                    result = _gather_satistics(run, path, puzzle, sub_seed, solver)
                    results.append(result)
    
    if args.analysis == "difficulty":
        _analyze_difficulty(args, results)
    if args.analysis == "rq1":
        rq1.plot_encoding_scaling(results, [solver["name"] for solver in args.solvers])
        rq1.plot_runtime_vs_size(results, [solver["name"] for solver in args.solvers])
        rq1.print_rq1_text_stats(results, report_sizes=[15, 25])
        rq1.print_encoding_text_stats(results, report_sizes=[10, 25])
    if args.analysis == "rq2":
        return
    if args.analysis =="write_csv":
        write_csv(results, CSV_FOLDER)


def _analyze_puzzle_properties_command(args: dict) -> None:
    """ Command for analyzing puzzle properties. Invoked through the CLI

    Args:
        args (dict): CLI arguments given for this command
    """
    puzzles = _read_files(args.file, args.folder, args.recursive, args.strict, True)
    puzzle_statistics = analyze_puzzle_statistics(puzzles)
    if args.z_value:
        print_outlying_puzzles(args.z_value, puzzle_statistics)
    
    if args.write:
        for size in puzzle_statistics.values():
            size_stats = {}
            for prop, stats in size["summary"].items():
                for stat, value in stats.items():
                        size_stats[f"{stat}_{prop}"] = value
            
            for stats in size["puzzles"].values():
                path = stats.pop("path", "")
                stats.update(size_stats)
                append_dict(path, stats)


def _compare_sets_command(args: dict) -> None:
    """ Compare two sets against eachother using a single solver. Invoked through the CLI

    Args:
        args (dict): CLI arguments given for this command
    """
    puzzles_set_a = _read_files(args.file1, args.folder1, args.recursive, args.strict, True)
    puzzles_set_b = _read_files(args.file2, args.folder2, args.recursive, args.strict, True)
    largest_set = max(len(puzzles_set_a), len(puzzles_set_b))

    results_set_a = []
    results_set_b = []
    for run in range(args.runs):
        # Generate a reproducable seed to be used for all solvers in this run
        seed = int(hashlib.sha256("|".join(map(str, [run, args.runs, len(puzzles_set_a), len(args.solver)])).encode()).hexdigest()[:8], 16)
        for i in range(largest_set):
            if i < len(puzzles_set_a):
                path, puzzle, _ = puzzles_set_a[i]
                results_set_a.append(_gather_satistics(run, path, puzzle, seed, args.solver))
            if i < len(puzzles_set_b):
                path, puzzle, _ = puzzles_set_b[i]
                results_set_b.append(_gather_satistics(run, path, puzzle, seed, args.solver))
    
    analyze_sets(results_set_a, results_set_b)

def _parse_solver_specs(s: str) -> dict:
    parts = [i for i in s.split("+") if i]
    if not parts:
        raise argparse.ArgumentTypeError("Empty solver specification")
    
    base = parts[0]
    constraints = parts[1:]

    if base not in SOLVERS:
        raise argparse.ArgumentTypeError(f"Unknown base solver: {base}")
    
    unknown = [i for i in constraints if i not in CONSTRAINTS]
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown constraints: {', '.join(unknown)}")
    
    return {"base": SOLVERS[base], "constraints": [CONSTRAINTS[i] for i in constraints], "name": s}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hitori SMT solver and checker")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Command for solving puzzles
    solve_parser = subparsers.add_parser("solve", help="Solve puzzle(s) using the SMT solver")
    group_solve = solve_parser.add_mutually_exclusive_group(required=True)
    group_solve.add_argument("-f", "--file", action="append", type=str, help="Path to a puzzle file")
    group_solve.add_argument("-d", "--folder", action="append", type=str, help="Path to folder containing puzzle files")
    solve_parser.add_argument("-r", "--recursive", action="store_true", help="Recursively read subfolders")
    solve_parser.add_argument("-s", "--strict", action="store_true", help="Exit when wrong file type is found")
    solve_parser.add_argument("-w", "--write", action="store_true", help="Write to file")
    solve_parser.add_argument("-p", "--plot", action="append", type=int, choices=plots.PLOT_TYPES.keys(), help="Generate a specific plot type")
    solve_parser.add_argument("solvers", nargs="+", type=_parse_solver_specs, help="One or more solver variants to run")
    solve_parser.set_defaults(func=_solve_command)

    # Command for checking puzzles
    check_parser = subparsers.add_parser("check", help="Check solution(s) for correctness")
    group_check = check_parser.add_mutually_exclusive_group(required=True)
    group_check.add_argument("-f", "--file", action="append", type=str, help="Path to a solution file")
    group_check.add_argument("-d", "--folder", action="append", type=str, help="Path to folder containing solution files")
    check_parser.add_argument("-r", "--recursive", action="store_true", help="Recursively read subfolders")
    check_parser.add_argument("-s", "--strict", action="store_true", help="Exit when wrong file type is found")
    check_parser.add_argument("-w", "--write", action="store_true", help="Write to file")
    check_parser.set_defaults(func=_check_command)

    # Command for analyzing difficulty
    analyze_parser = subparsers.add_parser("analyze", help="Analyze puzzle difficulty using the SMT solver")
    analyze_parser.add_argument("analysis", choices=ANALYSIS_OPTIONS, help="Analysis mode")
    group_analyze = analyze_parser.add_mutually_exclusive_group(required=True)
    group_analyze.add_argument("-f", "--file", action="append", type=str, help="Path to a puzzle file")
    group_analyze.add_argument("-d", "--folder", action="append", type=str, help="Path to folder containing puzzle files")
    group_analyze.add_argument("-e", "--csv", action="append", type=str, help="Path to csv file")
    analyze_parser.add_argument("-r", "--recursive", action="store_true", help="Recursively read subfolders")
    analyze_parser.add_argument("-s", "--strict", action="store_true", help="Exit when wrong file type is found")
    analyze_parser.add_argument("-i", "--runs", default=1, type=int, help="Number of runs to complete")
    analyze_parser.add_argument("-th", "--hard_threshold", default=3.0, type=float, help="Threshold for hard difficulty score")
    analyze_parser.add_argument("-te", "--easy_threshold", default=3.0, type=float, help="Threshold for easy difficulty score")
    analyze_parser.add_argument("-p", "--print", action="store_true", help="Print difficult puzzles to terminal")
    analyze_parser.add_argument("-c", "--copy", type=str, help="Copy difficult puzzles to new relative folder")
    analyze_parser.add_argument("solvers", nargs="+", type=_parse_solver_specs, help="Solver(s) used to run analysis")
    analyze_parser.set_defaults(func=_analyze_command)

    # Command for analyzing puzzle properties
    analyze_puzzle_properties_parser = subparsers.add_parser("analyze_puzzle_properties", help="Analyze puzzles for patterns and properties")
    group_analyze_puzzle_properties = analyze_puzzle_properties_parser.add_mutually_exclusive_group(required=True)
    group_analyze_puzzle_properties.add_argument("-f", "--file", action="append", type=str, help="Path to a puzzle file")
    group_analyze_puzzle_properties.add_argument("-d", "--folder", action="append", type=str, help="Path to folder containing puzzle files")
    analyze_puzzle_properties_parser.add_argument("-r", "--recursive", action="store_true", help="Recursively read subfolders")
    analyze_puzzle_properties_parser.add_argument("-s", "--strict", action="store_true", help="Exit when wrong file type is found")
    analyze_puzzle_properties_parser.add_argument("-w", "--write", action="store_true", help="Write to file")
    analyze_puzzle_properties_parser.add_argument("-z", "--z_value", type=float, help="Print outliers with z value")
    analyze_puzzle_properties_parser.set_defaults(func=_analyze_puzzle_properties_command)


    # Command for comparing two sets against eachother
    analyze_sets_parser = subparsers.add_parser("analyze_sets", help="Analyze sets of puzzles against eachother")
    group1_analyze_sets = analyze_sets_parser.add_mutually_exclusive_group(required=True)
    group2_analyze_sets = analyze_sets_parser.add_mutually_exclusive_group(required=True)
    group1_analyze_sets.add_argument("-f1", "--file1", action="append", type=str, help="Path to a first puzzle file")
    group1_analyze_sets.add_argument("-d1", "--folder1", action="append", type=str, help="Path to first folder containing puzzle files")
    group2_analyze_sets.add_argument("-f2", "--file2", action="append", type=str, help="Path to a second puzzle file")
    group2_analyze_sets.add_argument("-d2", "--folder2", action="append", type=str, help="Path to second folder containing puzzle files")
    analyze_sets_parser.add_argument("-r", "--recursive", action="store_true", help="Recursively read subfolders")
    analyze_sets_parser.add_argument("-s", "--strict", action="store_true", help="Exit when wrong file type is found")
    analyze_sets_parser.add_argument("-i", "--runs", default=1, type=int, help="Number of runs to complete")
    analyze_sets_parser.add_argument("solver", type=_parse_solver_specs, help="Solver used to run comparison")
    analyze_sets_parser.set_defaults(func=_compare_sets_command)

    args = parser.parse_args()
    args.func(args)