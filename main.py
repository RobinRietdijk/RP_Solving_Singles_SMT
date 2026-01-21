import argparse
import time
import os
import z3solver
import hashlib
import shutil
import z3solver_locals
import z3solver_globals
import rq1
import rq2
import rq3
import plots
from datetime import datetime
from file_utils import read_puzzle, read_puzzle_dir, read_solution, read_solution_dir, write_file, append_comment, write_csv,read_csv, read_csv_folder
from utils import format_elapsed
from checker import check_puzzle

# Default path to the puzzle folder
PUZZLES_FOLDER = os.path.abspath("puzzles")
# Default path to the solutions folder
SOLUTIONS_FOLDER = os.path.abspath("solutions")
# Default path to the csv folder
CSV_FOLDER = os.path.abspath("csvs")
# Options for the analysis command
ANALYSIS_OPTIONS = ["write_csv", "rq1", "rq2", "rq3"]
# Multiplier for solver that triggered a timeout
PAR_MULTIPLIER = 2

# Available base solvers that can be called using the CLI
SOLVERS = {
    "qf_ia": z3solver.qf_ia,
    "qf_ia_alt_u": z3solver.qf_ia_alt_u,
    "qf_ia_alt_c": z3solver.qf_ia_alt_c,
    "qf_ia_tree_c": z3solver.qf_ia_tree_c,
    "qf_bv": z3solver.qf_bv,
    "qf_bool": z3solver.boolean,
    "qf_ia-c": z3solver.lazy,
    "lazy": z3solver.lazy,
    "qf_ia_external": z3solver.lazy
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
    """ Gather statistics from a solver run

    Args:
        run (int): Run that we are in
        path (str): Path of the puzzle
        puzzle (list): Puzzle to be ran
        seed (int): Seed for the solver
        solver (dict): Solver to be used

    Returns:
        dict: Dictionary of statistics from the solver run
    """
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
    elif args.csv_dir:
        if isinstance(args.csv_dir, list):
            paths = args.csv_dir
        else:
            paths = [args.dsv_dir]
        for path in paths:
            results.extend(read_csv_folder(path, args.strict, args.recursive))
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

    if args.analysis == "rq1":
        rq1.plot_encoding_scaling(results, [solver["name"] for solver in args.solvers])
        rq1.plot_runtime_vs_size(results, [solver["name"] for solver in args.solvers])
        rq1.print_rq1_text_stats(results, report_sizes=[5, 6, 7, 8, 9, 13, 21, 25])
        rq1.print_encoding_text_stats(results, report_sizes=[10, 25])
    if args.analysis == "rq2":
        rq2.run_wilcoxon(results, "qf_ia", constraints=["qf_ia+sp", "qf_ia+pi", "qf_ia+wb", "qf_ia+wn", "qf_ia+lw", "qf_ia+ce"], sizes=[10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25])
    if args.analysis == "rq3":
        rq3.run_all(results, k_out=1.5, k_fout=3.0)
    if args.analysis =="write_csv":
        write_csv(results, CSV_FOLDER)


def _parse_solver_specs(solver: str) -> dict:
    """ Parses the solver argument

    Args:
        solver (str): Argument for solver to be used

    Raises:
        argparse.ArgumentTypeError: _description_
        argparse.ArgumentTypeError: _description_
        argparse.ArgumentTypeError: _description_

    Returns:
        dict: _description_
    """
    parts = [i for i in solver.split("+") if i]
    if not parts:
        raise argparse.ArgumentTypeError("Empty solver specification")
    
    base = parts[0]
    constraints = parts[1:]

    if base not in SOLVERS:
        raise argparse.ArgumentTypeError(f"Unknown base solver: {base}")
    
    unknown = [i for i in constraints if i not in CONSTRAINTS]
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown constraints: {', '.join(unknown)}")
    
    return {"base": SOLVERS[base], "constraints": [CONSTRAINTS[i] for i in constraints], "name": solver}

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
    group_analyze.add_argument("-ed", "--csv_dir", action="append", type=str, help="Path to csv folder")
    analyze_parser.add_argument("-r", "--recursive", action="store_true", help="Recursively read subfolders")
    analyze_parser.add_argument("-s", "--strict", action="store_true", help="Exit when wrong file type is found")
    analyze_parser.add_argument("-i", "--runs", default=1, type=int, help="Number of runs to complete")
    analyze_parser.add_argument("-th", "--hard_threshold", default=3.0, type=float, help="Threshold for hard difficulty score")
    analyze_parser.add_argument("-te", "--easy_threshold", default=3.0, type=float, help="Threshold for easy difficulty score")
    analyze_parser.add_argument("-p", "--print", action="store_true", help="Print difficult puzzles to terminal")
    analyze_parser.add_argument("-c", "--copy", type=str, help="Copy difficult puzzles to new relative folder")
    analyze_parser.add_argument("solvers", nargs="+", type=_parse_solver_specs, help="Solver(s) used to run analysis")
    analyze_parser.set_defaults(func=_analyze_command)

    args = parser.parse_args()
    args.func(args)