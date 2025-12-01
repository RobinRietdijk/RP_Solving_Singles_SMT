import argparse
import time
import os
import z3solver
from datetime import datetime
from file_utils import read_puzzle, read_puzzle_dir, read_solution, read_solution_dir, write_file, append_comment
from plots import PLOT_TYPES, plot
from checker import check_puzzle

PUZZLES_FOLDER = os.path.abspath("puzzles")
SOLUTIONS_FOLDER = os.path.abspath("solutions")
SOLVERS = {
    "baseline": z3solver.solve_qf_ia,
    "bitvector": z3solver.solve_qf_bv,
    "line_pattern": z3solver.solve_qf_ia_redundant1,
    "corner_pattern_1": z3solver.solve_qf_ia_redundant2,
    "corner_pattern_2": z3solver.solve_qf_ia_redundant3,
    "corner_constraints": z3solver.solve_qf_ia_redundant4,
    "white_neighbours": z3solver.solve_qf_ia_redundant5,
    "unique_values": z3solver.solve_qf_ia_redundant6
}

def _format_elapsed(elapsed: float) -> str:
    if elapsed < 1.0:
        ms = elapsed*1000
        return f"{ms:.3f} ms"
    else:
        return f"{elapsed:.6f} s"

def _check_command(args):
    solutions = []
    if args.file:
        args_file = args.file if isinstance(args.file, list) else [args.file]   
        for file in args_file:
            solutions.append(read_solution(file, args.strict))
    if args.folder:
        args_folder = args.folder if isinstance(args.folder, list) else [args.folder]
        for folder in args_folder:
            solutions.extend(read_solution_dir(folder, args.recursive, args.strict))

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

def _solve_command(args):
    puzzles = []
    if args.file:
        args_file = args.file if isinstance(args.file, list) else [args.file]    
        for file in args_file:
            puzzles.append(read_puzzle(file, args.strict))
    if args.folder:
        args_folder = args.folder if isinstance(args.folder, list) else [args.folder]
        for folder in args_folder:
            puzzles.extend(read_puzzle_dir(folder, args.recursive, args.strict))

    results = []
    for path, puzzle, seed in puzzles:
        fname = os.path.splitext(os.path.basename(path))[0]
        n = len(puzzle)

        best_elapsed = None
        best_solution = None
        best_statistics = None
        best_solver = None

        for solver in args.solvers:
            start = time.perf_counter()
            solution, statistics = SOLVERS[solver](puzzle)
            end = time.perf_counter()
            elapsed = end-start
            print(f"Solved {len(puzzle)}x{len(puzzle)} puzzle {fname} using {solver} solver in {_format_elapsed(elapsed)}")

            results.append({
                "puzzle": fname,
                "size": n,
                "solver": solver,
                "elapsed": elapsed,
                "statistics": statistics
            })

            if best_elapsed is None or elapsed < best_elapsed:
                best_elapsed = elapsed
                best_solution = solution
                best_statistics = statistics
                best_solver = solver

        if len(args.solvers) > 1 and best_solver is not None:
            print(f"Fastest for {fname}: {best_solver}")

        if args.write and best_solver is not None:
            absolute_path = os.path.abspath(path)
            try:
                relative_path = os.path.relpath(absolute_path, PUZZLES_FOLDER)
                in_puzzles = not relative_path.startswith(os.pardir + os.sep) and relative_path != os.pardir
            except ValueError:
                in_puzzles = False
            
            if in_puzzles:
                no_extension, _ = os.path.splitext(relative_path)
                solution_path = os.path.join(SOLUTIONS_FOLDER, no_extension + ".singlessol")
            else:
                solution_fname = fname + ".singlessol"
                solution_path = os.path.join(SOLUTIONS_FOLDER, solution_fname)
            os.makedirs(os.path.dirname(solution_path), exist_ok=True)

            statistics_str = "\n".join(f"{k}: {v}" for k, v in best_statistics.items())
            comment = f"Solved in {_format_elapsed(best_elapsed)} using {best_solver}\n{statistics_str}\n"
            write_file(solution_path, best_solution, seed, comment)

    if args.plot is not None:
        if not results:
            print("No results to plot")
        else:
            args_plot = args.plot if isinstance(args.plot, list) else [args.plot]
            for plt in args_plot:
                plot(plt, results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hitori SMT solver and checker")
    subparsers = parser.add_subparsers(dest="command", required=True)

    solve_parser = subparsers.add_parser("solve", help="Solve puzzle(s) using the SMT solver")
    group_solve = solve_parser.add_mutually_exclusive_group(required=True)
    group_solve.add_argument("-f", "--file", action="append", type=str, help="Path to a puzzle file")
    group_solve.add_argument("-d", "--folder", action="append", type=str, help="Path to folder containing puzzle files")
    solve_parser.add_argument("-r", "--recursive", action="store_true", help="Recursively read subfolders")
    solve_parser.add_argument("-s", "--strict", action="store_true", help="Exit when wrong file type is found")
    solve_parser.add_argument("-w", "--write", action="store_true", help="Write to file")
    solve_parser.add_argument("-p", "--plot", action="append", type=int, choices=PLOT_TYPES.keys(), help="Generate a specific plot type")
    solve_parser.add_argument("solvers", nargs="+", choices=SOLVERS.keys(), help="One or more solver variants to run")
    solve_parser.set_defaults(func=_solve_command)

    check_parser = subparsers.add_parser("check", help="Check solution(s) for correctness")
    group_check = check_parser.add_mutually_exclusive_group(required=True)
    group_check.add_argument("-f", "--file", action="append", type=str, help="Path to a solution file")
    group_check.add_argument("-d", "--folder", action="append", type=str, help="Path to folder containing solution files")
    check_parser.add_argument("-r", "--recursive", action="store_true", help="Recursively read subfolders")
    check_parser.add_argument("-s", "--strict", action="store_true", help="Exit when wrong file type is found")
    check_parser.add_argument("-w", "--write", action="store_true", help="Write to file")
    check_parser.set_defaults(func=_check_command)

    args = parser.parse_args()
    args.func(args)