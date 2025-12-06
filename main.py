import argparse
import time
import os
import z3solver
from datetime import datetime
from file_utils import read_puzzle, read_puzzle_dir, read_solution, read_solution_dir, write_file, append_comment
from analysis import print_outliers
from utils import format_elapsed
from plots import PLOT_TYPES, plot
from checker import check_puzzle

PUZZLES_FOLDER = os.path.abspath("puzzles")
SOLUTIONS_FOLDER = os.path.abspath("solutions")
SOLVERS = {
    "baseline": z3solver.solve_qf_ia,
    "alternate_uniqueness": z3solver.solve_qf_ia_unique_improved,
    "alternate_connectivity": z3solver.solve_qf_ia_connect_improved,
    "tree_connectivity": z3solver.solve_qf_ia_connect_tree,
    "bitvector": z3solver.solve_qf_bv,
    "pattern_1": z3solver.solve_qf_ia_p1,
    "pattern_2": z3solver.solve_qf_ia_p2,
    "pattern_3": z3solver.solve_qf_ia_p3,
    "pattern_4": z3solver.solve_qf_ia_p4,
    "pattern_5": z3solver.solve_qf_ia_p5,
    "pattern_6": z3solver.solve_qf_ia_p6,
    "pattern_7": z3solver.solve_qf_ia_p7,
    "pattern_8": z3solver.solve_qf_ia_p8,
    "white_neighbours": z3solver.solve_r_white_neighbours,
    "atleast_whites": z3solver.solve_r_atleast_whites,
    "atmost_blacks": z3solver.solve_r_atmost_blacks,
    "corner_implications": z3solver.solve_r_corner_implications,
    "pair_implications": z3solver.solve_r_pair_implications,
    "between": z3solver.solve_r_between,
    "force_double_edge": z3solver.solve_r_force_double_edge,
    "close_edge": z3solver.solve_r_close_edge,
    "bridges": z3solver.solve_r_bridges,
    "all": z3solver.solve_all,
    "best": z3solver.solve_best
}

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
            solution, solver_statistics, puzzle_statistics = SOLVERS[solver](puzzle)
            end = time.perf_counter()
            elapsed = end-start
            print(f"Solved {len(puzzle)}x{len(puzzle)} puzzle {fname} using {solver} solver in {format_elapsed(elapsed)}")

            results.append({
                "puzzle": fname,
                "size": n,
                "solver": solver,
                "elapsed": elapsed,
                "statistics": solver_statistics,
                "puzzle_statistics": puzzle_statistics
            })

            if best_elapsed is None or elapsed < best_elapsed:
                best_elapsed = elapsed
                best_solution = solution
                best_statistics = solver_statistics
                best_solver = solver

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
            comment = f"Solved in {format_elapsed(best_elapsed)} using {best_solver}\n{statistics_str}\n"
            write_file(solution_path, best_solution, seed, comment)

    if args.plot is not None:
        if not results:
            print("No results to plot")
        else:
            args_plot = args.plot if isinstance(args.plot, list) else [args.plot]
            for plt in args_plot:
                plot(plt, results)

def _analyze_command(args):
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
    for run in range(args.runs):
        for path, puzzle, _ in puzzles:
            fname = os.path.splitext(os.path.basename(path))[0]
            n = len(puzzle)

            for solver in args.solvers:
                start = time.perf_counter()
                _, solver_statistics, puzzle_statistics = SOLVERS[solver](puzzle)
                end = time.perf_counter()
                elapsed = end-start
                print(f"Run {run+1}: Solved {len(puzzle)}x{len(puzzle)} puzzle {fname} using {solver} solver in {format_elapsed(elapsed)}")

                results.append({
                    "run": run,
                    "puzzle": fname,
                    "size": n,
                    "solver": solver,
                    "elapsed": elapsed,
                    "statistics": solver_statistics,
                    "puzzle_statistics": puzzle_statistics
                })
    
    print_outliers(results, 2.0)

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

    analyze_parser = subparsers.add_parser("analyze", help="Analyze puzzles using the SMT solver")
    group_analyze = analyze_parser.add_mutually_exclusive_group(required=True)
    group_analyze.add_argument("-f", "--file", action="append", type=str, help="Path to a puzzle file")
    group_analyze.add_argument("-d", "--folder", action="append", type=str, help="Path to folder containing puzzle files")
    analyze_parser.add_argument("-r", "--recursive", action="store_true", help="Recursively read subfolders")
    analyze_parser.add_argument("-s", "--strict", action="store_true", help="Exit when wrong file type is found")
    analyze_parser.add_argument("-i", "--runs", default=1, type=int, help="Number of runs to complete")
    analyze_parser.add_argument("solvers", nargs="+", choices=SOLVERS.keys(), help="One or more solver variants to run")
    analyze_parser.set_defaults(func=_analyze_command)

    args = parser.parse_args()
    args.func(args)