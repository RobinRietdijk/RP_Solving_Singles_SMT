import sys
import time
import z3solver_base
from z3 import * # type: ignore

TIMEOUT = 10000


def _find_white_components(white_cells: list, n: int) -> list:
    visited = [[False]*n for _ in range(n)]
    components = []

    for i in range(n):
        for j in range(n):
            if white_cells[i][j] and not visited[i][j]:
                stack = [(i, j)]
                component = []
                visited[i][j] = True
                while stack:
                    x, y = stack.pop()
                    component.append((x, y))
                    for (nx, ny) in [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]:
                        if 0 <= nx < n and 0 <= ny < n:
                            if white_cells[nx][ny] and not visited[nx][ny]:
                                visited[nx][ny] = True
                                stack.append((nx, ny))
                components.append(component)
    return components


def _add_constraint_connectivity_cut(s: Solver, component: list, colored: list, n: int) -> None:
    boundary = set()
    for (i, j) in component:
        for (ni, nj) in [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]:
            if 0 <= ni < n and 0 <= nj < n and (ni, nj) not in component:
                boundary.add((ni, nj))
    
    s.add(Or(Or(colored[i][j] for (i, j) in component), Or(Not(colored[i][j]) for (i, j) in boundary)))


def _solve(s: Solver, colored: list, puzzle: list, n: int, encoding_size: dict) -> tuple[bool, list|None, dict|None, dict|None]:
    result = s.check()
    if result == unsat:
        sys.exit(f"Error: Could not find a satisfiable answer to the puzzle")
    elif result == unknown:
        solution = None
        puzzle_statistics = None
    else:
        m = s.model()
        sat_model = [[z3.is_true(m.evaluate(colored[r][c])) for c in range(n)] for r in range(n)]
        black_cells = 0
        solution = []
        for i in range(n):
            row = []
            for j in range(n):
                cell = str(puzzle[i][j])
                if sat_model[i][j]:
                    black_cells += 1
                    cell = f"{cell}B"
                row.append(cell)
            solution.append(row)
        
        puzzle_statistics = {
            "black_cells": black_cells
        }
    st = s.statistics()
    encoding_size["assertions"] = len(s.assertions())
    solver_statistics = {
        "restarts": st.get_key_value("restarts") if "restarts" in st.keys() else 0,
        "propagations": st.get_key_value("propagations") if "propagations" in st.keys() else 0,
        "rlimit_count": st.get_key_value("rlimit count") if "rlimit count" in st.keys() else 0,
        "bool_vars": st.get_key_value("mk bool var") if "mk bool var" in st.keys() else 0,
        "clauses": st.get_key_value("mk clause") if "mk clause" in st.keys() else 0,
        "bin_clauses": st.get_key_value("mk clause binary") if "mk clause binary" in st.keys() else 0,
        "conflicts": st.get_key_value("conflicts") if "conflicts" in st.keys() else 0,
        "decisions": st.get_key_value("decisions") if "decisions" in st.keys() else 0,
        "memory": st.get_key_value("memory") if "memory" in st.keys() else 0,
        "max_memory": st.get_key_value("max memory") if "max memory" in st.keys() else 0,
        "time": st.get_key_value("time") if "time" in st.keys() else 0,
        "encoding_size": encoding_size
    }
    timed_out = solution is None
    return timed_out, solution, solver_statistics, puzzle_statistics


def _solve_lazy(s: Solver, colored: list, puzzle: list, n: int, encoding_size: dict) -> tuple[bool, list|None, dict|None, dict|None]:
    start = time.perf_counter()
    while True:
        if start + time.perf_counter() >= TIMEOUT:
            st = s.statistics()
            encoding_size["assertions"] = len(s.assertions())
            solver_statistics = {
                "propagations": st.get_key_value("propagations") if "propagations" in st.keys() else 0,
                "rlimit": st.get_key_value("rlimit count") if "rlimit count" in st.keys() else 0,
                "bool_vars": st.get_key_value("mk bool var") if "mk bool var" in st.keys() else 0,
                "clauses": st.get_key_value("mk clause") if "mk clause" in st.keys() else 0,
                "bin_clauses": st.get_key_value("mk clause binary") if "mk clause binary" in st.keys() else 0,
                "conflicts": st.get_key_value("conflicts") if "conflicts" in st.keys() else 0,
                "decisions": st.get_key_value("decisions") if "decisions" in st.keys() else 0,
                "memory": st.get_key_value("memory") if "memory" in st.keys() else 0,
                "max_memory": st.get_key_value("max memory") if "max memory" in st.keys() else 0,
                "time": st.get_key_value("time") if "time" in st.keys() else 0,
                "encoding_size": encoding_size
            }
            
            return False, None, solver_statistics, None
        
        result = s.check()
        if result != sat:
            sys.exit(f"Error: Could not find a satisfiable answer to the puzzle")

        m = s.model()
        sat_model = [[m.evaluate(colored[r][c]) == False for c in range(n)] for r in range(n)]

        components = _find_white_components(sat_model, n)
        if len(components) <= 1:
            break
        
        component = components[1]
        _add_constraint_connectivity_cut(s, component, colored, n)
    return _solve(s, colored, puzzle, n, encoding_size)


def _init_solver(n: int, seed: int|None) -> tuple[Solver, list, dict]:
    s = Solver()
    encoding_size = { "int_vars": 0, "bool_vars": 0, "bv_vars": 0 }
    s.set("timeout", TIMEOUT)
    if seed:
        s.set("random_seed", seed)
        
    colored = [[Bool(f"B_{i},{j}") for j in range(n)] for i in range(n)]
    encoding_size["bool_vars"] += n*n
    return s, colored, encoding_size


def solve(base: Callable, constraints: list, puzzle: list, seed: int|None = None) -> tuple[bool, list|None, dict|None, dict|None]:
    n = len(puzzle)
    s, colored, encoding_size = _init_solver(n, seed)
    base(s, colored, puzzle, n, encoding_size)
    for constraint in constraints:
        constraint(s, colored, puzzle, n, encoding_size)
    
    if base == lazy:
        _solve_lazy(s, colored, puzzle, n, encoding_size)
    return _solve(s, colored, puzzle, n, encoding_size)


def qf_ia(s: Solver, colored: list, puzzle: list, n: int, encoding_size: dict) -> None:
    z3solver_base.uniqueness_pairs(s, colored, puzzle, n, encoding_size)
    z3solver_base.neighbours(s, colored, puzzle, n, encoding_size)
    z3solver_base.connectivity_ranking(s, colored, puzzle, n, encoding_size)


def qf_ia_alt_u(s: Solver, colored: list, puzzle: list, n: int, encoding_size: dict) -> None:
    z3solver_base.uniqueness_atmost(s, colored, puzzle, n, encoding_size)
    z3solver_base.neighbours(s, colored, puzzle, n, encoding_size)
    z3solver_base.connectivity_ranking(s, colored, puzzle, n, encoding_size)


def qf_ia_alt_c(s: Solver, colored: list, puzzle: list, n: int, encoding_size: dict) -> None:
    z3solver_base.uniqueness_pairs(s, colored, puzzle, n, encoding_size)
    z3solver_base.neighbours(s, colored, puzzle, n, encoding_size)
    z3solver_base.connectivity_ranking_alt(s, colored, puzzle, n, encoding_size)


def qf_ia_tree_c(s: Solver, colored: list, puzzle: list, n: int, encoding_size: dict) -> None:
    z3solver_base.uniqueness_pairs(s, colored, puzzle, n, encoding_size)
    z3solver_base.neighbours(s, colored, puzzle, n, encoding_size)
    z3solver_base.connectivity_tree(s, colored, puzzle, n, encoding_size)


def qf_bv(s: Solver, colored: list, puzzle: list, n: int, encoding_size: dict) -> None:
    z3solver_base.uniqueness_pairs(s, colored, puzzle, n, encoding_size)
    z3solver_base.neighbours(s, colored, puzzle, n, encoding_size)
    z3solver_base.connectivity_bitvector(s, colored, puzzle, n, encoding_size)


def boolean(s: Solver, colored: list, puzzle: list, n: int, encoding_size: dict) -> None:
    z3solver_base.uniqueness_pairs(s, colored, puzzle, n, encoding_size)
    z3solver_base.neighbours(s, colored, puzzle, n, encoding_size)
    z3solver_base.connectivity_boolean(s, colored, puzzle, n, encoding_size)


def lazy(s: Solver, colored: list, puzzle: list, n: int, encoding_size: dict) -> None:
    z3solver_base.uniqueness_pairs(s, colored, puzzle, n, encoding_size)
    z3solver_base.neighbours(s, colored, puzzle, n, encoding_size)