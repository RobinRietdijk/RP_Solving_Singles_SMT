import sys
from z3 import *

# Distinct tactic following Z3py basics, SMT/SAT describes a one-hot approach for latin squares
def _add_constraint_uniquecells(s: Solver, colored: list[list[BoolRef]], grid: list[list[int]], n: int) -> None:
    for i in range(n):
        for j in range(n):
            for k in range(j+1, n):
                # If 2 cells have an equal symbol, we must color one black
                if (grid[i][j] == grid[i][k]):
                    s.add(Or(colored[i][j], colored[i][k]))
                    
    for i in range(n):
        for j in range(n):
            for k in range(j+1, n):
                if (grid[j][i] == grid[k][i]):
                    s.add(Or(colored[j][i], colored[k][i]))

def _add_constraint_uniquecells2(s: Solver, colored: list[list[BoolRef]], grid: list[list[int]], n: int) -> None:
    for i in range(n):
        row_values = {}
        col_values = {}

        for j in range(n):
            v_r = grid[i][j]
            v_c = grid[j][i]

            if v_r not in row_values:
                row_values[v_r] = []
            if v_c not in col_values:
                col_values[v_c] = []
            row_values[v_r].append((i, j))
            col_values[v_c].append((j, i))
        
        for _, cells in row_values.items():
            if len(cells) <= 1:
                continue

            whites = [Not(colored[j][k]) for (j, k) in cells]
            s.add(AtMost(*whites, 1))
        
        for _, cells in col_values.items():
            if len(cells) <= 1:
                continue

            whites = [Not(colored[j][k]) for (j, k) in cells]
            s.add(AtMost(*whites, 1))

# Trivial, remove i-1 and j-1, since we already check those in earlier cells
# Van der Knijff uses a slightly different approach where two cells cannot be both white if they have equal numbers
def _add_constraint_neighbours(s: Solver, colored: list[list[BoolRef]], n: int) -> None:
    for i in range(n):
        for j in range(n):
            if i+1 < n:
                # Horizontal neighbours
                s.add(Not(And(colored[i][j], colored[i+1][j])))
            if j+1 < n:
                # Vertical neighbours
                s.add(Not(And(colored[i][j], colored[i][j+1])))

def _add_constraint_connectedwhite_QFIA(s: Solver, colored: list[list[BoolRef]], n: int) -> None:
    root_row = Int("root_r")
    root_col = Int("root_c")
    s.add(root_row == 0)
    s.add(Or(root_col == 0, root_col == 1))
    s.add(Or(And(root_col == 0, Not(colored[0][0])), And(root_col == 1, Not(colored[0][1]))))

    Number = [[Int(f"num_{r}_{c}") for c in range(n)] for r in range(n)]

    for i in range(n):
        for j in range(n):
            s.add(If(Not(colored[i][j]),
                     If(And(i == root_row, j == root_col),
                        Number[i][j] == 0,
                        Number[i][j] > 0),
                    Number[i][j] == -1))
            
    for i in range(n):
        for j in range(n):
            conditions = []
            if i > 0:
                conditions.append(And(Not(colored[i-1][j]), Number[i-1][j] < Number[i][j]))
            if j > 0:
                conditions.append(And(Not(colored[i][j-1]), Number[i][j-1] < Number[i][j]))
            if i+1 < n:
                conditions.append(And(Not(colored[i+1][j]), Number[i+1][j] < Number[i][j]))
            if j+1 < n:
                conditions.append(And(Not(colored[i][j+1]), Number[i][j+1] < Number[i][j]))

            if conditions:
                s.add(Implies(
                        And(Not(colored[i][j]), Not(And(root_row == i, root_col == j))),
                        Or(*conditions)
                    ))
                
def _add_constraint_connectedwhite_QFBV(s: Solver, colored: list[list[BoolRef]], n: int) -> None:
    max_num = n*n+1
    k = max_num.bit_length()

    root_row = BitVec("root_r", k)
    root_col = BitVec("root_c", k)
    zero_bv = BitVecVal(0, k)
    n_bv = BitVecVal(n, k)
    s.add(ULE(zero_bv, root_row), ULT(root_row, n_bv))
    s.add(ULE(zero_bv, root_col), ULT(root_col, n_bv))

    Number = [[BitVec(f"num_{r}_{c}", k) for c in range(n)] for r in range(n)]

    for i in range(n):
        for j in range(n):
            i_bv = BitVecVal(i, k)
            j_bv = BitVecVal(j, k)

            s.add(Or(Number[i][j] == BitVecVal(n*n+1, k), ULE(Number[i][j], BitVecVal(n*n, k))))
            s.add(Implies(And(root_row == i_bv, root_col == j_bv), Not(colored[i][j])))
            s.add(If(Not(colored[i][j]),
                     If(And(i_bv == root_row, j_bv == root_col),
                        Number[i][j] == zero_bv,
                        UGT(Number[i][j], zero_bv)),
                    Number[i][j] == BitVecVal(n*n+1, k)))
            
    for i in range(n):
        for j in range(n):
            i_bv = BitVecVal(i, k)
            j_bv = BitVecVal(j, k)

            conditions = []
            if i > 0:
                conditions.append(And(Not(colored[i-1][j]), ULT(Number[i-1][j], Number[i][j])))
            if j > 0:
                conditions.append(And(Not(colored[i][j-1]), ULT(Number[i][j-1], Number[i][j])))
            if i+1 < n:
                conditions.append(And(Not(colored[i+1][j]), ULT(Number[i+1][j], Number[i][j])))
            if j+1 < n:
                conditions.append(And(Not(colored[i][j+1]), ULT(Number[i][j+1], Number[i][j])))

            if conditions:
                s.add(Implies(
                        And(Not(colored[i][j]), Not(And(root_row == i_bv, root_col == j_bv))),
                        Or(*conditions)
                    ))

# Redundant constraint to make sure evert white cell has atleast one white neighbour
def _add_r_constraint_whiteneighbours(s: Solver, colored: list[list[BoolRef]], n: int) -> None:
    for i in range(n):
        for j in range(n):
            neighbours = []
            if i > 0:
                neighbours.append(Not(colored[i-1][j]))
            if j > 0:
                neighbours.append(Not(colored[i][j-1]))
            if i+1 < n:
                neighbours.append(Not(colored[i+1][j]))
            if j+1 < n:
                neighbours.append(Not(colored[i][j+1]))
            
            if neighbours:
                s.add(Implies(Not(colored[i][j]), Or(*neighbours)))
def _add_r_constraint_leastwhites(s: Solver, colored: list[list[BoolRef]], n: int) -> None:
    for i in range(n):
        rows = []
        cols = []
        for j in range(n):
            rows.append(Not(colored[i][j]))
            cols.append(Not(colored[j][i]))
        s.add(Or(*rows))
        s.add(Or(*cols))
def _add_r_constraint_corners(s: Solver, colored: list[list[BoolRef]], n: int) -> None:
    s.add(Implies(colored[0][1], Not(colored[1][0])))
    s.add(Implies(colored[1][0], Not(colored[0][1])))
    s.add(Implies(colored[0][n-2], Not(colored[1][n-1])))
    s.add(Implies(colored[1][n-1], Not(colored[0][n-2])))
    s.add(Implies(colored[n-2][0], Not(colored[n-1][1])))
    s.add(Implies(colored[n-1][1], Not(colored[n-2][0])))
    s.add(Implies(colored[n-1][n-2], Not(colored[n-2][n-1])))
    s.add(Implies(colored[n-2][n-1], Not(colored[n-1][n-2])))
def _add_r_constraint_pattern1(s: Solver, colored: list[list[BoolRef]], grid: list[list[int]], n: int) -> None:
    for i in range(n):
        for j in range(n):
            if i < n-2:
                if grid[i][j] == grid[i+1][j] and grid[i][j] == grid[i+2][j]:
                    v = grid[i][j]
                    s.add(colored[i][j])
                    s.add(colored[i+2][j])
                    
                    for k in range(n):
                        if k not in (i, i+1, i+2) and grid[k][j] == v:
                            s.add(colored[k][j])
            if j < n-2:
                if grid[i][j] == grid[i][j+1] and grid[i][j] == grid[i][j+2]:
                    v = grid[i][j]
                    s.add(colored[i][j])
                    s.add(colored[i][j+2])

                    for k in range(n):
                        if k not in (j, j+1, j+2) and grid[i][k] == v:
                            s.add(colored[i][k])

def _add_r_constraint_pattern2(s: Solver, colored: list[list[BoolRef]], grid: list[list[int]], n: int) -> None:
    if (grid[0][0] == grid[0][1] and grid[0][0] == grid[1][0]):
        s.add(colored[0][0])
    if (grid[0][n-1] == grid[0][n-2] and grid[0][n-1] == grid[1][n-1]):
        s.add(colored[0][n-1])
    if (grid[n-1][0] == grid[n-2][0] and grid[n-1][0] == grid[n-1][1]):
        s.add(colored[n-1][0])
    if (grid[n-1][n-1] == grid[n-1][n-2] and grid[n-1][n-1] == grid[n-2][n-1]):
        s.add(colored[n-1][n-1])
def _add_r_constraint_pattern3(s: Solver, colored: list[list[BoolRef]], grid: list[list[int]], n: int) -> None:
    if (grid[0][0] == grid[0][1] and grid[0][0] == grid[1][0] and grid[0][0] == grid[1][1]):
        s.add(colored[0][0])
        s.add(colored[1][1])
    if (grid[0][n-1] == grid[0][n-2] and grid[0][n-1] == grid[1][n-1] and grid[0][n-1] == grid[1][n-2]):
        s.add(colored[0][n-1])
        s.add(colored[1][n-2])
    if (grid[n-1][0] == grid[n-2][0] and grid[n-1][0] == grid[n-1][1] and grid[n-1][0] == grid[n-2][1]):
        s.add(colored[n-1][0])
        s.add(colored[n-2][1])
    if (grid[n-1][n-1] == grid[n-1][n-2] and grid[n-1][n-1] == grid[n-2][n-1] and grid[n-1][n-1] == grid[n-2][n-2]):
        s.add(colored[n-1][n-1])
        s.add(colored[n-2][n-2])

def _solve(s: Solver, n: int, puzzle: list[list[int]], colored: list[list[BoolRef]]) -> None:
    if s.check() != sat:
        sys.exit(f"Error: Could not find a satisfiable answer to the puzzle")

    m = s.model()
    sat_model = [[m.evaluate(colored[r][c]) == True for c in range(n)] for r in range(n)]
    solution = []
    for i in range(n):
        row = []
        for j in range(n):
            cell = str(puzzle[i][j])
            if sat_model[i][j]:
                cell = f"{cell}B"
            row.append(cell)
        solution.append(row)

    st = s.statistics()
    statistics = {
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
    }
    return solution, statistics

def solve_qf_ia(puzzle: list[list[int]]) -> tuple[list[list[str]], dict]:
    n = len(puzzle)
    s = Solver()
    colored = [[Bool(f"B_{i},{j}") for j in range(n)] for i in range(n)]
    _add_constraint_uniquecells(s, colored, puzzle, n)
    _add_constraint_neighbours(s, colored, n)
    _add_constraint_connectedwhite_QFIA(s, colored, n)
    return _solve(s, n, puzzle, colored)

def solve_qf_bv(puzzle: list[list[int]]) -> tuple[list[list[str]], dict]:
    n = len(puzzle)
    s = Solver()
    colored = [[Bool(f"B_{i},{j}") for j in range(n)] for i in range(n)]
    _add_constraint_uniquecells(s, colored, puzzle, n)
    _add_constraint_neighbours(s, colored, n)
    _add_constraint_connectedwhite_QFBV(s, colored, n)
    return _solve(s, n, puzzle, colored)

def solve_qf_ia_redundant_unique_values(puzzle: list[list[int]]) -> tuple[list[list[str]], dict]:
    n = len(puzzle)
    s = Solver()
    colored = [[Bool(f"B_{i},{j}") for j in range(n)] for i in range(n)]
    _add_constraint_uniquecells2(s, colored, puzzle, n)
    _add_constraint_neighbours(s, colored, n)
    _add_constraint_connectedwhite_QFIA(s, colored, n)
    return _solve(s, n, puzzle, colored)

def solve_qf_ia_redundant_pattern1(puzzle: list[list[int]]) -> tuple[list[list[str]], dict]:
    n = len(puzzle)
    s = Solver()
    colored = [[Bool(f"B_{i},{j}") for j in range(n)] for i in range(n)]
    _add_constraint_uniquecells(s, colored, puzzle, n)
    _add_constraint_neighbours(s, colored, n)
    _add_constraint_connectedwhite_QFIA(s, colored, n)
    _add_r_constraint_pattern1(s, colored, puzzle, n)
    return _solve(s, n, puzzle, colored)

def solve_qf_ia_redundant_pattern2(puzzle: list[list[int]]) -> tuple[list[list[str]], dict]:
    n = len(puzzle)
    s = Solver()
    colored = [[Bool(f"B_{i},{j}") for j in range(n)] for i in range(n)]
    _add_constraint_uniquecells(s, colored, puzzle, n)
    _add_constraint_neighbours(s, colored, n)
    _add_constraint_connectedwhite_QFIA(s, colored, n)
    _add_r_constraint_pattern2(s, colored, puzzle, n)
    return _solve(s, n, puzzle, colored)

def solve_qf_ia_redundant_pattern3(puzzle: list[list[int]]) -> tuple[list[list[str]], dict]:
    n = len(puzzle)
    s = Solver()
    colored = [[Bool(f"B_{i},{j}") for j in range(n)] for i in range(n)]
    _add_constraint_uniquecells(s, colored, puzzle, n)
    _add_constraint_neighbours(s, colored, n)
    _add_constraint_connectedwhite_QFIA(s, colored, n)
    _add_r_constraint_pattern3(s, colored, puzzle, n)
    return _solve(s, n, puzzle, colored)

def solve_qf_ia_redundant_corners(puzzle: list[list[int]]) -> tuple[list[list[str]], dict]:
    n = len(puzzle)
    s = Solver()
    colored = [[Bool(f"B_{i},{j}") for j in range(n)] for i in range(n)]
    _add_constraint_uniquecells(s, colored, puzzle, n)
    _add_constraint_neighbours(s, colored, n)
    _add_constraint_connectedwhite_QFIA(s, colored, n)
    _add_r_constraint_corners(s, colored, n)
    return _solve(s, n, puzzle, colored)

def solve_qf_ia_redundant_atleast1white(puzzle: list[list[int]]) -> tuple[list[list[str]], dict]:
    n = len(puzzle)
    s = Solver()
    colored = [[Bool(f"B_{i},{j}") for j in range(n)] for i in range(n)]
    _add_constraint_uniquecells(s, colored, puzzle, n)
    _add_constraint_neighbours(s, colored, n)
    _add_constraint_connectedwhite_QFIA(s, colored, n)
    _add_r_constraint_leastwhites(s, colored, n)
    return _solve(s, n, puzzle, colored)

def solve_qf_ia_redundant_whiteneighbour(puzzle: list[list[int]]) -> tuple[list[list[str]], dict]:
    n = len(puzzle)
    s = Solver()
    colored = [[Bool(f"B_{i},{j}") for j in range(n)] for i in range(n)]
    _add_constraint_uniquecells(s, colored, puzzle, n)
    _add_constraint_neighbours(s, colored, n)
    _add_constraint_connectedwhite_QFIA(s, colored, n)
    _add_r_constraint_whiteneighbours(s, colored, n)
    return _solve(s, n, puzzle, colored)