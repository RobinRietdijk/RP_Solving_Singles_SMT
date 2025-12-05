import sys
import z3solver_redundants
import z3solver_patterns
from z3 import *

# Distinct tactic following Z3py basics, SMT/SAT describes a one-hot approach for latin squares
def _add_constraint_uniquecells_pairs(s: Solver, colored: list[list[BoolRef]], grid: list[list[int]], n: int) -> None:
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

# Alternate implemention of the uniquecells constraint by counting the values and asserting at most 1 value per column and row
def _add_constraint_uniquecells_atmost(s: Solver, colored: list[list[BoolRef]], grid: list[list[int]], n: int) -> None:
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

def _add_constraint_connectedwhite_QFIA_ranking(s: Solver, colored: list[list[BoolRef]], n: int) -> None:
    root_row = Int("root_r")
    root_col = Int("root_c")
    s.add(root_row == 0)
    s.add(Or(root_col == 0, root_col == 1))
    s.add(Or(And(root_col == 0, Not(colored[0][0])), And(root_col == 1, Not(colored[0][1]))))

    rank = [[Int(f"num_{r}_{c}") for c in range(n)] for r in range(n)]

    for i in range(n):
        for j in range(n):
            s.add(Implies(colored[i][j], rank[i][j] == -1))
            s.add(Implies(And(Not(colored[i][j]), i == root_row, j == root_col), rank[i][j] == 0))
            s.add(Implies(And(Not(colored[i][j]), Not(And(i == root_row, j == root_col))), rank[i][j] > 0))
            
    for i in range(n):
        for j in range(n):
            conditions = []
            if i > 0:
                conditions.append(And(Not(colored[i-1][j]), rank[i-1][j] < rank[i][j]))
            if j > 0:
                conditions.append(And(Not(colored[i][j-1]), rank[i][j-1] < rank[i][j]))
            if i+1 < n:
                conditions.append(And(Not(colored[i+1][j]), rank[i+1][j] < rank[i][j]))
            if j+1 < n:
                conditions.append(And(Not(colored[i][j+1]), rank[i][j+1] < rank[i][j]))

            if conditions:
                s.add(Implies(
                        And(Not(colored[i][j]), Not(And(root_row == i, root_col == j))),
                        Or(*conditions)
                    ))
                
def _add_constraint_connectedwhite_QFIA_ranking_improved(s: Solver, colored: list[list[BoolRef]], n: int) -> None:
    max_rank = n*n-1
    rank = [[Int(f"rank_{i}_{j}") for j in range(n)] for i in range(n)]
    root = [[Bool(f"root_{i}_{j}") for j in range(n)] for i in range(n)]

    root_flat = [root[i][j] for i in range(n) for j in range(n)]
    s.add(PbEq([(r, 1) for r in root_flat], 1))

    for i in range(n):
        for j in range(n):
            s.add(Implies(colored[i][j], And(rank[i][j] == -1, Not(root[i][j]))))
            s.add(Implies(root[i][j], And(Not(colored[i][j]), rank[i][j] == 0)))
            s.add(Implies(And(Not(colored[i][j]), Not(root[i][j])), And(rank[i][j] > 0, rank[i][j] <= max_rank)))
            
    for i in range(n):
        for j in range(n):
            conditions = []
            if i > 0:
                conditions.append(And(Not(colored[i-1][j]), rank[i-1][j] < rank[i][j]))
            if j > 0:
                conditions.append(And(Not(colored[i][j-1]), rank[i][j-1] < rank[i][j]))
            if i+1 < n:
                conditions.append(And(Not(colored[i+1][j]), rank[i+1][j] < rank[i][j]))
            if j+1 < n:
                conditions.append(And(Not(colored[i][j+1]), rank[i][j+1] < rank[i][j]))

            if conditions:
                s.add(Implies(
                        And(Not(colored[i][j]), Not(root[i][j])),
                        Or(*conditions)
                    ))
                
def _add_constraint_connectedwhite_QFIA_tree(s: Solver, colored: list[list[BoolRef]], n: int) -> None:
    max_rank = n*n-1
    depth = [[Int(f"rank_{i}_{j}") for j in range(n)] for i in range(n)]
    root = [[Bool(f"root_{i}_{j}") for j in range(n)] for i in range(n)]

    parent_up = [[Bool(f"Up_{i}_{j}") for j in range(n)] for i in range(n)]
    parent_down = [[Bool(f"Down_{i}_{j}") for j in range(n)] for i in range(n)]
    parent_left = [[Bool(f"Left_{i}_{j}") for j in range(n)] for i in range(n)]
    parent_right = [[Bool(f"Right_{i}_{j}") for j in range(n)] for i in range(n)]

    root_flat = [root[i][j] for i in range(n) for j in range(n)]
    s.add(PbEq([(r, 1) for r in root_flat], 1))

    for i in range(n):
        for j in range(n):
            parents = [parent_up[i][j], parent_down[i][j], parent_left[i][j], parent_right[i][j]]
            s.add(Implies(colored[i][j], And(depth[i][j] == -1, Not(root[i][j]), *[Not(p) for p in parents])))
            s.add(Implies(root[i][j], And(Not(colored[i][j]), depth[i][j] == 0, *[Not(p) for p in parents])))
            s.add(Implies(And(Not(colored[i][j]), Not(root[i][j])), And(depth[i][j] > 0, depth[i][j] <= max_rank, PbEq([(parents[k], 1) for k in range(4)], 1))))
            
            if i > 0:
                s.add(Implies(parent_up[i][j], And(Not(colored[i][j]), depth[i-1][j] < depth[i][j])))
            else:
                s.add(Not(parent_up[i][j]))
            if j > 0:
                s.add(Implies(parent_left[i][j], And(Not(colored[i][j-1]), depth[i][j-1] < depth[i][j])))
            else:
                s.add(Not(parent_left[i][j]))
            if i+1 < n:
                s.add(Implies(parent_down[i][j], And(Not(colored[i+1][j]), depth[i+1][j] < depth[i][j])))
            else:
                s.add(Not(parent_down[i][j]))
            if j+1 < n:
                s.add(Implies(parent_right[i][j], And(Not(colored[i][j+1]), depth[i][j+1] < depth[i][j])))
            else:
                s.add(Not(parent_right[i][j]))
                
def _add_constraint_connectedwhite_QFBV_ranking(s: Solver, colored: list[list[BoolRef]], n: int) -> None:
    max_num = n*n+1
    k = max_num.bit_length()

    root_row = BitVec("root_r", k)
    root_col = BitVec("root_c", k)
    zero_bv = BitVecVal(0, k)
    n_bv = BitVecVal(n, k)
    max_bv = BitVecVal(max_num, k)
    max_valid = BitVecVal(max_num-1, k)

    is_white = [[Bool(f"white_{i}_{j}") for j in range(n)] for i in range(n)]
    is_root = [[Bool(f"root_{i}_{j}") for j in range(n)] for i in range(n)]

    s.add(ULT(root_row, n_bv))
    s.add(ULT(root_col, n_bv))

    Number = [[BitVec(f"num_{r}_{c}", k) for c in range(n)] for r in range(n)]

    row_bvs = [BitVecVal(i, k) for i in range(n)]
    col_bvs = [BitVecVal(i, k) for i in range(n)]
    for i in range(n):
        for j in range(n):
            i_bv = row_bvs[i]
            j_bv = col_bvs[j]

            s.add(is_white[i][j] == Not(colored[i][j]))
            s.add(is_root[i][j] == And(root_row == i_bv, root_col == j_bv))

            s.add(Implies(Not(is_white[i][j]), Number[i][j] == max_bv))
            s.add(Implies(is_white[i][j], ULE(Number[i][j], max_valid)))

            s.add(Implies(And(is_white[i][j], is_root[i][j]), Number[i][j] == zero_bv))
            s.add(Implies(And(is_white[i][j], Not(is_root[i][j])), UGT(Number[i][j], zero_bv)))
            
    for i in range(n):
        for j in range(n):
            conditions = []
            if i > 0:
                conditions.append(And(is_white[i-1][j], ULT(Number[i-1][j], Number[i][j])))
            if j > 0:
                conditions.append(And(is_white[i][j-1], ULT(Number[i][j-1], Number[i][j])))
            if i+1 < n:
                conditions.append(And(is_white[i+1][j], ULT(Number[i+1][j], Number[i][j])))
            if j+1 < n:
                conditions.append(And(is_white[i][j+1], ULT(Number[i][j+1], Number[i][j])))

            if conditions:
                s.add(Implies(And(is_white[i][j], Not(is_root[i][j])), Or(*conditions)))

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
    _add_constraint_uniquecells_pairs(s, colored, puzzle, n)
    _add_constraint_neighbours(s, colored, n)
    _add_constraint_connectedwhite_QFIA_ranking(s, colored, n)
    return _solve(s, n, puzzle, colored)

def solve_qf_ia_unique_improved(puzzle: list[list[int]]) -> tuple[list[list[str]], dict]:
    n = len(puzzle)
    s = Solver()
    colored = [[Bool(f"B_{i},{j}") for j in range(n)] for i in range(n)]
    _add_constraint_uniquecells_atmost(s, colored, puzzle, n)
    _add_constraint_neighbours(s, colored, n)
    _add_constraint_connectedwhite_QFIA_ranking(s, colored, n)
    return _solve(s, n, puzzle, colored)

def solve_qf_ia_connect_improved(puzzle: list[list[int]]) -> tuple[list[list[str]], dict]:
    n = len(puzzle)
    s = Solver()
    colored = [[Bool(f"B_{i},{j}") for j in range(n)] for i in range(n)]
    _add_constraint_uniquecells_pairs(s, colored, puzzle, n)
    _add_constraint_neighbours(s, colored, n)
    _add_constraint_connectedwhite_QFIA_ranking_improved(s, colored, n)
    return _solve(s, n, puzzle, colored)

def solve_qf_ia_connect_tree(puzzle: list[list[int]]) -> tuple[list[list[str]], dict]:
    n = len(puzzle)
    s = Solver()
    colored = [[Bool(f"B_{i},{j}") for j in range(n)] for i in range(n)]
    _add_constraint_uniquecells_pairs(s, colored, puzzle, n)
    _add_constraint_neighbours(s, colored, n)
    _add_constraint_connectedwhite_QFIA_tree(s, colored, n)
    return _solve(s, n, puzzle, colored)

def solve_qf_bv(puzzle: list[list[int]]) -> tuple[list[list[str]], dict]:
    n = len(puzzle)
    s = Solver()
    colored = [[Bool(f"B_{i},{j}") for j in range(n)] for i in range(n)]
    _add_constraint_uniquecells_pairs(s, colored, puzzle, n)
    _add_constraint_neighbours(s, colored, n)
    _add_constraint_connectedwhite_QFBV_ranking(s, colored, n)
    return _solve(s, n, puzzle, colored)

def solve_qf_ia_p1(puzzle: list[list[int]]) -> tuple[list[list[str]], dict]:
    n = len(puzzle)
    s = Solver()
    colored = [[Bool(f"B_{i},{j}") for j in range(n)] for i in range(n)]
    _add_constraint_uniquecells_pairs(s, colored, puzzle, n)
    _add_constraint_neighbours(s, colored, n)
    _add_constraint_connectedwhite_QFIA_ranking(s, colored, n)
    z3solver_patterns.gh_pattern_1(s, colored, puzzle, n)
    return _solve(s, n, puzzle, colored)


def solve_qf_ia_p2(puzzle: list[list[int]]) -> tuple[list[list[str]], dict]:
    n = len(puzzle)
    s = Solver()
    colored = [[Bool(f"B_{i},{j}") for j in range(n)] for i in range(n)]
    _add_constraint_uniquecells_pairs(s, colored, puzzle, n)
    _add_constraint_neighbours(s, colored, n)
    _add_constraint_connectedwhite_QFIA_ranking(s, colored, n)
    z3solver_patterns.gh_pattern_2(s, colored, puzzle, n)
    return _solve(s, n, puzzle, colored)


def solve_qf_ia_p3(puzzle: list[list[int]]) -> tuple[list[list[str]], dict]:
    n = len(puzzle)
    s = Solver()
    colored = [[Bool(f"B_{i},{j}") for j in range(n)] for i in range(n)]
    _add_constraint_uniquecells_pairs(s, colored, puzzle, n)
    _add_constraint_neighbours(s, colored, n)
    _add_constraint_connectedwhite_QFIA_ranking(s, colored, n)
    z3solver_patterns.gh_pattern_3(s, colored, puzzle, n)
    return _solve(s, n, puzzle, colored)


def solve_qf_ia_p4(puzzle: list[list[int]]) -> tuple[list[list[str]], dict]:
    n = len(puzzle)
    s = Solver()
    colored = [[Bool(f"B_{i},{j}") for j in range(n)] for i in range(n)]
    _add_constraint_uniquecells_pairs(s, colored, puzzle, n)
    _add_constraint_neighbours(s, colored, n)
    _add_constraint_connectedwhite_QFIA_ranking(s, colored, n)
    z3solver_patterns.gh_pattern_4(s, colored, puzzle, n)
    return _solve(s, n, puzzle, colored)


def solve_qf_ia_p5(puzzle: list[list[int]]) -> tuple[list[list[str]], dict]:
    n = len(puzzle)
    s = Solver()
    colored = [[Bool(f"B_{i},{j}") for j in range(n)] for i in range(n)]
    _add_constraint_uniquecells_pairs(s, colored, puzzle, n)
    _add_constraint_neighbours(s, colored, n)
    _add_constraint_connectedwhite_QFIA_ranking(s, colored, n)
    z3solver_patterns.gh_pattern_5(s, colored, puzzle, n)
    return _solve(s, n, puzzle, colored)


def solve_qf_ia_p6(puzzle: list[list[int]]) -> tuple[list[list[str]], dict]:
    n = len(puzzle)
    s = Solver()
    colored = [[Bool(f"B_{i},{j}") for j in range(n)] for i in range(n)]
    _add_constraint_uniquecells_pairs(s, colored, puzzle, n)
    _add_constraint_neighbours(s, colored, n)
    _add_constraint_connectedwhite_QFIA_ranking(s, colored, n)
    z3solver_patterns.gh_pattern_6(s, colored, puzzle, n)
    return _solve(s, n, puzzle, colored)


def solve_qf_ia_p7(puzzle: list[list[int]]) -> tuple[list[list[str]], dict]:
    n = len(puzzle)
    s = Solver()
    colored = [[Bool(f"B_{i},{j}") for j in range(n)] for i in range(n)]
    _add_constraint_uniquecells_pairs(s, colored, puzzle, n)
    _add_constraint_neighbours(s, colored, n)
    _add_constraint_connectedwhite_QFIA_ranking(s, colored, n)
    z3solver_patterns.gh_pattern_7(s, colored, puzzle, n)
    return _solve(s, n, puzzle, colored)


def solve_qf_ia_p8(puzzle: list[list[int]]) -> tuple[list[list[str]], dict]:
    n = len(puzzle)
    s = Solver()
    colored = [[Bool(f"B_{i},{j}") for j in range(n)] for i in range(n)]
    _add_constraint_uniquecells_pairs(s, colored, puzzle, n)
    _add_constraint_neighbours(s, colored, n)
    _add_constraint_connectedwhite_QFIA_ranking(s, colored, n)
    z3solver_patterns.gh_pattern_8(s, colored, puzzle, n)
    return _solve(s, n, puzzle, colored)
