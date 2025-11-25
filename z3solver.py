import sys
from z3 import *

# Distinct tactic following Z3py basics, SMT/SAT describes a one-hot approach for latin squares
def add_constraint_uniquecells(s: Solver, colored: list, grid: list, n: int) -> None:
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

# Trivial, remove i-1 and j-1, since we already check those in earlier cells
# Van der Knijff uses a slightly different approach where two cells cannot be both white if they have equal numbers
def add_constraint_neighbours(s: Solver, colored: list, n: int) -> None:
    for i in range(n):
        for j in range(n):
            if i+1 < n:
                # Horizontal neighbours
                s.add(Not(And(colored[i][j], colored[i+1][j])))
            if j+1 < n:
                # Vertical neighbours
                s.add(Not(And(colored[i][j], colored[i][j+1])))

def add_constraint_connectedwhite(s: Solver, colored: list, n: int) -> None:
    root_row = Int("root_r")
    root_col = Int("root_c")
    s.add(0 <= root_row, root_row < n, 0 <= root_col, root_col < n)

    Number = [[Int(f"num_{r}_{c}") for c in range(n)] for r in range(n)]

    for i in range(n):
        for j in range(n):
            s.add(Implies(And(root_row == i, root_col == j), Not(colored[i][j])))
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

def solve(puzzle: list) -> None:
    n = len(puzzle)
    s = Solver()
    colored = [[Bool(f"B_{i},{j}") for j in range(n)] for i in range(n)]
    add_constraint_uniquecells(s, colored, puzzle, n)
    add_constraint_neighbours(s, colored, n)
    add_constraint_connectedwhite(s, colored, n)

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
    
    return solution
        