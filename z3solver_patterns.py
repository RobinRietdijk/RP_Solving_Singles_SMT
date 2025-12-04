from z3 import *

# Implementation of pattern 1 according to chapter 3.3 of Hitori Solver
def gh_pattern_1(s: Solver, colored: list[list[BoolRef]], grid: list[list[int]], n: int) -> None:
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

# Implementation of pattern 2 according to chapter 3.3 of Hitori Solver
def gh_pattern_2(s: Solver, colored: list[list[BoolRef]], grid: list[list[int]], n: int) -> None:
    if (grid[0][0] == grid[0][1] and grid[0][0] == grid[1][0]):
        s.add(colored[0][0])
    if (grid[0][n-1] == grid[0][n-2] and grid[0][n-1] == grid[1][n-1]):
        s.add(colored[0][n-1])
    if (grid[n-1][0] == grid[n-2][0] and grid[n-1][0] == grid[n-1][1]):
        s.add(colored[n-1][0])
    if (grid[n-1][n-1] == grid[n-1][n-2] and grid[n-1][n-1] == grid[n-2][n-1]):
        s.add(colored[n-1][n-1])

# Implementation of pattern 3 according to chapter 3.3 of Hitori Solver
def gh_pattern_3(s: Solver, colored: list[list[BoolRef]], grid: list[list[int]], n: int) -> None:
    if (grid[0][0] == grid[0][1] and grid[0][0] == grid[1][0] and grid[0][0] == grid[1][1]):
        s.add(colored[0][0])
        s.add(colored[1][1])
        if n >= 3:
            s.add(Not(colored[0][2]))
            s.add(Not(colored[2][0]))
    if (grid[0][n-1] == grid[0][n-2] and grid[0][n-1] == grid[1][n-1] and grid[0][n-1] == grid[1][n-2]):
        s.add(colored[0][n-1])
        s.add(colored[1][n-2])
        if n >= 3:
            s.add(Not(colored[0][n-3]))
            s.add(Not(colored[2][n-1]))
    if (grid[n-1][0] == grid[n-2][0] and grid[n-1][0] == grid[n-1][1] and grid[n-1][0] == grid[n-2][1]):
        s.add(colored[n-1][0])
        s.add(colored[n-2][1])
        if n >= 3:
            s.add(Not(colored[n-3][0]))
            s.add(Not(colored[n-1][2]))
    if (grid[n-1][n-1] == grid[n-1][n-2] and grid[n-1][n-1] == grid[n-2][n-1] and grid[n-1][n-1] == grid[n-2][n-2]):
        s.add(colored[n-1][n-1])
        s.add(colored[n-2][n-2])
        if n >= 3:
            s.add(Not(colored[n-3][n-1]))
            s.add(Not(colored[n-1][n-3]))

# Implementation of pattern 4 according to chapter 3.3 of Hitori Solver
def gh_pattern_4(s: Solver, colored: list[list[BoolRef]], grid: list[list[int]], n: int) -> None:
    if n < 4:
        return
    
    for i in range(1, n-2):
        if grid[0][i] == grid[0][i+1] and grid[1][i] == grid[1][i+1] and grid[2][i] == grid[2][i+1]:
            s.add(Not(colored[0][i-1]))
            s.add(Not(colored[1][i-1]))
            s.add(Not(colored[0][i+2]))
            s.add(Not(colored[1][i+2]))
        if grid[n-1][i] == grid[n-1][i+1] and grid[n-2][i] == grid[n-2][i+1] and grid[n-3][i] == grid[n-3][i+1]:
            s.add(Not(colored[n-1][i-1]))
            s.add(Not(colored[n-2][i-1]))
            s.add(Not(colored[n-1][i+2]))
            s.add(Not(colored[n-2][i+2]))
        if grid[i][0] == grid[i+1][0] and grid[i][1] == grid[i+1][1] and grid[i][2] == grid[i+1][2]:
            s.add(Not(colored[i-1][0]))
            s.add(Not(colored[i-1][1]))
            s.add(Not(colored[i+2][0]))
            s.add(Not(colored[i+2][1]))
        if grid[i][n-1] == grid[i+1][n-1] and grid[i][n-2] == grid[i+1][n-2] and grid[i][n-3] == grid[i+1][n-3]:
            s.add(Not(colored[i-1][n-1]))
            s.add(Not(colored[i-1][n-2]))
            s.add(Not(colored[i+2][n-1]))
            s.add(Not(colored[i+2][n-2]))

# Implementation of pattern 5 according to chapter 3.3 of Hitori Solver
def gh_pattern_5(s: Solver, colored: list[list[BoolRef]], grid: list[list[int]], n: int) -> None:
    if n < 4:
        return
    
    for i in range(1, n-2):
        if grid[0][i] == grid[0][i+1] and grid[1][i] == grid[1][i+1]:
            s.add(Not(colored[0][i-1]))
            s.add(Not(colored[0][i+2]))
        if grid[n-1][i] == grid[n-1][i+1] and grid[n-2][i] == grid[n-2][i+1]:
            s.add(Not(colored[n-1][i-1]))
            s.add(Not(colored[n-1][i+2]))
        if grid[i][0] == grid[i+1][0] and grid[i][1] == grid[i+1][1]:
            s.add(Not(colored[i-1][0]))
            s.add(Not(colored[i+2][0]))
        if grid[i][n-1] == grid[i+1][n-1] and grid[i][n-2] == grid[i+1][n-2]:
            s.add(Not(colored[i-1][n-1]))
            s.add(Not(colored[i+2][n-1]))

# Implementation of pattern 6 according to chapter 3.3 of Hitori Solver
def gh_pattern_6(s: Solver, colored: list[list[BoolRef]], grid: list[list[int]], n: int) -> None:
    for i in range(n):
        for j in range(n-1):
            if grid[i][j] == grid[i][j+1]:
                for k in range(n):
                    if k not in (j, j+1) and grid[i][k] == grid[i][j]:
                        s.add(colored[i][k])

            
            if grid[j][i] == grid[j+1][i]:
                for k in range(n):
                    if k not in (j, j+1) and grid[k][i] == grid[j][i]:
                        s.add(colored[k][i])

# Implementation of pattern 7 according to chapter 3.3 of Hitori Solver
def gh_pattern_7(s: Solver, colored: list[list[BoolRef]], grid: list[list[int]], n: int) -> None:
    for i in range(n):
        for j in range(n-2):
            if grid[i][j] == grid[i][j+2]:
                conditions = []
                if i > 0:
                    conditions.append(colored[i-1][j+1])
                if i < n-1:
                    conditions.append(colored[i+1][j+1])
                
                if not conditions:
                    continue
                condition = And(*conditions) if len(conditions) > 1 else conditions[0]

                for k in range(n):
                    if k not in (j, j+2) and grid[i][k] == grid[i][j]:
                        s.add(Implies(condition, colored[i][k]))

    for j in range(n):
        for i in range(n-2):
            if grid[i][j] == grid[i+2][j]:
                conditions = []
                if j > 0:
                    conditions.append(colored[i+1][j-1])
                if j < n-1:
                    conditions.append(colored[i+1][j+1])
                
                if not conditions:
                    continue
                condition = And(*conditions) if len(conditions) > 1 else conditions[0]

                for k in range(n):
                    if k not in (i, i+2) and grid[k][j] == grid[i][j]:
                        s.add(Implies(condition, colored[k][j]))

# Implementation of pattern 8 according to chapter 3.3 of Hitori Solver
def gh_pattern_8(s: Solver, colored: list[list[BoolRef]], grid: list[list[int]], n: int) -> None:
    for i in range(n):
        if grid[0][i] == grid[1][i]:
            if i > 1:
                if grid[0][i] == grid[1][i-1]:
                    s.add(Implies(colored[0][i-2], colored[1][i]))
            if i < n-2:
                if grid[0][i] == grid[1][i+1]:
                    s.add(Implies(colored[0][i+2], colored[1][i]))
         
        if grid[n-1][i] == grid[n-2][i]:
            if i > 1:
                if grid[n-1][i] == grid[n-2][i-1]:
                    s.add(Implies(colored[n-1][i-2], colored[n-2][i]))
            if i < n-2:
                if grid[n-1][i] == grid[n-2][i+1]:
                    s.add(Implies(colored[n-1][i+2], colored[n-2][i]))
        
        if grid[i][0] == grid[i][1]:
            if i > 1:
                if grid[i][0] == grid[i-1][1]:
                    s.add(Implies(colored[i-2][0], colored[i][1]))
            if i < n-2:
                if grid[i][0] == grid[i+1][1]:
                    s.add(Implies(colored[i+2][0], colored[i][1]))
        
        if grid[i][n-1] == grid[i][n-2]:
            if i > 1:
                if grid[i][n-1] == grid[i-1][n-2]:
                    s.add(Implies(colored[i-2][n-1], colored[i][n-2]))
            if i < n-2:
                if grid[i][n-1] == grid[i+1][n-2]:
                    s.add(Implies(colored[i+2][n-1], colored[i][n-2]))