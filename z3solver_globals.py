from z3 import * # type: ignore

def least_whites(s: Solver, colored: list, puzzle: list, n: int, encoding_size: dict) -> None:
    """ Makes sure there is at least n/2 white cells in each row and column

    Args:
        s (Solver): Solver to add assertions to
        colored (list): Matrix of BoolRef values for solver to fill
        puzzle (list): Matrix of Integers representing the number grid of the puzzle instance
        n (int): Size of the puzzle
        encoding_size (dict): Variable counts for this encoding
    """
    min_white = n//2
    for i in range(n):
        s.add(AtLeast(*[Not(colored[i][j]) for j in range(n)], min_white))
        s.add(AtLeast(*[Not(colored[j][i]) for j in range(n)], min_white))

def most_blacks(s: Solver, colored: list, puzzle: list, n: int, encoding_size: dict) -> None:
    """ Makes sure there is at most (n/2)+1 colored cells in each row and column

    Args:
        s (Solver): Solver to add assertions to
        colored (list): Matrix of BoolRef values for solver to fill
        puzzle (list): Matrix of Integers representing the number grid of the puzzle instance
        n (int): Size of the puzzle
        encoding_size (dict): Variable counts for this encoding
    """
    max_black = (n//2)+1
    for i in range(n):
        s.add(AtMost(*[colored[i][j] for j in range(n)], max_black))
        s.add(AtMost(*[colored[j][i] for j in range(n)], max_black))

def pair_isolation(s: Solver, colored: list, puzzle: list, n: int, encoding_size: dict) -> None:
    """ If a pair of equal adjacent numbers exist in a row or column, all other occurences of that number must be colored

    Args:
        s (Solver): Solver to add assertions to
        colored (list): Matrix of BoolRef values for solver to fill
        puzzle (list): Matrix of Integers representing the number grid of the puzzle instance
        n (int): Size of the puzzle
        encoding_size (dict): Variable counts for this encoding
    """
    if n < 4:
        return
    
    for i in range(n):
        for j in range(n-1):
            # Rows
            if puzzle[i][j] == puzzle[i][j+1]:
                for k in range(n):
                    if k not in (j-1, j, j+1, j+2) and puzzle[i][j] == puzzle[i][k]:
                        s.add(colored[i][k])
            # Columns
            if puzzle[j][i] == puzzle[j+1][i]:
                for k in range(n):
                    if k not in (j-1, j, j+1, j+2) and puzzle[j][i] == puzzle[k][i]:
                        s.add(colored[k][i])

def close_isolation(s: Solver, colored: list, puzzle: list, n: int, encoding_size: dict) -> None:
    """ If a pair of equal numbers threatens to close of a white cell, all other occurences of that number must be colored

    Args:
        s (Solver): Solver to add assertions to
        colored (list): Matrix of BoolRef values for solver to fill
        puzzle (list): Matrix of Integers representing the number grid of the puzzle instance
        n (int): Size of the puzzle
        encoding_size (dict): Variable counts for this encoding
    """
    
    # For all rows
    for i in range(n):
        for j in range(n-2):
            if puzzle[i][j] == puzzle[i][j+2]:
                conditions = []
                if i > 0:
                    conditions.append(colored[i-1][j+1])
                if i < n-1:
                    conditions.append(colored[i+1][j+1])
                
                if not conditions:
                    continue
                condition = And(*conditions) if len(conditions) > 1 else conditions[0]

                for k in range(n):
                    if k not in (j, j+2) and puzzle[i][k] == puzzle[i][j]:
                        s.add(Implies(condition, colored[i][k]))

    # For all columns
    for j in range(n):
        for i in range(n-2):
            if puzzle[i][j] == puzzle[i+2][j]:
                conditions = []
                if j > 0:
                    conditions.append(colored[i+1][j-1])
                if j < n-1:
                    conditions.append(colored[i+1][j+1])
                
                if not conditions:
                    continue
                condition = And(*conditions) if len(conditions) > 1 else conditions[0]

                for k in range(n):
                    if k not in (i, i+2) and puzzle[k][j] == puzzle[i][j]:
                        s.add(Implies(condition, colored[k][j]))

def white_bridges(s: Solver, colored: list, puzzle: list, n: int, encoding_size: dict) -> None:
    """ All pairs of adjacent rows and columns must have at least two adjacent white cells

    Args:
        s (Solver): Solver to add assertions to
        colored (list): Matrix of BoolRef values for solver to fill
        puzzle (list): Matrix of Integers representing the number grid of the puzzle instance
        n (int): Size of the puzzle
        encoding_size (dict): Variable counts for this encoding
    """
    for i in range(n-1):
        whites_in_row1 = Or(*[Not(colored[i][j]) for j in range(n)])
        whites_in_row2 = Or(*[Not(colored[i+1][j]) for j in range(n)])

        row_bridge = Or(*[And(Not(colored[i][j]), Not(colored[i+1][j])) for j in range(  n)])
        s.add(Implies(And(whites_in_row1, whites_in_row2), row_bridge))

        whites_in_col1 = Or(*[Not(colored[j][i]) for j in range(n)])
        whites_in_col2 = Or(*[Not(colored[j][i+1]) for j in range(n)])

        col_bridge = Or(*[And(Not(colored[j][i]), Not(colored[j][i+1])) for j in range(n)])
        s.add(Implies(And(whites_in_col1, whites_in_col2), col_bridge))