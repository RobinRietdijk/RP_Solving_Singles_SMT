from z3 import * # type: ignore

def white_neighbours(s: Solver, colored: list, puzzle: list, n: int, encoding_size: dict) -> None:
    """ All white cells have atleast one white neighbour

    Args:
        s (Solver): Solver to add assertions to
        colored (list): Matrix of BoolRef values for solver to fill
        puzzle (list): Matrix of Integers representing the number grid of the puzzle instance
        n (int): Size of the puzzle
        encoding_size (dict): Variable counts for this encoding
    """
    for i in range(n):
        for j in range(n):
            neighbours = []
            # Up
            if i > 0:
                neighbours.append(Not(colored[i-1][j]))
            # Left
            if j > 0:
                neighbours.append(Not(colored[i][j-1]))
            # Down
            if i+1 < n: 
                neighbours.append(Not(colored[i+1][j]))
            # Right
            if j+1 < n:
                neighbours.append(Not(colored[i][j+1]))
            
            if neighbours:
                s.add(Implies(Not(colored[i][j]), Or(*neighbours)))

def corner_close(s: Solver, colored: list, puzzle: list, n: int, encoding_size: dict) -> None:
    """ Constraint for a corner pattern to prevent isolating the cornermost cell

    Args:
        s (Solver): Solver to add assertions to
        colored (list): Matrix of BoolRef values for solver to fill
        puzzle (list): Matrix of Integers representing the number grid of the puzzle instance
        n (int): Size of the puzzle
        encoding_size (dict): Variable counts for this encoding
    """
    s.add(Implies(colored[0][1], Not(colored[1][0])))
    s.add(Implies(colored[1][0], Not(colored[0][1])))
    s.add(Implies(colored[0][n-2], Not(colored[1][n-1])))
    s.add(Implies(colored[1][n-1], Not(colored[0][n-2])))
    s.add(Implies(colored[n-2][0], Not(colored[n-1][1])))
    s.add(Implies(colored[n-1][1], Not(colored[n-2][0])))
    s.add(Implies(colored[n-1][n-2], Not(colored[n-2][n-1])))
    s.add(Implies(colored[n-2][n-1], Not(colored[n-1][n-2])))

def sandwich_triple(s: Solver, colored: list, puzzle: list, n: int, encoding_size: dict) -> None:
    """ If three adjacent cells have an equal number, the middle cell is white and all other occurences of the number are colored

    Args:
        s (Solver): Solver to add assertions to
        colored (list): Matrix of BoolRef values for solver to fill
        puzzle (list): Matrix of Integers representing the number grid of the puzzle instance
        n (int): Size of the puzzle
        encoding_size (dict): Variable counts for this encoding
    """
    for i in range(n):
        for j in range(n):
            # Rows
            if i < n-2:
                if puzzle[i][j] == puzzle[i+1][j] and puzzle[i][j] == puzzle[i+2][j]:
                    s.add(colored[i][j])
                    s.add(colored[i+2][j])

            # Columns
            if j < n-2:
                if puzzle[i][j] == puzzle[i][j+1] and puzzle[i][j] == puzzle[i][j+2]:
                    s.add(colored[i][j])
                    s.add(colored[i][j+2])

def sandwich_pair(s: Solver, colored: list, puzzle: list, n: int, encoding_size: dict) -> None:
    """ If a single cell is between two equal numbers, this cell is always white

    Args:
        s (Solver): Solver to add assertions to
        colored (list): Matrix of BoolRef values for solver to fill
        puzzle (list): Matrix of Integers representing the number grid of the puzzle instance
        n (int): Size of the puzzle
        encoding_size (dict): Variable counts for this encoding
    """
    if n < 3:
        return 
    
    for i in range(n):
        for j in range(n-2):
            # Rows
            if puzzle[i][j] == puzzle[i][j+2]:
                s.add(Not(colored[i][j+1]))
            # Columns
            if puzzle[j][i] == puzzle[j+2][i]:
                s.add(Not(colored[j+1][i]))

def triple_corner(s: Solver, colored: list, puzzle: list, n: int, encoding_size: dict) -> None:
    """ Assumptions that can be made if the three corner cells are equal to protect the corner-most cell from being isolated

    Args:
        s (Solver): Solver to add assertions to
        colored (list): Matrix of BoolRef values for solver to fill
        puzzle (list): Matrix of Integers representing the number grid of the puzzle instance
        n (int): Size of the puzzle
        encoding_size (dict): Variable counts for this encoding
    """
    # Top-left corner
    if (puzzle[0][0] == puzzle[0][1] and puzzle[0][0] == puzzle[1][0]):
        s.add(colored[0][0])
    # Top-right corner
    if (puzzle[0][n-1] == puzzle[0][n-2] and puzzle[0][n-1] == puzzle[1][n-1]):
        s.add(colored[0][n-1])
    # Bottom-left corner
    if (puzzle[n-1][0] == puzzle[n-2][0] and puzzle[n-1][0] == puzzle[n-1][1]):
        s.add(colored[n-1][0])
    # Bottom-right corner
    if (puzzle[n-1][n-1] == puzzle[n-1][n-2] and puzzle[n-1][n-1] == puzzle[n-2][n-1]):
        s.add(colored[n-1][n-1])

def quad_corner(s: Solver, colored: list, puzzle: list, n: int, encoding_size: dict) -> None:
    """ Assumptions that can be made if the corner 2x2 square is equal to protect the corner-most cell from being isolated

    Args:
        s (Solver): Solver to add assertions to
        colored (list): Matrix of BoolRef values for solver to fill
        puzzle (list): Matrix of Integers representing the number grid of the puzzle instance
        n (int): Size of the puzzle
        encoding_size (dict): Variable counts for this encoding
    """
    # Top-left corner
    if (puzzle[0][0] == puzzle[0][1] and puzzle[0][0] == puzzle[1][0] and puzzle[0][0] == puzzle[1][1]):
        s.add(colored[0][0])
        s.add(colored[1][1])
        if n >= 3:
            s.add(Not(colored[0][2]))
            s.add(Not(colored[2][0]))
    # Top-right corner
    if (puzzle[0][n-1] == puzzle[0][n-2] and puzzle[0][n-1] == puzzle[1][n-1] and puzzle[0][n-1] == puzzle[1][n-2]):
        s.add(colored[0][n-1])
        s.add(colored[1][n-2])
        if n >= 3:
            s.add(Not(colored[0][n-3]))
            s.add(Not(colored[2][n-1]))
    # Bottom-left corner
    if (puzzle[n-1][0] == puzzle[n-2][0] and puzzle[n-1][0] == puzzle[n-1][1] and puzzle[n-1][0] == puzzle[n-2][1]):
        s.add(colored[n-1][0])
        s.add(colored[n-2][1])
        if n >= 3:
            s.add(Not(colored[n-3][0]))
            s.add(Not(colored[n-1][2]))
    # Bottom-right corner
    if (puzzle[n-1][n-1] == puzzle[n-1][n-2] and puzzle[n-1][n-1] == puzzle[n-2][n-1] and puzzle[n-1][n-1] == puzzle[n-2][n-2]):
        s.add(colored[n-1][n-1])
        s.add(colored[n-2][n-2])
        if n >= 3:
            s.add(Not(colored[n-3][n-1]))
            s.add(Not(colored[n-1][n-3]))

def triple_edge_pair(s: Solver, colored: list, puzzle: list, n: int, encoding_size: dict) -> None:
    """ Pattern that can be found along borders where assumptions can be made to prevent isolated cells along the border

    Args:
        s (Solver): Solver to add assertions to
        colored (list): Matrix of BoolRef values for solver to fill
        puzzle (list): Matrix of Integers representing the number grid of the puzzle instance
        n (int): Size of the puzzle
        encoding_size (dict): Variable counts for this encoding
    """
    if n < 4:
        return
    
    for i in range(1, n-2):
        # Top row
        if puzzle[0][i] == puzzle[0][i+1] and puzzle[1][i] == puzzle[1][i+1] and puzzle[2][i] == puzzle[2][i+1]:
            s.add(Not(colored[0][i-1]))
            s.add(Not(colored[1][i-1]))
            s.add(Not(colored[0][i+2]))
            s.add(Not(colored[1][i+2]))
        # Bottom row
        if puzzle[n-1][i] == puzzle[n-1][i+1] and puzzle[n-2][i] == puzzle[n-2][i+1] and puzzle[n-3][i] == puzzle[n-3][i+1]:
            s.add(Not(colored[n-1][i-1]))
            s.add(Not(colored[n-2][i-1]))
            s.add(Not(colored[n-1][i+2]))
            s.add(Not(colored[n-2][i+2]))
        # Left column
        if puzzle[i][0] == puzzle[i+1][0] and puzzle[i][1] == puzzle[i+1][1] and puzzle[i][2] == puzzle[i+1][2]:
            s.add(Not(colored[i-1][0]))
            s.add(Not(colored[i-1][1]))
            s.add(Not(colored[i+2][0]))
            s.add(Not(colored[i+2][1]))
        # Right column
        if puzzle[i][n-1] == puzzle[i+1][n-1] and puzzle[i][n-2] == puzzle[i+1][n-2] and puzzle[i][n-3] == puzzle[i+1][n-3]:
            s.add(Not(colored[i-1][n-1]))
            s.add(Not(colored[i-1][n-2]))
            s.add(Not(colored[i+2][n-1]))
            s.add(Not(colored[i+2][n-2]))

def double_edge_pair(s: Solver, colored: list, puzzle: list, n: int, encoding_size: dict) -> None:
    """ Pattern that can be found along borders where assumptions can be made to prevent isolated cells along the border

    Args:
        s (Solver): Solver to add assertions to
        colored (list): Matrix of BoolRef values for solver to fill
        puzzle (list): Matrix of Integers representing the number grid of the puzzle instance
        n (int): Size of the puzzle
        encoding_size (dict): Variable counts for this encoding
    """
    if n < 4:
        return
    
    for i in range(1, n-2):
        # Top row
        if puzzle[0][i] == puzzle[0][i+1] and puzzle[1][i] == puzzle[1][i+1]:
            s.add(Not(colored[0][i-1]))
            s.add(Not(colored[0][i+2]))
        # Bottom row
        if puzzle[n-1][i] == puzzle[n-1][i+1] and puzzle[n-2][i] == puzzle[n-2][i+1]:
            s.add(Not(colored[n-1][i-1]))
            s.add(Not(colored[n-1][i+2]))
        # Left column
        if puzzle[i][0] == puzzle[i+1][0] and puzzle[i][1] == puzzle[i+1][1]:
            s.add(Not(colored[i-1][0]))
            s.add(Not(colored[i+2][0]))
        # Right column
        if puzzle[i][n-1] == puzzle[i+1][n-1] and puzzle[i][n-2] == puzzle[i+1][n-2]:
            s.add(Not(colored[i-1][n-1]))
            s.add(Not(colored[i+2][n-1]))

def force_double_edge(s: Solver, colored: list, puzzle: list, n: int, encoding_size: dict) -> None:
    """ Pattern that can be found along borders where assumptions can be made to prevent isolated cells along the border

    Args:
        s (Solver): Solver to add assertions to
        colored (list): Matrix of BoolRef values for solver to fill
        puzzle (list): Matrix of Integers representing the number grid of the puzzle instance
        n (int): Size of the puzzle
        encoding_size (dict): Variable counts for this encoding
    """
    if n < 3:
        return
    for i in range(0, n):
        if i > 1:
            # Top row
            if puzzle[1][i-2] == puzzle[1][i-1]:
                s.add(Implies(colored[0][i], Not(colored[0][i-2])))
            # Bottom row
            if puzzle[n-2][i-2] == puzzle[n-2][i-1]:
                s.add(Implies(colored[n-1][i], Not(colored[n-1][i-2])))
            # Left column
            if puzzle[i-2][1] == puzzle[i-1][1]:
                s.add(Implies(colored[i][0], Not(colored[i-2][0])))
            # Right column
            if puzzle[i-2][n-2] == puzzle[i-1][n-2]:
                s.add(Implies(colored[i][n-1], Not(colored[i-2][n-1])))
        if i < n-2:
            # Top row
            if puzzle[1][i+2] == puzzle[1][i+1]:
                s.add(Implies(colored[0][i], Not(colored[0][i+2])))
            # Bottom row
            if puzzle[n-2][i+2] == puzzle[n-2][i+1]:
                s.add(Implies(colored[n-1][i], Not(colored[n-1][i+2])))
            # Left column
            if puzzle[i+2][1] == puzzle[i+1][1]:
                s.add(Implies(colored[i][0], Not(colored[i+2][0])))
            # Right column
            if puzzle[i+2][n-2] == puzzle[i+1][n-2]:
                s.add(Implies(colored[i][n-1], Not(colored[i+2][n-1])))

def close_edge(s: Solver, colored: list, puzzle: list, n: int, encoding_size: dict) -> None:
    """ Pattern that can be found along borders where assumptions can be made to prevent isolated cells along the border

    Args:
        s (Solver): Solver to add assertions to
        colored (list): Matrix of BoolRef values for solver to fill
        puzzle (list): Matrix of Integers representing the number grid of the puzzle instance
        n (int): Size of the puzzle
        encoding_size (dict): Variable counts for this encoding
    """
    if n < 3:
        return
    for i in range(0, n):
        if i > 1:
            # Top row
            s.add(Implies(And(colored[0][i], colored[1][i-1]), Not(colored[0][i-2])))
            # Bottom row
            s.add(Implies(And(colored[n-1][i], colored[n-2][i-1]), Not(colored[n-1][i-2])))
            # Left column
            s.add(Implies(And(colored[i][0], colored[i-1][1]), Not(colored[i-2][0])))
            # Right column
            s.add(Implies(And(colored[i][n-1], colored[i-1][n-2]), Not(colored[i-2][n-1])))
        if i < n-2:
            # Top row
            s.add(Implies(And(colored[0][i], colored[1][i+1]), Not(colored[0][i+2])))
            # Bottom row
            s.add(Implies(And(colored[n-1][i], colored[n-2][i+1]), Not(colored[n-1][i+2])))
            # Left column
            s.add(Implies(And(colored[i][0], colored[i+1][1]), Not(colored[i+2][0])))
            # Right column
            s.add(Implies(And(colored[i][n-1], colored[i+1][n-2]), Not(colored[i+2][n-1])))
            
            # Top row
            s.add(Implies(And(colored[0][i], colored[0][i+2]), Not(colored[1][i+1])))
            # Bottom row
            s.add(Implies(And(colored[n-1][i], colored[n-1][i+2]), Not(colored[n-2][i+1])))
            # Left column
            s.add(Implies(And(colored[i][0], colored[i+2][0]), Not(colored[i+1][1])))
            # Right column
            s.add(Implies(And(colored[i][n-1], colored[i+2][n-1]), Not(colored[i+1][n-2])))

def border_close(s: Solver, colored: list, puzzle: list, n: int, encoding_size: dict) -> None:
    """ Pattern that can be found along borders where assumptions can be made to prevent isolated cells along the border

    Args:
        s (Solver): Solver to add assertions to
        colored (list): Matrix of BoolRef values for solver to fill
        puzzle (list): Matrix of Integers representing the number grid of the puzzle instance
        n (int): Size of the puzzle
        encoding_size (dict): Variable counts for this encoding
    """
    for i in range(n):
        # Top row
        if puzzle[0][i] == puzzle[1][i]:
            if i > 1:
                if puzzle[0][i] == puzzle[1][i-1]:
                    s.add(Implies(colored[0][i-2], colored[1][i]))
            if i < n-2:
                if puzzle[0][i] == puzzle[1][i+1]:
                    s.add(Implies(colored[0][i+2], colored[1][i]))
         
        # Bottom row
        if puzzle[n-1][i] == puzzle[n-2][i]:
            if i > 1:
                if puzzle[n-1][i] == puzzle[n-2][i-1]:
                    s.add(Implies(colored[n-1][i-2], colored[n-2][i]))
            if i < n-2:
                if puzzle[n-1][i] == puzzle[n-2][i+1]:
                    s.add(Implies(colored[n-1][i+2], colored[n-2][i]))
        
        # Left column
        if puzzle[i][0] == puzzle[i][1]:
            if i > 1:
                if puzzle[i][0] == puzzle[i-1][1]:
                    s.add(Implies(colored[i-2][0], colored[i][1]))
            if i < n-2:
                if puzzle[i][0] == puzzle[i+1][1]:
                    s.add(Implies(colored[i+2][0], colored[i][1]))
        
        # Right column
        if puzzle[i][n-1] == puzzle[i][n-2]:
            if i > 1:
                if puzzle[i][n-1] == puzzle[i-1][n-2]:
                    s.add(Implies(colored[i-2][n-1], colored[i][n-2]))
            if i < n-2:
                if puzzle[i][n-1] == puzzle[i+1][n-2]:
                    s.add(Implies(colored[i+2][n-1], colored[i][n-2]))