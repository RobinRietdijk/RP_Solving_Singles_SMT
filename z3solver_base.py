from z3 import * # type: ignore

def uniqueness_pairs(s: Solver, colored: list, puzzle: list, n: int, encoding_size: dict) -> None:
    """ Implementation of the uniqueness rule of Hitori, done by checking for equal pairs and not allowing both the be white

    Args:
        s (Solver): Solver to add assertions to
        colored (list): Matrix of BoolRef values for solver to fill
        puzzle (list): Matrix of Integers representing the number grid of the puzzle instance
        n (int): Size of the puzzle
        encoding_size (dict): Variable counts for this encoding
    """
    # Rows
    for i in range(n):
        for j in range(n):
            for k in range(j+1, n):
                if (puzzle[i][j] == puzzle[i][k]):
                    s.add(Or(colored[i][j], colored[i][k]))
    # Columns       
    for i in range(n):
        for j in range(n):
            for k in range(j+1, n):
                if (puzzle[j][i] == puzzle[k][i]):
                    s.add(Or(colored[j][i], colored[k][i]))

# Alternate implemention of the uniquecells constraint by counting the values and asserting at most 1 value per column and row
def uniqueness_atmost(s: Solver, colored: list, puzzle: list, n: int, encoding_size: dict) -> None:
    """ Implementation of the uniqueness rule of Hitori, done by counting each number occurence and enforcing this to be at most 1

    Args:
        s (Solver): Solver to add assertions to
        colored (list): Matrix of BoolRef values for solver to fill
        puzzle (list): Matrix of Integers representing the number grid of the puzzle instance
        n (int): Size of the puzzle
        encoding_size (dict): Variable counts for this encoding
    """
    for i in range(n):
        row_values = {}
        col_values = {}

        for j in range(n):
            v_r = puzzle[i][j]
            v_c = puzzle[j][i]

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

def neighbours(s: Solver, colored: list, puzzle: list, n: int, encoding_size: dict) -> None:
    """ Implementation of the neighbours rule of Hitori, done by only allowing this or its neighbours to be colored

    Args:
        s (Solver): Solver to add assertions to
        colored (list): Matrix of BoolRef values for solver to fill
        puzzle (list): Matrix of Integers representing the number grid of the puzzle instance
        n (int): Size of the puzzle
        encoding_size (dict): Variable counts for this encoding
    """
    for i in range(n):
        for j in range(n):
            # Horizontal neighbours
            if i+1 < n:
                s.add(Not(And(colored[i][j], colored[i+1][j])))
            # Vertical neighbours
            if j+1 < n:
                s.add(Not(And(colored[i][j], colored[i][j+1])))

def connectivity_ranking(s: Solver, colored: list, puzzle: list, n: int, encoding_size: dict) -> None:
    """ Implementation of the connectivity rule of Hitori, done by ranking each white cell in the grid and following the path from highest to lowest rank

    Args:
        s (Solver): Solver to add assertions to
        colored (list): Matrix of BoolRef values for solver to fill
        puzzle (list): Matrix of Integers representing the number grid of the puzzle instance
        n (int): Size of the puzzle
        encoding_size (dict): Variable counts for this encoding
    """
    root_row = Int("root_r")
    encoding_size["int_vars"] += 1
    root_col = Int("root_c")
    encoding_size["int_vars"] += 1

    # The root cell is located at either (0, 0) or (0, 1), as the neighbours rule does not allow both to be black
    s.add(root_row == 0)
    s.add(Or(root_col == 0, root_col == 1))
    s.add(Or(And(root_col == 0, Not(colored[0][0])), And(root_col == 1, Not(colored[0][1]))))

    rank = [[Int(f"num_{r}_{c}") for c in range(n)] for r in range(n)]
    encoding_size["int_vars"] += n*n

    # Setup ranking rules for each cell
    for i in range(n):
        for j in range(n):
            # Colored cells have a negative rank
            s.add(Implies(colored[i][j], rank[i][j] == -1))
            # Root has rank 1
            s.add(Implies(And(Not(colored[i][j]), i == root_row, j == root_col), rank[i][j] == 0))
            # Non-root white cells have a positive rank
            s.add(Implies(And(Not(colored[i][j]), Not(And(i == root_row, j == root_col))), rank[i][j] > 0))
            
    for i in range(n):
        for j in range(n):
            # Setup rules for the non-root white cells, which must always have atleast one white neighbour with a lower rank.
            conditions = []
            if i > 0:
                conditions.append(And(Not(colored[i-1][j]), rank[i-1][j] < rank[i][j]))
            if j > 0:
                conditions.append(And(Not(colored[i][j-1]), rank[i][j-1] < rank[i][j]))
            if i+1 < n:
                conditions.append(And(Not(colored[i+1][j]), rank[i+1][j] < rank[i][j]))
            if j+1 < n:
                conditions.append(And(Not(colored[i][j+1]), rank[i][j+1] < rank[i][j]))

            # Add rules only if this cell is not colored and not the root
            if conditions:
                s.add(Implies(
                        And(Not(colored[i][j]), Not(And(root_row == i, root_col == j))),
                        Or(*conditions)
                    ))
                
def connectivity_ranking_alt(s: Solver, colored: list, puzzle: list, n: int, encoding_size: dict) -> None:
    """ Implementation of the connectivity rule of Hitori, done by ranking each white cell in the grid and following the path from highest to lowest rank
    A slight alteration of the other ranking approach

    Args:
        s (Solver): Solver to add assertions to
        colored (list): Matrix of BoolRef values for solver to fill
        puzzle (list): Matrix of Integers representing the number grid of the puzzle instance
        n (int): Size of the puzzle
        encoding_size (dict): Variable counts for this encoding
    """
    max_rank = n*n-1
    # Allow the solver to randomly assign the root based on rules
    rank = [[Int(f"rank_{i}_{j}") for j in range(n)] for i in range(n)]
    encoding_size["int_vars"] += n*n
    root = [[Bool(f"root_{i}_{j}") for j in range(n)] for i in range(n)]
    encoding_size["bool_vars"] += n*n

    # Flatten the grid of root Boolean vars to a list
    root_flat = [root[i][j] for i in range(n) for j in range(n)]
    # Ensure that only a single root Boolean is true
    s.add(PbEq([(r, 1) for r in root_flat], 1))

    # Setup ranking rules for each cell
    for i in range(n):
        for j in range(n):
            # Colored cells have a negative rank
            s.add(Implies(colored[i][j], And(rank[i][j] == -1, Not(root[i][j]))))
            # Root has rank 1
            s.add(Implies(root[i][j], And(Not(colored[i][j]), rank[i][j] == 0)))
            # Non-root white cells have a positive rank
            s.add(Implies(And(Not(colored[i][j]), Not(root[i][j])), And(rank[i][j] > 0, rank[i][j] <= max_rank)))
            
    for i in range(n):
        for j in range(n):
            # Setup rules for the non-root white cells, which must always have atleast one white neighbour with a lower rank.
            conditions = []
            if i > 0:
                conditions.append(And(Not(colored[i-1][j]), rank[i-1][j] < rank[i][j]))
            if j > 0:
                conditions.append(And(Not(colored[i][j-1]), rank[i][j-1] < rank[i][j]))
            if i+1 < n:
                conditions.append(And(Not(colored[i+1][j]), rank[i+1][j] < rank[i][j]))
            if j+1 < n:
                conditions.append(And(Not(colored[i][j+1]), rank[i][j+1] < rank[i][j]))

            # Add rules only if this cell is not colored and not the root
            if conditions:
                s.add(Implies(
                        And(Not(colored[i][j]), Not(root[i][j])),
                        Or(*conditions)
                    ))
                
def connectivity_tree(s: Solver, colored: list, puzzle: list, n: int, encoding_size: dict) -> None:
    """ Implementation of the connectivity rule of Hitori, done by building a tree structure over the white cells and ensuring full connectiveness

    Args:
        s (Solver): Solver to add assertions to
        colored (list): Matrix of BoolRef values for solver to fill
        puzzle (list): Matrix of Integers representing the number grid of the puzzle instance
        n (int): Size of the puzzle
        encoding_size (dict): Variable counts for this encoding
    """

    max_rank = n*n-1
    depth = [[Int(f"rank_{i}_{j}") for j in range(n)] for i in range(n)]
    encoding_size["int_vars"] += n*n
    root = [[Bool(f"root_{i}_{j}") for j in range(n)] for i in range(n)]
    encoding_size["bool_vars"] += n*n

    # Setup Boolean variables for each cells to indicate their parent neighbour cell
    parent_up = [[Bool(f"Up_{i}_{j}") for j in range(n)] for i in range(n)]
    encoding_size["bool_vars"] += n*n
    parent_down = [[Bool(f"Down_{i}_{j}") for j in range(n)] for i in range(n)]
    encoding_size["bool_vars"] += n*n
    parent_left = [[Bool(f"Left_{i}_{j}") for j in range(n)] for i in range(n)]
    encoding_size["bool_vars"] += n*n
    parent_right = [[Bool(f"Right_{i}_{j}") for j in range(n)] for i in range(n)]
    encoding_size["bool_vars"] += n*n

    # Root is chosen by the solver by only allowing a single root Boolean to be true in the grid
    root_flat = [root[i][j] for i in range(n) for j in range(n)]
    s.add(PbEq([(r, 1) for r in root_flat], 1))

    for i in range(n):
        for j in range(n):
            parents = [parent_up[i][j], parent_down[i][j], parent_left[i][j], parent_right[i][j]]
            # Colored cells have a negative depth and no parents
            s.add(Implies(colored[i][j], And(depth[i][j] == -1, Not(root[i][j]), *[Not(p) for p in parents])))
            # The root cell has a depth of 0 and no parents
            s.add(Implies(root[i][j], And(Not(colored[i][j]), depth[i][j] == 0, *[Not(p) for p in parents])))
            # All non-root white cells have a positive depth and exactly 1 parent
            s.add(Implies(And(Not(colored[i][j]), Not(root[i][j])), And(depth[i][j] > 0, depth[i][j] <= max_rank, PbEq([(parents[k], 1) for k in range(4)], 1))))
            
            # Rules are added such that all non-root white cells have a single parent that is not white and has a lower depth
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
                
def connectivity_bitvector(s: Solver, colored: list, puzzle: list, n: int, encoding_size: dict) -> None:
    """ Implementation of the connectivity rule of Hitori, done by implementing the ranking approach using BitVectors

    Args:
        s (Solver): Solver to add assertions to
        colored (list): Matrix of BoolRef values for solver to fill
        puzzle (list): Matrix of Integers representing the number grid of the puzzle instance
        n (int): Size of the puzzle
        encoding_size (dict): Variable counts for this encoding
    """
    max_num = n*n+1
    k = max_num.bit_length()

    root_row = BitVec("root_r", k)
    encoding_size["bv_vars"] += 1
    root_col = BitVec("root_c", k)
    encoding_size["bv_vars"] += 1
    zero_bv = BitVecVal(0, k)
    n_bv = BitVecVal(n, k)
    max_bv = BitVecVal(max_num, k)
    max_valid = BitVecVal(max_num-1, k)

    # Helper variables to indicate white cells and the root cell
    is_white = [[Bool(f"white_{i}_{j}") for j in range(n)] for i in range(n)]
    encoding_size["bool_vars"] += n*n
    is_root = [[Bool(f"root_{i}_{j}") for j in range(n)] for i in range(n)]
    encoding_size["bool_vars"] += n*n

    s.add(ULT(root_row, n_bv))
    s.add(ULT(root_col, n_bv))

    Number = [[BitVec(f"num_{r}_{c}", k) for c in range(n)] for r in range(n)]
    encoding_size["bv_vars"] += n*n

    row_bvs = [BitVecVal(i, k) for i in range(n)]
    col_bvs = [BitVecVal(i, k) for i in range(n)]
    for i in range(n):
        for j in range(n):
            i_bv = row_bvs[i]
            j_bv = col_bvs[j]
            
            s.add(is_white[i][j] == Not(colored[i][j]))
            s.add(is_root[i][j] == And(root_row == i_bv, root_col == j_bv))

            # Colored cells have the max number since negative numbers are not possible
            s.add(Implies(Not(is_white[i][j]), Number[i][j] == max_bv))
            # Limit white cell numbers
            s.add(Implies(is_white[i][j], ULE(Number[i][j], max_valid)))

            # Root has rank 1
            s.add(Implies(And(is_white[i][j], is_root[i][j]), Number[i][j] == zero_bv))
            # Non-root white cells have a positive rank
            s.add(Implies(And(is_white[i][j], Not(is_root[i][j])), UGT(Number[i][j], zero_bv)))
            
    for i in range(n):
        for j in range(n):
            # Setup rules for the non-root white cells, which must always have atleast one white neighbour with a lower rank.
            conditions = []
            if i > 0:
                conditions.append(And(is_white[i-1][j], ULT(Number[i-1][j], Number[i][j])))
            if j > 0:
                conditions.append(And(is_white[i][j-1], ULT(Number[i][j-1], Number[i][j])))
            if i+1 < n:
                conditions.append(And(is_white[i+1][j], ULT(Number[i+1][j], Number[i][j])))
            if j+1 < n:
                conditions.append(And(is_white[i][j+1], ULT(Number[i][j+1], Number[i][j])))

            # Add rules only if this cell is not colored and not the root
            if conditions:
                s.add(Implies(And(is_white[i][j], Not(is_root[i][j])), Or(*conditions)))

def connectivity_boolean(s: Solver, colored: list, puzzle: list, n: int, encoding_size: dict) -> None:
    """ Implementation of the connectivity rule of Hitori, done by implementing a breath-first search using Booleans

    Args:
        s (Solver): Solver to add assertions to
        colored (list): Matrix of BoolRef values for solver to fill
        puzzle (list): Matrix of Integers representing the number grid of the puzzle instance
        n (int): Size of the puzzle
        encoding_size (dict): Variable counts for this encoding
    """
    max_steps = n*n+1
    root00 = Bool("root_0_0")
    encoding_size["bool_vars"] += 1
    root01 = Bool("root_0_1")
    encoding_size["bool_vars"] += 1
    # The root cell is located at either (0, 0) or (0, 1), as the neighbours rule does not allow both to be black
    s.add(Xor(root00, root01))
    s.add(Implies(root00, Not(colored[0][0])))
    s.add(Implies(root01, Not(colored[0][1])))

    # A 3D matrix to indicate wheter the cells have been visited in that step or previous steps
    visited = [[[Bool(f"visited_{k}_{i}_{j}") for j in range(n)] for i in range(n)] for k in range(max_steps+1)]
    encoding_size["bool_vars"] += n*n*(max_steps+1)
    for i in range(n):
        for j in range(n):
            if i == 0 and j == 0:
                # If (0, 0) is the root then it is visited on step 0
                s.add(visited[0][0][0] == root00)
            elif i == 0 and j == 1:
                # If (0, 1) is the root then it is visited on step 0
                s.add(visited[0][0][1] == root01)
            else:
                # All non-root cells are not visited on step 1
                s.add(visited[0][i][j] == False)
    
    # Step by step
    for k in range(1, max_steps+1):
        # For each cell
        for i in range(n):
            for j in range(n):
                # Check if any neighbour has been visited in the previous step
                neighbours = []
                if i > 0:
                    neighbours.append(visited[k-1][i-1][j])
                if j > 0:
                    neighbours.append(visited[k-1][i][j-1])
                if i < n-1:
                    neighbours.append(visited[k-1][i+1][j])
                if j < n-1:
                    neighbours.append(visited[k-1][i][j+1])
                
                or_neighbours = Or(neighbours) if neighbours else False
                # We can visit this cell in this step if any neighbour, or itself, has been visited in the previous step
                can_visit = And(Not(colored[i][j]), Or(visited[k-1][i][j], or_neighbours))

                s.add(Implies(visited[k][i][j], can_visit))
                s.add(Implies(can_visit, visited[k][i][j]))
    
    # Ensure that all non-colored cells are visited during the BFS
    for i in range(n):
        for j in range(n):
            s.add(Implies(Not(colored[i][j]), visited[max_steps][i][j]))    