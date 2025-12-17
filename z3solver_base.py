from z3 import * # type: ignore

# Distinct tactic following Z3py basics, SMT/SAT describes a one-hot approach for latin squares
def uniqueness_pairs(s: Solver, colored: list, puzzle: list, n: int, encoding_size: dict) -> None:
    for i in range(n):
        for j in range(n):
            for k in range(j+1, n):
                # If 2 cells have an equal symbol, we must color one black
                if (puzzle[i][j] == puzzle[i][k]):
                    s.add(Or(colored[i][j], colored[i][k]))
                    
    for i in range(n):
        for j in range(n):
            for k in range(j+1, n):
                if (puzzle[j][i] == puzzle[k][i]):
                    s.add(Or(colored[j][i], colored[k][i]))

# Alternate implemention of the uniquecells constraint by counting the values and asserting at most 1 value per column and row
def uniqueness_atmost(s: Solver, colored: list, puzzle: list, n: int, encoding_size: dict) -> None:
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

# Trivial, remove i-1 and j-1, since we already check those in earlier cells
# Van der Knijff uses a slightly different approach where two cells cannot be both white if they have equal numbers
def neighbours(s: Solver, colored: list, puzzle: list, n: int, encoding_size: dict) -> None:
    for i in range(n):
        for j in range(n):
            if i+1 < n:
                # Horizontal neighbours
                s.add(Not(And(colored[i][j], colored[i+1][j])))
            if j+1 < n:
                # Vertical neighbours
                s.add(Not(And(colored[i][j], colored[i][j+1])))

def connectivity_ranking(s: Solver, colored: list, puzzle: list, n: int, encoding_size: dict) -> None:
    root_row = Int("root_r")
    encoding_size["int_vars"] += 1
    root_col = Int("root_c")
    encoding_size["int_vars"] += 1
    s.add(root_row == 0)
    s.add(Or(root_col == 0, root_col == 1))
    s.add(Or(And(root_col == 0, Not(colored[0][0])), And(root_col == 1, Not(colored[0][1]))))

    rank = [[Int(f"num_{r}_{c}") for c in range(n)] for r in range(n)]
    encoding_size["int_vars"] += n*n

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
                
def connectivity_ranking_alt(s: Solver, colored: list, puzzle: list, n: int, encoding_size: dict) -> None:
    max_rank = n*n-1
    rank = [[Int(f"rank_{i}_{j}") for j in range(n)] for i in range(n)]
    encoding_size["int_vars"] += n*n
    root = [[Bool(f"root_{i}_{j}") for j in range(n)] for i in range(n)]
    encoding_size["bool_vars"] += n*n

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
                
def connectivity_tree(s: Solver, colored: list, puzzle: list, n: int, encoding_size: dict) -> None:
    max_rank = n*n-1
    depth = [[Int(f"rank_{i}_{j}") for j in range(n)] for i in range(n)]
    encoding_size["int_vars"] += n*n
    root = [[Bool(f"root_{i}_{j}") for j in range(n)] for i in range(n)]
    encoding_size["bool_vars"] += n*n

    parent_up = [[Bool(f"Up_{i}_{j}") for j in range(n)] for i in range(n)]
    encoding_size["bool_vars"] += n*n
    parent_down = [[Bool(f"Down_{i}_{j}") for j in range(n)] for i in range(n)]
    encoding_size["bool_vars"] += n*n
    parent_left = [[Bool(f"Left_{i}_{j}") for j in range(n)] for i in range(n)]
    encoding_size["bool_vars"] += n*n
    parent_right = [[Bool(f"Right_{i}_{j}") for j in range(n)] for i in range(n)]
    encoding_size["bool_vars"] += n*n

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
                
def connectivity_bitvector(s: Solver, colored: list, puzzle: list, n: int, encoding_size: dict) -> None:
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

def connectivity_boolean(s: Solver, colored: list, puzzle: list, n: int, encoding_size: dict) -> None:
    max_steps = n*n+1
    root00 = Bool("root_0_0")
    encoding_size["bool_vars"] += 1
    root01 = Bool("root_0_1")
    encoding_size["bool_vars"] += 1
    s.add(Xor(root00, root01))
    s.add(Implies(root00, Not(colored[0][0])))
    s.add(Implies(root01, Not(colored[0][1])))

    visited = [[[Bool(f"visited_{k}_{i}_{j}") for j in range(n)] for i in range(n)] for k in range(max_steps+1)]
    encoding_size["bool_vars"] += n*n*(max_steps+1)
    for i in range(n):
        for j in range(n):
            if i == 0 and j == 0:
                s.add(visited[0][0][0] == root00)
            elif i == 0 and j == 1:
                s.add(visited[0][0][1] == root01)
            else:
                s.add(visited[0][i][j] == False)
    
    for k in range(1, max_steps+1):
        for i in range(n):
            for j in range(n):
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
                can_visit = And(Not(colored[i][j]), Or(visited[k-1][i][j], or_neighbours))

                s.add(Implies(visited[k][i][j], can_visit))
                s.add(Implies(can_visit, visited[k][i][j]))
    
    for i in range(n):
        for j in range(n):
            s.add(Implies(Not(colored[i][j]), visited[max_steps][i][j]))    