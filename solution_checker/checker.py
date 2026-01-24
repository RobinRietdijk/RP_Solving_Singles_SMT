from collections import deque

# This solution checker was made as part of the collaboration of this project
# The original author of this specific implementation is Sophieke van Luenen (https://github.com/Sophieke32) and can be found in:
# https://github.com/sappho3/Thesis-Hitori-shared/tree/main/solutionChecker

def check_puzzle(solution: list) -> bool:
    """ Checks the validity of a solution

    Args:
        solution (list): A solution read from a solution file

    Returns:
        bool: True if the solution is valid, False otherwise
    """
    n = len(solution)
    rows = [set() for _ in range(n)]
    cols = [set() for _ in range(n)]
    num_white = 0
    start = None

    for i in range(n):
        for j in range(n):
            val = solution[i][j]
            # Neighbour check
            if val.endswith("B"):
                if i+1 < n:
                    if solution[i+1][j].endswith("B"):
                        return False
                if j+1 < n:
                    if solution[i][j+1].endswith("B"):
                        return False
                continue
            # Gather a root for the connectivity check
            if start is None:
                start = (i, j)

            # Uniqueness check
            if val in rows[i] or val in cols[j]:
                return False
            rows[i].add(val)
            cols[j].add(val)
            num_white += 1

    if num_white == 0 or start is None:
        return False

    # Connectivity check
    visited = set([start])
    queue = deque([start])
    while queue:
        i, j = queue.popleft()
        for di, dj in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            ni, nj = i+di, j+dj
            if 0 <= ni < n and 0 <= nj < n:
                if solution[ni][nj].endswith("B"):
                    continue
                if (ni, nj) not in visited:
                    visited.add((ni, nj))
                    queue.append((ni, nj))

    return len(visited) == num_white