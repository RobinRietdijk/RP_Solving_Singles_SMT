import os
import sys
import ast

PUZZLE_EXTENSIONS = [".singles"]
SOLUTION_EXTENSIONS = [".singlessol"]

def _is_puzzle(path: str) -> bool:
    _, ext = os.path.splitext(path)
    return ext.lower() in PUZZLE_EXTENSIONS

def _is_solution(path: str) -> bool:
    _, ext = os.path.splitext(path)
    return ext.lower() in SOLUTION_EXTENSIONS

def _parse_file(path: str) -> tuple[list[list[int|str]], str|None]:
    _, ext = os.path.splitext(path)
    with open(path, "r") as f:
        lines = f.readlines()

    try:
        n = int(lines[0])
    except (ValueError, IndexError):
        sys.exit(f"Error: First line must contain size in {path}")

    if len(lines) < n + 2:
        sys.exit(f"Error: Wrong format in {path}")

    grid = []
    for i in range(2, n + 2):
        row_values = lines[i].split()

        if len(row_values) != n:
            sys.exit(f"Error: Row length mismatch in {path}")
            
        try:
            if ext.lower() in SOLUTION_EXTENSIONS:
                row = row_values
            else:
                row = [int(x) for x in row_values]
        except ValueError:
            sys.exit(f"Error: Unknown value type in {path}")

        grid.append(row)
    
    seed = None
    if len(lines) > n+3:
        line = lines[n+3].strip()
        if line.startswith("@"):
            seed = line[1:]
        else:
            print(f"Warning: seed not prepended with @ in {path}")
    
    return grid, seed

def _read_file(path: str, puzzles: bool, strict: bool) -> list[tuple[str, list[list[int|str]], str|None]]:
    if not os.path.exists(path):
        sys.exit(f"Error: File does not exist at {path}")
    if (puzzles and not _is_puzzle(path)) or (not puzzles and not _is_solution(path)):
        if strict:
            sys.exit(f"Error: Not a {'puzzle' if puzzles else 'solution'} file at {path}")
        else:
            print(f"Warning: Not a {'puzzle' if puzzles else 'solution'} file at {path}")
            return []

    grid, seed = _parse_file(path)
    return [(path, grid, seed)]

def _read_dir(path: str, puzzles: bool, strict: bool, recursive: bool) -> list[tuple[str, list[list[int|str]], str|None]]:
    if not os.path.isdir(path):
        sys.exit(f"Error: Not a directory at {path}")

    grids = []
    for filename in os.listdir(path):
        file = os.path.join(path, filename)

        if not os.path.isfile(file):
            if recursive:
                grids.extend(_read_dir(file, puzzles, strict, recursive))
            continue

        grids.extend(_read_file(file, puzzles, strict))
    return grids

def read_puzzle(path: str, strict: bool) -> tuple[str, list[list[int|str]], str]:
    result = _read_file(path, True, strict)
    if not result:
        sys.exit(f"Error: No puzzle file found at {path}")
    return result[0]

def read_solution(path: str, strict: bool) -> tuple[str, list[list[int|str]], str]:
    result = _read_file(path, False, strict)
    if not result:
        sys.exit(f"Error: No solution file found at {path}")
    return result[0]

def read_puzzle_dir(path: str, recursive: bool, strict: bool) -> list[tuple[str, list[list[int|str]], str|None]]:
    puzzles = _read_dir(path, True, strict, recursive)
    if not puzzles:
        sys.exit(f"Error: No puzzle files found in {path}")
    return puzzles

def read_solution_dir(path: str, recursive: bool, strict: bool) -> list[tuple[str, list[list[int|str]], str|None]]:
    solutions = _read_dir(path, False, strict, recursive)
    if not solutions:
        sys.exit(f"Error: No solution files found in {path}")
    return solutions

def write_file(path: str, puzzle: list[list[int|str]], seed: str|None, extra: str|None) -> None:
    with open(path, "w") as file:
        file.write(f"{len(puzzle)}\n\n")
        for row in puzzle:
            file.write(" ".join(map(str, row)) + "\n")
        if seed is not None:
            file.write(f"\n@{seed}")
        if extra is not None:
            for line in extra.splitlines():
                file.write(f"\n#{line}")

def append_comment(path: str, comment: str) -> None:
    with open(path, "a") as file:
        for line in comment.splitlines():
            file.write(f"\n#{line}")

def append_dict(path: str, new_dict: dict) -> None:
    with open(path, "r") as file:
        lines = file.readlines()
    
    try:
        n = int(lines[0].strip())
    except (ValueError, IndexError):
        sys.exit(f"Error: First line must contain size in {path}")

    try:
        header = lines[:n+4]
        comments = lines[n+4:] if len(lines) > n+4 else []
    except IndexError:
        sys.exit(f"Error: Wrong number if lines in file {path}")
    
    data = {}
    for line in comments:
        line = line.strip()
        if not line.startswith("# ~["):
            continue
        try:
            tkey, tvalue = line[4:].split("]:", 1)
            key = tkey.strip()
            value = ast.literal_eval(tvalue.strip())
            data[key] = value
        except Exception:
            continue
    
    data.update(new_dict)
    to_pop = []
    for key in data.keys():
        if key not in new_dict.keys():
            to_pop.append(key)
    for key in to_pop:        
        data.pop(key, None)

    if not header[len(header)-1].endswith("\n"):
        header.append("\n")
    comments = [l for l in comments if not l.strip().startswith("# ~[")]
    if len(comments) > 0 and not comments[len(comments)-1].endswith("\n"):
        comments.append("\n")
    comments.extend([f"# ~[{k}]: {repr(v)}\n" for k, v in data.items()])

    with open(path, "w") as file:
        file.writelines(header+comments)
        