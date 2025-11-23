import argparse
import os
import sys

VALID_EXTENSIONS = [".singles"]

def is_puzzle(path):
    _, ext = os.path.splitext(path)
    return ext.lower() in VALID_EXTENSIONS

def read_puzzle(path):
    with open(path, "r") as f:
        lines = input.readlines()

    try:
        n = int(lines[0])
    except (ValueError, IndexError):
        sys.exit(f"Error: First line must contain puzzle size in {path}")

    if len(lines) < n + 2:
        sys.exit(f"Error: Wrong puzzle format in {path}")

    grid = []
    for i in range(2, n + 2):
        row_values = lines[i].split()

        if len(row_values) != n:
            sys.exit(f"Error: Row length mismatch in {path}")
            
        try:
            row = [int(x) for x in row_values]
        except ValueError:
            sys.exit(f"Error: Non-integer value in {path}")

        grid.append(row)
    return grid

def read_file(path):
    if not os.path.exists(path):
        sys.exit(f"Error: File does not exist at {path}")
    if not is_puzzle(path):
        sys.exit(f"Error: Not a puzzle file at {path}")

    puzzle = read_puzzle(path)
    return [(path, puzzle)]

def read_folder(path):
    if not os.path.isdir(path):
        sys.exit(f"Error: Not a directory at {path}")

    puzzles = []
    for filename in os.listdir(path):
        file = os.path.join(path, filename)

        # Filter out subfolders
        if not os.path.isfile(file):
            continue

        puzzles.extend(read_file(file))
    if not puzzles:
        sys.exit(f"Error: No puzzle files found in {path}")
    return puzzles

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hitori solver using Z3")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-f", "--file", type=str, help="Path to a puzzle file")
    group.add_argument("-d", "--folder", type=str, help="Path to folder containing puzzle files")
    args = parser.parse_args()

    if args.file:
        puzzles = read_file(args.file)
    else:
        puzzles = read_folder(args.folder)
