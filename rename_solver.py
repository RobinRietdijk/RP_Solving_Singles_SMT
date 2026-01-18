import os
import argparse
from file_utils import read_csv, write_csv

def _rename_command(args: dict) -> None:
    csv = read_csv(args.path)
    for i in csv:
        i["solver"] = args.new_name
    base_path = os.path.dirname(os.path.splitext(args.path)[0])
        
    write_csv(csv, base_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rename solver")
    parser.add_argument('path', type=str, help="Path to file")
    parser.add_argument('new_name', type=str, help="New name of the solver")
    parser.set_defaults(func=_rename_command)

    args = parser.parse_args()
    args.func(args)