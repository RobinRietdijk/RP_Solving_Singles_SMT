from clingo.control import Control
from clingo import Model

from typing import List

'''
This solver is a version of the ASP solver developed by Sappho de Nooij (https://github.com/sappho3)
'''

def solve(grid: List[List[int]]) -> tuple[bool, bool]:
    n = len(grid)

    model = open("generator/model.lp").read()
    ctl = Control(["-c", f"n={n}"])
    ctl.configuration.solve.models="2" # try to find at least 2 models
    ctl.add("base", [], model)

    for y, row in enumerate(grid):
        for x, number in enumerate(row):
            ctl.add("base", [], f"cell({x + 1}, {y + 1}, {number}).");
    ctl.ground([("base", [])])

    global num_models
    num_models = 0

    def on_model(m: Model):
        global num_models
        num_models += 1

    ctl.solve(on_model=on_model)

    has_solution = num_models > 0
    unique = num_models == 1
    return has_solution, unique

