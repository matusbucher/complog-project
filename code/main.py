"""
Planner for Sokoban game solutions. Provides parsing of textual maps, compilation to a SAT planning encoding, solving via an external MiniSat solver, and interpreting the solution.

Usage (CLI):
    python main.py [-s MAXSTEPS] [-i input] [-o output] [-r readablecnf] <mapfile> <minisat-path>

Dependencies:
- Python 3.7+
- NumPy library
- An external MiniSat solver binary
"""

import argparse

from logic.logic_program_interface import Action, Solution
from logic.planner import Planner
from logic.map_parser import MapParser, MapParserError


ARGPARSER = argparse.ArgumentParser(
    prog="python main.py",
    description="Solver for Sokoban game using SAT planning.")    

ARGPARSER.add_argument("mapfile", type=str, help="Path to the Sokoban map file.")
ARGPARSER.add_argument("minisat", type=str, help="Path to the MiniSat executable.")
ARGPARSER.add_argument("-O", "--optimize", action="store_true", help="Enable optimizations in the planning encoding.")
ARGPARSER.add_argument("-s", "--maxsteps", type=int, metavar="MAXSTEPS", default=50, help="Maximum number of steps to search for a solution (default 50).")
ARGPARSER.add_argument("-i", "--input", type=str, metavar="FILENAME", default="solver_input.cnf", help="File where the input for MiniSat will be stored. The file will be created/overwritten, so it need not exist beforehand (default 'solver_input.cnf').")
ARGPARSER.add_argument("-o", "--output", type=str, metavar="FILENAME", default="solver_output.cnf", help="File where the output from MiniSat will be stored. The file will be created/overwritten, so it need not exist beforehand (default 'solver_output.cnf').")
ARGPARSER.add_argument("-r", "--readablecnf", type=str, metavar="FILENAME", default=None, help="If provided, a human-readable version of the generated CNF will be saved to this file.")


def print_solution(solution: Solution) -> None:
    for action, x, y, direction, step in solution:
        if action == Action.NOOP:
            print(f"Step {step}: noop")
        else:
            print(f"Step {step}: {action.to_string()} {direction.to_string()} from ({x}, {y})")


if __name__ == "__main__":
    args = ARGPARSER.parse_args()

    try:
        sokoban_map = MapParser.from_file(args.mapfile)
    except (FileNotFoundError, MapParserError) as e:
        print(f"Error: {e}")
        exit(1)

    try:
        planner = Planner(
            sokoban_map=sokoban_map,
            max_steps=args.maxsteps,
            solver_path=args.minisat,
            solver_input=args.input,
            solver_output=args.output,
            optimize=args.optimize
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)

    solvable = planner.check_solvability()
    if not solvable:
        print("Problem is not solvable within the given maximum number of steps.")
    else:
        print("Problem is solvable, searching for shortest solution.")
        solution = planner.find_shortest_solution()
        print_solution(solution)

    if args.readablecnf is not None:
        planner.save_readable_cnf(args.readablecnf)