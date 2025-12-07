from __future__ import annotations

"""
sokoban.py
--------------
Planner for Sokoban game solutions. Provides parsing of textual maps,
compilation to a SAT planning encoding, and solving via an external MiniSat solver.

This module contains:
- small helpers and enums for tiles and directions,
- a `Parser` class to convert textual maps into an internal `SokobanMap`,
- a `LogicProgram` class that compiles a planning encoding into CNF (DIMACS or human-readable),
- a `Planner` class that uses the `LogicProgram` and an external MiniSat binary to find the minimal-step plan.

Usage (CLI):
    python sokoban.py <mapfile> <minisat-path> [-s MAXSTEPS] [-i input] [-o output] [-r readablecnf]

Dependencies:
- Python 3.7+
- NumPy library
- An external MiniSat solver binary
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import subprocess
import argparse
import os

import numpy as np


Position = Tuple[int, int]
PathLike = Union[str, Path]


class Tile(Enum):
    WALL = "#"
    STORAGE = "X"
    FLOOR = " "


class Direction(Enum):
    RIGHT = 0
    UP = 1
    LEFT = 2
    DOWN = 3

    def to_string(self) -> str:
        """
        Return a string name representing the direction.
        """
        match self:
            case Direction.RIGHT:
                return "right"
            case Direction.UP:
                return "up"
            case Direction.LEFT:
                return "left"
            case Direction.DOWN:
                return "down"
    
    def to_vector(self) -> Tuple[int, int]:
        """
        Return a 2D integer displacement for the direction.
        The returned object is a NumPy array to simplify vector arithmetic.
        """
        match self:
            case Direction.RIGHT:
                return np.array([0, 1])
            case Direction.UP:
                return np.array([-1, 0])
            case Direction.LEFT:
                return np.array([0, -1])
            case Direction.DOWN:
                return np.array([1, 0])



@dataclass(frozen=True)
class SokobanMap:
    height: int
    width: int
    grid: List[List[Tile]]
    crates: List[Position]
    sokoban: Position


class ParserError(ValueError):
    """
    Raised when map parsing encounters invalid or inconsistent state.
    """

class Parser:
    """
    Parse a sokoban map in text representation into a SokobanMap object.
    """

    _WALL_CHAR: str = "#"
    _STORAGE_CHARS: List[str] = ["X", "c", "s"]
    _CRATE_CHARS: List[str] = ["C", "c"]
    _SOKOBAN_CHARS: List[str] = ["S", "s"]

    def __init__(self, lines: Sequence[str]):
        """
        Construct a Parser from a sequence of string lines
        representing map's rows.

        Use `from_file()` to parse directly from file.
        """
        self._lines: List[str] = [line.rstrip("\n") for line in lines]
        self._height: int = len(self._lines)
        self._width: int = max((len(line) for line in self._lines), default=0)

    @classmethod
    def from_file(cls, path: PathLike) -> SokobanMap:
        """
        Read the file, parse it and return a SokobanMap.
        """
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Map file not found: {path!s}")

        with p.open("r") as file:
            lines = [line.rstrip("\n") for line in file]

        parser = cls(lines)
        return parser.parse()

    def parse(self) -> SokobanMap:
        grid: List[List[Tile]] = []
        crates: List[Position] = []
        sokoban: Optional[Position] = None

        for i, line in enumerate(self._lines):
            row: List[Tile] = []
            for j in range(self._width):
                ch = line[j] if j < len(line) else " "
                if ch == self._WALL_CHAR:
                    tile = Tile.WALL
                elif ch in self._STORAGE_CHARS:
                    tile = Tile.STORAGE
                else:
                    tile = Tile.FLOOR

                if ch in self._CRATE_CHARS:
                    crates.append((i, j))
                if ch in self._SOKOBAN_CHARS:
                    if sokoban is not None:
                        # Map with multiple sokoban positions is an invalid map.
                        raise ParserError(f"Map contains more than one sokoban player position (previous at {sokoban}, additional at {(i, j)})")
                    sokoban = (i, j)

                row.append(tile)
            grid.append(row)
        
        # Map with no sokoban positions is an invalid map.
        if sokoban is None:
            raise ParserError("Map does not contain a sokoban player start position.")

        return SokobanMap(width=self._width, height=self._height, grid=grid, crates=crates, sokoban=sokoban)



class LogicProgram():
    """
    Generate a planning logic program (CNF) for a Sokoban map.

    The encoding creates boolean variables for:
    - static facts: `wall(i,j)` and `storage(i,j)`,
    - fluents per step: `crate(i,j,t)` and `sokoban(i,j,t)`,
    - actions per step: `move(i,j,dir,t)`, `push(i,j,dir,t)`, and `noop(t)`.

    The class has methods to write DIMACS and a human-readable CNF dump.
    """
    def __init__(self, sokoban_map: SokobanMap, max_steps: int = 50):
        """
        Initialize the logic program encoding for the given Sokoban map
        and for the given maximum steps.

        To set the goal step, use `set_goal()` after inicialization.
        """
        self._sokoban_map: SokobanMap = sokoban_map
        self._max_steps: int = max_steps

        # variable name -> integer index for DIMACS vars
        self._vars_map: Optional[List[str]] = []
        # caches for variable lookups to avoid creating duplicates
        self._wall_vars: Optional[Dict[Tuple[int, int], int]] = {}
        self._storage_vars: Optional[Dict[Tuple[int, int], int]] = {}
        self._crate_vars: Optional[Dict[Tuple[int, int, int], int]] = {}
        self._sokoban_vars: Optional[Dict[Tuple[int, int, int], int]] = {}
        self._move_vars: Optional[Dict[Tuple[int, int, Direction, int], int]] = {}
        self._push_vars: Optional[Dict[Tuple[int, int, Direction, int], int]] = {}
        self._noop_vars: Optional[Dict[int, int]] = {}

        # Initial state and clause store. Clauses are grouped per step for
        # convenience when writing out step-bounded encodings and to avoid
        # generating clauses repeatedly.
        self._init_state: Optional[List[int]] = []
        self._clauses: Optional[List[List[List[int]]]] = [[] for _ in range(self._max_steps)]

        # Build the encoding.
        self.__add_init()
        self.__add_move_effects()
        self.__add_push_effects()
        self.__add_one_action_per_step()
        self.__add_frame_axioms()

        # goal step to be set before saving DIMACS
        self._goal_state: Optional[List[int]] = None

    @classmethod
    def __wall(cls, i: int, j: int) -> str:
        return f"wall({i},{j})"
    
    @classmethod
    def __storage(cls, i: int, j: int) -> str:
        return f"storage({i},{j})"
    
    @classmethod
    def __crate(cls, i: int, j: int, step: int) -> str:
        return f"crate({i},{j},{step})"
    
    @classmethod
    def __sokoban(cls, i: int, j: int, step: int) -> str:
        return f"sokoban({i},{j},{step})"
    
    @classmethod
    def __move(cls, i: int, j: int, direction: Direction, step: int) -> str:
        return f"move({i},{j},{direction.to_string()},{step})"
    
    @classmethod
    def __push(cls, i: int, j: int, direction: Direction, step: int) -> str:
        return f"push({i},{j},{direction.to_string()},{step})"
    
    @classmethod
    def __noop(cls, step: int) -> str:
        return f"noop({step})"
    
    def __new_variable(self, name: str) -> int:
        # Append and return variable index for DIMACS (1-based).
        self._vars_map.append(name)
        return len(self._vars_map)

    def __wall_var(self, i: int, j: int) -> int:
        if (i, j) not in self._wall_vars:
            self._wall_vars[(i, j)] = self.__new_variable(LogicProgram.__wall(i, j))
        return self._wall_vars[(i, j)]

    def __storage_var(self, i: int, j: int) -> int:
        if (i, j) not in self._storage_vars:
            self._storage_vars[(i, j)] = self.__new_variable(LogicProgram.__storage(i, j))
        return self._storage_vars[(i, j)]
    
    def __crate_var(self, i: int, j: int, step: int) -> int:
        if (i, j, step) not in self._crate_vars:
            self._crate_vars[(i, j, step)] = self.__new_variable(LogicProgram.__crate(i, j, step))
        return self._crate_vars[(i, j, step)]
    
    def __sokoban_var(self, i: int, j: int, step: int) -> int:
        if (i, j, step) not in self._sokoban_vars:
            self._sokoban_vars[(i, j, step)] = self.__new_variable(LogicProgram.__sokoban(i, j, step))
        return self._sokoban_vars[(i, j, step)]
    
    def __move_var(self, i: int, j: int, direction: Direction, step: int) -> int:
        if (i, j, direction, step) not in self._move_vars:
            self._move_vars[(i, j, direction, step)] = self.__new_variable(LogicProgram.__move(i, j, direction, step))
        return self._move_vars[(i, j, direction, step)]
    
    def __push_var(self, i: int, j: int, direction: Direction, step: int) -> int:
        if (i, j, direction, step) not in self._push_vars:
            self._push_vars[(i, j, direction, step)] = self.__new_variable(LogicProgram.__push(i, j, direction, step))
        return self._push_vars[(i, j, direction, step)]
    
    def __noop_var(self, step: int) -> int:
        if step not in self._noop_vars:
            self._noop_vars[step] = self.__new_variable(LogicProgram.__noop(step))
        return self._noop_vars[step]
    
    def __valid_coords(self, i: int, j: int) -> bool:
        return 0 <= i < self._sokoban_map.height and 0 <= j < self._sokoban_map.width
    
    def __add_init(self) -> None:
        # Add initial state of sokoban map
        for i in range(self._sokoban_map.height):
            for j in range(self._sokoban_map.width):
                w = self.__wall_var(i, j)
                st = self.__storage_var(i, j)
                c = self.__crate_var(i, j, 0)
                so = self.__sokoban_var(i, j, 0)

                # Static facts: whether a cell is a floor/wall/storage.
                match self._sokoban_map.grid[i][j]:
                    case Tile.FLOOR:
                        self._init_state.append(-w)
                        self._init_state.append(-st)
                    case Tile.WALL:
                        self._init_state.append(w)
                        self._init_state.append(-st)
                    case Tile.STORAGE:
                        self._init_state.append(-w)
                        self._init_state.append(st)

                # Crate fluents in step 0
                if (i, j) in self._sokoban_map.crates:
                    self._init_state.append(c)
                else:
                    self._init_state.append(-c)

                # Sokoban fluents in step 0
                if self._sokoban_map.sokoban == (i, j):
                    self._init_state.append(so)
                else:
                    self._init_state.append(-so)
    
    def __add_move_effects(self) -> None:
        # Add preconditions and effects of move actions
        # (sokoban moves without pushing any crate)
        for step in range(self._max_steps):
            for i in range(self._sokoban_map.height):
                for j in range(self._sokoban_map.width):
                    for direction in Direction:
                        move = self.__move_var(i, j, direction, step)
                        target = np.add((i, j), direction.to_vector())

                        # Forbid moves that would send sokoban out of map
                        if not self.__valid_coords(*target):
                            self._clauses[step].append([-move])
                            continue
                        
                        # Preconditions: sokoban must be at (i,j) and target cell
                        # must not be a wall or a crate.
                        self._clauses[step].append([-move, self.__sokoban_var(i, j, step)])
                        self._clauses[step].append([-move, -self.__wall_var(*target)])
                        self._clauses[step].append([-move, -self.__crate_var(*target, step)])
                        
                        # Effects: sokoban moves to target and leaves old position.
                        self._clauses[step].append([-move, self.__sokoban_var(*target, step + 1)])
                        self._clauses[step].append([-move, -self.__sokoban_var(i, j, step + 1)])
    
    def __add_push_effects(self) -> None:
        # Add preconditions and effects of push actions
        # (sokoban moves together with pushing crate)
        for step in range(self._max_steps):
            for i in range(self._sokoban_map.height):
                for j in range(self._sokoban_map.width):
                    for direction in Direction:
                        push = self.__push_var(i, j, direction, step)
                        crate_pos = np.add((i, j), direction.to_vector())
                        target = np.add(crate_pos, direction.to_vector())

                        # Forbid pushes that would send crate out of map
                        if not self.__valid_coords(*target):
                            self._clauses[step].append([-push])
                            continue

                        # Preconditions: sokoban at (i,j), crate at crate_pos, and
                        # target cell free of walls and other crates.
                        self._clauses[step].append([-push, self.__sokoban_var(i, j, step)])
                        self._clauses[step].append([-push, self.__crate_var(*crate_pos, step)])
                        self._clauses[step].append([-push, -self.__wall_var(*target)])
                        self._clauses[step].append([-push, -self.__crate_var(*target, step)])

                        # Effects: sokoban steps into crate's old position and the
                        # crate moves to the target cell.
                        self._clauses[step].append([-push, self.__sokoban_var(*crate_pos, step + 1)])
                        self._clauses[step].append([-push, -self.__sokoban_var(i, j, step + 1)])
                        self._clauses[step].append([-push, self.__crate_var(*target, step + 1)])
                        self._clauses[step].append([-push, -self.__crate_var(*crate_pos, step + 1)])

    def __add_one_action_per_step(self) -> None:
        # Add clauses ensuring exactly one action per step
        for step in range(self._max_steps):
            # At least one action is chosen
            action_vars: List[int] = []
            for i in range(self._sokoban_map.height):
                for j in range(self._sokoban_map.width):
                    for direction in Direction:
                        action_vars.append(self.__move_var(i, j, direction, step))
                        action_vars.append(self.__push_var(i, j, direction, step))
            action_vars.append(self.__noop_var(step))
            
            # No more than one action is chosen
            self._clauses[step].append(action_vars)
            for idx1 in range(len(action_vars)):
                for idx2 in range(idx1 + 1, len(action_vars)):
                    self._clauses[step].append([-action_vars[idx1], -action_vars[idx2]])
    
    def __add_frame_axioms(self) -> None:
        # Add explanatory frame problem axioms
        for step in range(self._max_steps):
            for i in range(self._sokoban_map.height):
                for j in range(self._sokoban_map.width):
                    # Sokoban moved from cell (i,j)
                    clause = [-self.__sokoban_var(i, j, step), self.__sokoban_var(i, j, step + 1)]
                    for direction in Direction:
                        target = np.add((i, j), direction.to_vector())
                        # Skipping invalid moves/pushes
                        if not self.__valid_coords(*target):
                            continue
                        clause.append(self.__move_var(i, j, direction, step))
                        clause.append(self.__push_var(i, j, direction, step))
                    self._clauses[step].append(clause)
                    
                    # Sokoban moved to cell (i,j)
                    clause = [self.__sokoban_var(i, j, step), -self.__sokoban_var(i, j, step + 1)]
                    for direction in Direction:
                        source = np.subtract((i, j), direction.to_vector())
                        # Skipping invalid moves/pushes
                        if not self.__valid_coords(*source):
                            continue
                        clause.append(self.__move_var(*source, direction, step))
                        clause.append(self.__push_var(*source, direction, step))
                    self._clauses[step].append(clause)

                    # Crate was moved from cell (i,j)
                    clause = [-self.__crate_var(i, j, step), self.__crate_var(i, j, step + 1)]
                    for direction in Direction:
                        source = np.subtract((i, j), direction.to_vector())
                        target = np.add((i, j), direction.to_vector())
                        # Skipping invalid pushes
                        if not self.__valid_coords(*source) or not self.__valid_coords(*target):
                            continue
                        clause.append(self.__push_var(*source, direction, step))
                    self._clauses[step].append(clause)
                    
                    # Crate was moved to cell (i,j)
                    clause = [self.__crate_var(i, j, step), -self.__crate_var(i, j, step + 1)]
                    for direction in Direction:
                        source = np.subtract((i, j), 2 * direction.to_vector())
                        # Skipping invalid pushes
                        if not self.__valid_coords(*source):
                            continue
                        clause.append(self.__push_var(*source, direction, step))
                    self._clauses[step].append(clause)
    
    def get_max_steps(self) -> int:
        """
        Return the maximum number of steps provided during class construction.
        """
        return self._max_steps

    def lit2str(self, lit: int) -> str:
        """
        Convert a signed literal back to its textual variable name.
        """
        if abs(lit) > len(self._vars_map):
            raise ValueError(f"Literal {lit} is not a valid variable index.")

        return f"{self._vars_map[abs(lit) - 1]}" if lit > 0 else f"-{self._vars_map[abs(lit) - 1]}"

    def set_goal(self, steps: int) -> LogicProgram:
        """
        Set the goal step - that is, at which step should be all
        crates already placed in storage cells. This results in a
        satisfiable or unsatisfiable program.
        """

        if steps > self._max_steps:
            raise ValueError(f"Requested steps {steps} exceed maximum configured steps {self._max_steps}.")
        
        self._goal_step = steps
        self._goal_state = []

        for i in range(self._sokoban_map.height):
            for j in range(self._sokoban_map.width):
                if self._sokoban_map.grid[i][j] == Tile.STORAGE:
                    self._goal_state.append(self.__crate_var(i, j, steps))
        
        return self
    
    def save_dimacs(self, path: PathLike) -> LogicProgram:      
        """
        Write the current encoding to `path` in DIMACS format as a MiniSat solver input.
        
        If the goal step is not set yet, the encoding will contain clauses for
        each step up to the maximum steps specified during construction.
        """

        num_vars = len(self._vars_map)
        num_clauses = len(self._init_state) + len(self._goal_state) + sum(len(self._clauses[s]) for s in range(self._goal_step))

        p = Path(path)
        with p.open("w") as file:
            file.write(f"p cnf {num_vars} {num_clauses}\n")

            for lit in self._init_state:
                file.write(f"{lit} 0\n")

            for s in range(self._goal_step if self._goal_step is not None else self._max_steps):
                for clause in self._clauses[s]:
                    file.write(" ".join(str(lit) for lit in clause) + " 0\n")

            if self._goal_state is not None:
                for lit in self._goal_state:
                    file.write(f"{lit} 0\n")
        
        return self
    
    def save_cnf_readable(self, path: PathLike) -> LogicProgram:
        """
        Write a human-readable dump of the CNF to `path`.
        
        If the goal step is not set yet, the dump will contain clauses for
        each step up to the maximum steps specified during construction.
        """

        p = Path(path)
        with p.open("w") as file:
            file.write("Initial State:\n")
            for lit in self._init_state:
                file.write(f"  {self.lit2str(lit)}\n")

            file.write("\nClauses per step:\n")
            for s in range(self._goal_step if self._goal_step is not None else self._max_steps):
                file.write(f"\n  Step {s}:\n")
                for clause in self._clauses[s]:
                    file.write("    " + " v ".join(self.lit2str(lit) for lit in clause) + "\n")
            
            if self._goal_state is not None:
                file.write("\nGoal State:\n")
                for lit in self._goal_state:
                    file.write(f"  {self.lit2str(lit)}\n")
        
        return self


class Planner():
    """
    Finds the shortest solution for the given sokoban map.

    Wrapper around `LogicProgram` class, implementing binary search.
    """
    def __init__(self, sokoban_map: SokobanMap, max_steps: int, solver_path: PathLike, solver_input: PathLike, solver_output: PathLike):
        if solver_path is None or not os.path.isfile(solver_path):
            raise FileNotFoundError(f"MiniSat binary not found at path: {solver_path!s}")
        
        if solver_input is None or os.path.exists(solver_input) and not os.access(solver_input, os.W_OK):
            raise FileNotFoundError(f"Cannot write to solver input file: {solver_input!s}")
        
        if solver_output is None or os.path.exists(solver_output) and not os.access(solver_output, os.W_OK):
            raise FileNotFoundError(f"Cannot write to solver output file: {solver_output!s}")

        self._logic_program: LogicProgram = LogicProgram(sokoban_map, max_steps)
        self._solver_path: PathLike = solver_path
        self._solver_input: PathLike = solver_input
        self._solver_output: PathLike = solver_output

    def __parse_solution(self) -> List[str]:
        solution: List[str] = []
        with open(self._solver_output, "r") as file:
            for i, line in enumerate(file):
                # MiniSat prints the model as the second line (index 1)
                if i == 1:
                    literals = line.strip().split()
                    for lit in [int(x) for x in literals]:
                        if lit > 0:
                            var_name = self._logic_program.lit2str(lit)
                            if var_name.startswith(("move", "push", "noop")):
                                solution.append(var_name)

        def extract_step(action: str) -> int:
            # extract final numeric argument which is the step index
            if action.startswith("noop"):
                return int(action[action.index("(") + 1 : action.index(")")])
            else:
                return int(action[action.rindex(",") + 1 : action.index(")")])

        # return sorted solution by steps (sequence of actions in order)
        return sorted(solution, key=extract_step)

    def find_solution(self) -> Optional[List[str]]:
        """
        Search for a shortest-step plan by calling the MiniSat solver and return ordered actions if found.
        """
        least_steps = -1
        left, right = 1, self._logic_program.get_max_steps()

        # Binary search (possible because of noop action)
        while left <= right:
            mid = (left + right) // 2
            self._logic_program.set_goal(mid).save_dimacs(self._solver_input)
            
            result = subprocess.run(
                [self._solver_path, self._solver_input, self._solver_output],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 10:
                least_steps = mid
                right = mid - 1
            else:
                left = mid + 1
        
        if least_steps == -1:
            return None
        
        # Run MiniSat again for the shortest plan to extract the sequence of actions
        self._logic_program.set_goal(least_steps).save_dimacs(self._solver_input)
        result = subprocess.run(
            [self._solver_path, self._solver_input, self._solver_output],
            capture_output=True,
            text=True
        )

        return self.__parse_solution()
    
    def save_readable_cnf(self, path: PathLike) -> Planner:
        """
        Save a human-readable CNF of the current underlying logic program.
        """
        self._logic_program.save_cnf_readable(path)
        return self



ARGPARSER = argparse.ArgumentParser(
    prog="python sokoban.py",
    description="Solver for Sokoban game using SAT planning.")    

ARGPARSER.add_argument("mapfile", type=str, help="Path to the Sokoban map file.")
ARGPARSER.add_argument("minisat", type=str, help="Path to the MiniSat executable.")
ARGPARSER.add_argument("-s", "--maxsteps", type=int, default=50, help="Maximum number of steps to search for a solution (default 50).")
ARGPARSER.add_argument("-i", "--input", type=str, default="solver_input.cnf", help="File where the input for MiniSat will be stored. The file will be created/overwritten, so it need not exist beforehand (default 'solver_input.cnf').")
ARGPARSER.add_argument("-o", "--output", type=str, default="solver_output.cnf", help="File where the output from MiniSat will be stored. The file will be created/overwritten, so it need not exist beforehand (default 'solver_output.cnf').")
ARGPARSER.add_argument("-r", "--readablecnf", type=str, default=None, help="If provided, a human-readable version of the generated CNF will be saved to this file.")



if __name__ == "__main__":
    args = ARGPARSER.parse_args()

    try:
        sokoban_map = Parser.from_file(args.mapfile)
    except (FileNotFoundError, ParserError) as e:
        print(f"Error: {e}")
        exit(1)

    try:
        planner = Planner(
            sokoban_map=sokoban_map,
            max_steps=args.maxsteps,
            solver_path=args.minisat,
            solver_input=args.input,
            solver_output=args.output
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)

    if args.readablecnf is not None:
        planner.save_readable_cnf(args.readablecnf)

    solution = planner.find_solution()

    if solution is None:
        print("No solution found within the given step limit.")
    else:
        print("Solution found:")
        for action in solution:
            print(action)
