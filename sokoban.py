from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import subprocess
import argparse
import os



Position = Tuple[int, int]
PathLike = Union[str, Path]


class Tile(Enum):
    WALL = "#"
    STORAGE = "X"
    FLOOR = " "

    @classmethod
    def from_char(cls, ch: str) -> "Tile":
        """Return Tile enum for a character. Unknown characters map to FLOOR."""
        if ch == "#":
            return cls.WALL
        if ch == "X":
            return cls.STORAGE
        return cls.FLOOR



class Direction(Enum):
    RIGHT = 0
    UP = 1
    LEFT = 2
    DOWN = 3

    def to_string(self) -> str:
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
    crates: Set[Position]
    sokoban: Optional[Position]


class ParserError(ValueError):
    """Raised when map parsing encounters invalid or inconsistent state."""

class Parser:
    """
    Parse a sokoban map in text representation into a SokobanMap object.
    """

    _CRATE_CHARS: Set[str] = {"C", "c"}
    _SOKOBAN_CHARS: Set[str] = {"S", "s"}

    def __init__(self, lines: Sequence[str]):
        """
        Construct a Parser from a sequence of string lines.
        Use from_file() to parse directly from file.
        """
        self._lines: List[str] = [line.rstrip("\n") for line in lines]
        self._height: int = len(self._lines)
        self._width: int = max((len(line) for line in self._lines), default=0)

    @classmethod
    def from_file(cls, path: PathLike) -> SokobanMap:
        """
        Read the file, parse it and return a SokobanMap.
        Raises FileNotFoundError if the file does not exist
        and ParserError on parsing problems.
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
                if ch == "#":
                    tile = Tile.WALL
                elif ch in ("X", "c", "s"):
                    tile = Tile.STORAGE
                else:
                    tile = Tile.FLOOR

                if ch in self._CRATE_CHARS:
                    crates.append((i, j))
                if ch in self._SOKOBAN_CHARS:
                    if sokoban is not None:
                        raise ParserError(
                            f"Map contains more than one sokoban player position (previous at {sokoban}, additional at {(i, j)})"
                        )
                    sokoban = (i, j)

                row.append(tile)
            grid.append(row)

        return SokobanMap(width=self._width, height=self._height, grid=grid, crates=crates, sokoban=sokoban)



class LogicProgram():
    '''
    Generate a logic program for the given sokoban map.
    '''
    def __init__(self, sokoban_map: SokobanMap, max_steps: int = 50):
        self._sokoban_map: SokobanMap = sokoban_map
        self._max_steps: int = max_steps
        self._goal_steps: Optional[int] = None

        self._vars_map: Optional[List[str]] = []
        self._wall_vars: Optional[Dict[Tuple[int, int], int]] = {}
        self._storage_vars: Optional[Dict[Tuple[int, int], int]] = {}
        self._crate_vars: Optional[Dict[Tuple[int, int, int], int]] = {}
        self._sokoban_vars: Optional[Dict[Tuple[int, int, int], int]] = {}
        self._move_vars: Optional[Dict[Tuple[int, int, Direction, int], int]] = {}
        self._push_vars: Optional[Dict[Tuple[int, int, Direction, int], int]] = {}
        self._noop_vars: Optional[Dict[int, int]] = {}

        self._init_state: Optional[List[int]] = []
        self._clauses: Optional[List[List[List[int]]]] = [[] for _ in range(self._max_steps)]

        self.__add_init()
        self.__add_init()
        self.__add_move_effects()
        self.__add_push_effects()
        self.__add_one_action_per_step()
        self.__add_frame_axioms()

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
        for i in range(self._sokoban_map.height):
            for j in range(self._sokoban_map.width):
                w = self.__wall_var(i, j)
                st = self.__storage_var(i, j)
                c = self.__crate_var(i, j, 0)
                so = self.__sokoban_var(i, j, 0)

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
                
                if (i, j) in self._sokoban_map.crates:
                    self._init_state.append(c)
                else:
                    self._init_state.append(-c)

                if self._sokoban_map.sokoban == (i, j):
                    self._init_state.append(so)
                else:
                    self._init_state.append(-so)
    
    def __add_move_effects(self) -> None:
        for step in range(self._max_steps):
            for i in range(self._sokoban_map.height):
                for j in range(self._sokoban_map.width):
                    for direction in Direction:
                        move = self.__move_var(i, j, direction, step)
                        target = np.add((i, j), direction.to_vector())

                        if not self.__valid_coords(*target):
                            self._clauses[step].append([-move])
                            continue
                        
                        # Preconditions
                        self._clauses[step].append([-move, self.__sokoban_var(i, j, step)])
                        self._clauses[step].append([-move, -self.__wall_var(*target)])
                        self._clauses[step].append([-move, -self.__crate_var(*target, step)])
                        # Effects
                        self._clauses[step].append([-move, self.__sokoban_var(*target, step + 1)])
                        self._clauses[step].append([-move, -self.__sokoban_var(i, j, step + 1)])
    
    def __add_push_effects(self) -> None:
        for step in range(self._max_steps):
            for i in range(self._sokoban_map.height):
                for j in range(self._sokoban_map.width):
                    for direction in Direction:
                        push = self.__push_var(i, j, direction, step)
                        crate_pos = np.add((i, j), direction.to_vector())
                        target = np.add(crate_pos, direction.to_vector())

                        if not self.__valid_coords(*target):
                            self._clauses[step].append([-push])
                            continue
                        
                        # Preconditions
                        self._clauses[step].append([-push, self.__sokoban_var(i, j, step)])
                        self._clauses[step].append([-push, self.__crate_var(*crate_pos, step)])
                        self._clauses[step].append([-push, -self.__wall_var(*target)])
                        self._clauses[step].append([-push, -self.__crate_var(*target, step)])
                        # Effects
                        self._clauses[step].append([-push, self.__sokoban_var(*crate_pos, step + 1)])
                        self._clauses[step].append([-push, -self.__sokoban_var(i, j, step + 1)])
                        self._clauses[step].append([-push, self.__crate_var(*target, step + 1)])
                        self._clauses[step].append([-push, -self.__crate_var(*crate_pos, step + 1)])

    def __add_one_action_per_step(self) -> None:
        for step in range(self._max_steps):
            action_vars: List[int] = []
            for i in range(self._sokoban_map.height):
                for j in range(self._sokoban_map.width):
                    for direction in Direction:
                        action_vars.append(self.__move_var(i, j, direction, step))
                        action_vars.append(self.__push_var(i, j, direction, step))
            action_vars.append(self.__noop_var(step))
            
            self._clauses[step].append(action_vars)
            for idx1 in range(len(action_vars)):
                for idx2 in range(idx1 + 1, len(action_vars)):
                    self._clauses[step].append([-action_vars[idx1], -action_vars[idx2]])
    
    def __add_frame_axioms(self) -> None:
        for step in range(self._max_steps):
            for i in range(self._sokoban_map.height):
                for j in range(self._sokoban_map.width):
                    # Sokoban frame axioms
                    clause = [-self.__sokoban_var(i, j, step), self.__sokoban_var(i, j, step + 1)]
                    for direction in Direction:
                        target = np.add((i, j), direction.to_vector())
                        if not self.__valid_coords(*target):
                            continue
                        clause.append(self.__move_var(i, j, direction, step))
                        clause.append(self.__push_var(i, j, direction, step))
                    self._clauses[step].append(clause)
                    
                    clause = [self.__sokoban_var(i, j, step), -self.__sokoban_var(i, j, step + 1)]
                    for direction in Direction:
                        source = np.subtract((i, j), direction.to_vector())
                        if not self.__valid_coords(*source):
                            continue
                        clause.append(self.__move_var(*source, direction, step))
                        clause.append(self.__push_var(*source, direction, step))
                    self._clauses[step].append(clause)

                    # Crate frame axioms
                    clause = [-self.__crate_var(i, j, step), self.__crate_var(i, j, step + 1)]
                    for direction in Direction:
                        source = np.subtract((i, j), direction.to_vector())
                        target = np.add((i, j), direction.to_vector())
                        if not self.__valid_coords(*source) or not self.__valid_coords(*target):
                            continue
                        clause.append(self.__push_var(*source, direction, step))
                    self._clauses[step].append(clause)
                    
                    clause = [self.__crate_var(i, j, step), -self.__crate_var(i, j, step + 1)]
                    for direction in Direction:
                        source = np.subtract((i, j), 2 * direction.to_vector())
                        if not self.__valid_coords(*source):
                            continue
                        clause.append(self.__push_var(*source, direction, step))
                    self._clauses[step].append(clause)
    
    def get_max_steps(self) -> int:
        return self._max_steps

    def lit2str(self, lit: int) -> str:
        return f"{self._vars_map[abs(lit) - 1]}" if lit > 0 else f"-{self._vars_map[abs(lit) - 1]}"

    def set_goal(self, steps: int) -> LogicProgram:
        if steps > self._max_steps:
            raise ValueError(f"Requested steps {steps} exceed maximum configured steps {self._max_steps}.")
        
        self._goal_steps = steps
        self._goal_state = []

        for i in range(self._sokoban_map.height):
            for j in range(self._sokoban_map.width):
                if self._sokoban_map.grid[i][j] == Tile.STORAGE:
                    self._goal_state.append(self.__crate_var(i, j, steps))
        
        return self
    
    def save_dimacs(self, path: PathLike) -> LogicProgram:      
        num_vars = len(self._vars_map)
        num_clauses = len(self._init_state) + len(self._goal_state) + sum(len(self._clauses[s]) for s in range(self._goal_steps))

        p = Path(path)
        with p.open("w") as file:
            file.write(f"p cnf {num_vars} {num_clauses}\n")

            for lit in self._init_state:
                file.write(f"{lit} 0\n")

            for s in range(self._goal_steps if self._goal_steps is not None else self._max_steps):
                for clause in self._clauses[s]:
                    file.write(" ".join(str(lit) for lit in clause) + " 0\n")

            if self._goal_state is not None:
                for lit in self._goal_state:
                    file.write(f"{lit} 0\n")
        
        return self
    
    def save_cnf_readable(self, path: PathLike) -> LogicProgram:
        p = Path(path)
        with p.open("w") as file:
            file.write("Initial State:\n")
            for lit in self._init_state:
                file.write(f"  {self.lit2str(lit)}\n")

            file.write("\nClauses per step:\n")
            for s in range(self._goal_steps if self._goal_steps is not None else self._max_steps):
                file.write(f"\n  Step {s}:\n")
                for clause in self._clauses[s]:
                    file.write("    " + " v ".join(self.lit2str(lit) for lit in clause) + "\n")
            
            if self._goal_state is not None:
                file.write("\nGoal State:\n")
                for lit in self._goal_state:
                    file.write(f"  {self.lit2str(lit)}\n")
        
        return self



class PlannerError(ValueError):
    """Raised when planner encounters an error."""

class Planner():
    """
    Finds the shortest solution for the given sokoban map.
    """
    def __init__(self, sokoban_map: SokobanMap, max_steps: int, solver_path: PathLike, solver_input: PathLike, solver_output: PathLike):
        if solver_path is None or not os.path.isfile(solver_path):
            raise PlannerError(f"SAT solver not found at path: {solver_path!s}")
        
        if solver_input is None or os.path.exists(solver_input) and not os.access(solver_input, os.W_OK):
            raise PlannerError(f"Cannot write to solver input file: {solver_input!s}")
        
        if solver_output is None or os.path.exists(solver_output) and not os.access(solver_output, os.W_OK):
            raise PlannerError(f"Cannot write to solver output file: {solver_output!s}")

        self._logic_program: LogicProgram = LogicProgram(sokoban_map, max_steps)
        self._solver_path: PathLike = solver_path
        self._solver_input: PathLike = solver_input
        self._solver_output: PathLike = solver_output

    def __parse_solution(self) -> List[str]:
        """Parse SAT solver output and extract solution."""
        solution: List[str] = []
        with open(self._solver_output, "r") as file:
            for i, line in enumerate(file):
                if i == 1:
                    literals = line.strip().split()
                    for lit in [int(x) for x in literals]:
                        if lit > 0:
                            var_name = self._logic_program.lit2str(lit)
                            if var_name.startswith(("move", "push", "noop")):
                                solution.append(var_name)

        def extract_step(action: str) -> int:
            if action.startswith("noop"):
                return int(action[action.index("(") + 1 : action.index(")")])
            else:
                return int(action[action.rindex(",") + 1 : action.index(")")])
        
        return sorted(solution, key=extract_step)

    def find_solution(self) -> Optional[List[str]]:
        """Find solution using binary search on number of steps."""
        least_steps = -1
        left, right = 1, self._logic_program.get_max_steps()

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
        
        self._logic_program.set_goal(least_steps).save_dimacs(self._solver_input)
        result = subprocess.run(
            [self._solver_path, self._solver_input, self._solver_output],
            capture_output=True,
            text=True
        )

        return self.__parse_solution()
    
    def save_readable_cnf(self, path: PathLike) -> Planner:
        """Save human-readable CNF to the given path."""
        self._logic_program.save_cnf_readable(path)
        return self



ARGPARSER = argparse.ArgumentParser(
    prog="python sokoban.py",
    description="Solver for Sokoban game using SAT planning.")    

ARGPARSER.add_argument("mapfile", type=str, help="Path to the Sokoban map file.")
ARGPARSER.add_argument("minisat", type=str, help="Path to the Minisat executable.")
ARGPARSER.add_argument("-s", "--maxsteps", type=int, default=50, help="Maximum number of steps to search for a solution (default 50).")
ARGPARSER.add_argument("-i", "--input", type=str, default="solver_input.cnf", help="File where the input for Minisat will be stored. The file will be created/overwritten, so it need not exist beforehand (default 'solver_input.cnf').")
ARGPARSER.add_argument("-o", "--output", type=str, default="solver_output.txt", help="File where the output from Minisat will be stored. The file will be created/overwritten, so it need not exist beforehand (default 'solver_output.txt').")
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
    except PlannerError as e:
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
