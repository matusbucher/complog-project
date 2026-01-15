from __future__ import annotations
from pathlib import Path
import numpy as np
from typing import List, Dict, Tuple, Optional, Union

from logic.logic_program_interface import Action, Direction, LogicProgramInterface, Solution, SolutionStep
from logic.sokoban_map import MapObject, SokobanMap, Ground


PathLike = Union[str, Path]


class LogicProgramBasic(LogicProgramInterface):
    """
    Generate a basic planning logic program (CNF) for a Sokoban map.

    The encoding creates boolean variables for:
    - static facts: `wall(x,y)` and `storage(x,y)`,
    - fluents per step: `crate(x,y,t)` and `sokoban(x,y,t)`,
    - actions per step: `move(x,y,dir,t)`, `push(x,y,dir,t)`, and `noop(t)`.
    """

    def __init__(self, sokoban_map: SokobanMap, max_steps: int = 50):
        """
        Initialize the logic program encoding for the given Sokoban map and for the given maximum steps.

        To set the goal step, use `set_goal()` after inicialization.
        """
        self._sokoban_map: SokobanMap = sokoban_map
        self._max_steps: int = max_steps

        # Mappings from integer literals to variables
        self._vars_map: Optional[List[str]] = []
        self._action_map: Optional[Dict[int, SolutionStep]] = {}

        # Caches for variable lookups to avoid creating duplicates
        self._wall_vars: Optional[Dict[Tuple[int, int], int]] = {}
        self._storage_vars: Optional[Dict[Tuple[int, int], int]] = {}
        self._crate_vars: Optional[Dict[Tuple[int, int, int], int]] = {}
        self._sokoban_vars: Optional[Dict[Tuple[int, int, int], int]] = {}
        self._move_vars: Optional[Dict[Tuple[int, int, Direction, int], int]] = {}
        self._push_vars: Optional[Dict[Tuple[int, int, Direction, int], int]] = {}
        self._noop_vars: Optional[Dict[int, int]] = {}

        # Initial state and clause store. Clauses are grouped per step for convenience when
        # writing out step-bounded encodings and to avoid generating clauses repeatedly.
        self._init_state: Optional[List[int]] = []
        self._clauses: Optional[List[List[List[int]]]] = [[] for _ in range(self._max_steps)]

        # Build the encoding.
        self.__add_init()
        self.__add_move_effects()
        self.__add_push_effects()
        self.__add_one_action_per_step()
        self.__add_frame_axioms()

        # Goal step to be set before saving DIMACS
        self._goal_state: Optional[List[int]] = None

    def set_goal(self, goal_step: int) -> LogicProgramBasic:
        if goal_step > self._max_steps:
            raise ValueError(f"Requested steps {goal_step} exceed maximum configured steps {self._max_steps}.")
        
        self._goal_step = goal_step
        self._goal_state = []

        for x in range(self._sokoban_map.height):
            for y in range(self._sokoban_map.width):
                if self._sokoban_map.grid[x][y].ground == Ground.STORAGE:
                    self._goal_state.append(self.__crate_var(x, y, goal_step))
        return self
    
    def save_dimacs(self, path: PathLike) -> LogicProgramBasic:      
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
    
    def save_cnf_readable(self, path: PathLike) -> LogicProgramBasic:
        p = Path(path)
        with p.open("w") as file:
            file.write("Initial State:\n")
            for lit in self._init_state:
                file.write(f"  {self.lit_to_str(lit)}\n")

            file.write("\nClauses per step:\n")
            for s in range(self._goal_step if self._goal_step is not None else self._max_steps):
                file.write(f"\n  Step {s}:\n")
                for clause in self._clauses[s]:
                    file.write("    " + " v ".join(self.lit_to_str(lit) for lit in clause) + "\n")
            
            if self._goal_state is not None:
                file.write("\nGoal State:\n")
                for lit in self._goal_state:
                    file.write(f"  {self.lit_to_str(lit)}\n")   
        return self
    
    def get_max_steps(self) -> int:
        return self._max_steps
    
    def lit_to_str(self, literal: int) -> str:
        index = abs(literal) - 1
        if index >= len(self._vars_map):
            raise ValueError(f"Literal {literal} is out of bounds for variable map of size {len(self._vars_map)}.")
        var_str = self._vars_map[index]
        return var_str if literal > 0 else f"-{var_str}"
    
    def extract_solution(self, literals: List[int]) -> Solution:
        solution : Solution = []
        for lit in literals:
            if lit > 0 and lit in self._action_map:
                step = self._action_map[lit]
                if step.action != Action.NOOP:
                    solution.append(step)

        solution.sort(key=lambda x: x.step)
        for i in range(len(solution)):
            solution[i].step = i + 1
        return solution

    @classmethod
    def __wall(cls, x: int, y: int) -> str:
        return f"wall({x},{y})"
    
    @classmethod
    def __storage(cls, x: int, y: int) -> str:
        return f"storage({x},{y})"
    
    @classmethod
    def __crate(cls, x: int, y: int, step: int) -> str:
        return f"crate({x},{y},{step})"
    
    @classmethod
    def __sokoban(cls, x: int, y: int, step: int) -> str:
        return f"sokoban({x},{y},{step})"
    
    @classmethod
    def __move(cls, x: int, y: int, direction: Direction, step: int) -> str:
        return f"move({x},{y},{direction.to_string()},{step})"
    
    @classmethod
    def __push(cls, x: int, y: int, direction: Direction, step: int) -> str:
        return f"push({x},{y},{direction.to_string()},{step})"
    
    @classmethod
    def __noop(cls, step: int) -> str:
        return f"noop({step})"
    
    def __new_variable(self, name: str) -> int:
        # Append and return variable index for DIMACS
        self._vars_map.append(name)
        return len(self._vars_map)

    def __wall_var(self, x: int, y: int) -> int:
        if (x, y) not in self._wall_vars:
            self._wall_vars[(x, y)] = self.__new_variable(LogicProgramBasic.__wall(x, y))
        return self._wall_vars[(x, y)]

    def __storage_var(self, x: int, y: int) -> int:
        if (x, y) not in self._storage_vars:
            self._storage_vars[(x, y)] = self.__new_variable(LogicProgramBasic.__storage(x, y))
        return self._storage_vars[(x, y)]
    
    def __crate_var(self, x: int, y: int, step: int) -> int:
        if (x, y, step) not in self._crate_vars:
            self._crate_vars[(x, y, step)] = self.__new_variable(LogicProgramBasic.__crate(x, y, step))
        return self._crate_vars[(x, y, step)]
    
    def __sokoban_var(self, x: int, y: int, step: int) -> int:
        if (x, y, step) not in self._sokoban_vars:
            self._sokoban_vars[(x, y, step)] = self.__new_variable(LogicProgramBasic.__sokoban(x, y, step))
        return self._sokoban_vars[(x, y, step)]
    
    def __move_var(self, x: int, y: int, direction: Direction, step: int) -> int:
        if (x, y, direction, step) not in self._move_vars:
            lit = self.__new_variable(LogicProgramBasic.__move(x, y, direction, step))
            self._move_vars[(x, y, direction, step)] = lit
            self._action_map[lit] = SolutionStep(Action.MOVE, x, y, direction, step)
        return self._move_vars[(x, y, direction, step)]
    
    def __push_var(self, x: int, y: int, direction: Direction, step: int) -> int:
        if (x, y, direction, step) not in self._push_vars:
            lit = self.__new_variable(LogicProgramBasic.__push(x, y, direction, step))
            self._push_vars[(x, y, direction, step)] = lit
            self._action_map[lit] = SolutionStep(Action.PUSH, x, y, direction, step)
        return self._push_vars[(x, y, direction, step)]
    
    def __noop_var(self, step: int) -> int:
        if step not in self._noop_vars:
            lit = self.__new_variable(LogicProgramBasic.__noop(step))
            self._noop_vars[step] = lit
            self._action_map[lit] = SolutionStep(Action.NOOP, 0, 0, Direction.UP, step)
        return self._noop_vars[step]
    
    def __valid_coords(self, x: int, y: int) -> bool:
        return 0 <= x < self._sokoban_map.height and 0 <= y < self._sokoban_map.width
    
    def __add_init(self) -> None:
        # Add initial state of sokoban map
        for x in range(self._sokoban_map.height):
            for y in range(self._sokoban_map.width):
                w = self.__wall_var(x, y)
                st = self.__storage_var(x, y)
                c = self.__crate_var(x, y, 0)
                so = self.__sokoban_var(x, y, 0)

                # Static facts: whether a cell is a floor/wall/storage
                match self._sokoban_map.grid[x][y].ground:
                    case Ground.FLOOR:
                        self._init_state.append(-w)
                        self._init_state.append(-st)
                    case Ground.WALL:
                        self._init_state.append(w)
                        self._init_state.append(-st)
                    case Ground.STORAGE:
                        self._init_state.append(-w)
                        self._init_state.append(st)
                
                # Dynamic facts: initial positions of crates and sokoban
                if self._sokoban_map.grid[x][y].map_object is not None:
                    if self._sokoban_map.grid[x][y].map_object == MapObject.CRATE:
                        self._init_state.append(c)
                        self._init_state.append(-so)
                    elif self._sokoban_map.grid[x][y].map_object == MapObject.SOKOBAN:
                        self._init_state.append(-c)
                        self._init_state.append(so)
                else:
                    self._init_state.append(-c)
                    self._init_state.append(-so)
    
    def __add_move_effects(self) -> None:
        # Add preconditions and effects of move actions (without pushing any crate)
        for step in range(self._max_steps):
            for x in range(self._sokoban_map.height):
                for y in range(self._sokoban_map.width):
                    for direction in Direction:
                        move = self.__move_var(x, y, direction, step)
                        target = np.add((x, y), direction.to_vector())

                        # Forbid moves that would send sokoban out of map
                        if not self.__valid_coords(*target):
                            self._clauses[step].append([-move])
                            continue
                        
                        # Preconditions: sokoban must be at (x,y) and target cell must not be a wall or a crate
                        self._clauses[step].append([-move, self.__sokoban_var(x, y, step)])
                        self._clauses[step].append([-move, -self.__wall_var(*target)])
                        self._clauses[step].append([-move, -self.__crate_var(*target, step)])
                        
                        # Effects: sokoban moves to target and leaves old position
                        self._clauses[step].append([-move, self.__sokoban_var(*target, step + 1)])
                        self._clauses[step].append([-move, -self.__sokoban_var(x, y, step + 1)])
    
    def __add_push_effects(self) -> None:
        # Add preconditions and effects of push actions (sokoban moves together with pushing crate)
        for step in range(self._max_steps):
            for x in range(self._sokoban_map.height):
                for y in range(self._sokoban_map.width):
                    for direction in Direction:
                        push = self.__push_var(x, y, direction, step)
                        crate_pos = np.add((x, y), direction.to_vector())
                        target = np.add(crate_pos, direction.to_vector())

                        # Forbid pushes that would send crate out of map
                        if not self.__valid_coords(*target):
                            self._clauses[step].append([-push])
                            continue

                        # Preconditions: sokoban at (x,y), crate at crate_pos, and target cell free of walls and other crates
                        self._clauses[step].append([-push, self.__sokoban_var(x, y, step)])
                        self._clauses[step].append([-push, self.__crate_var(*crate_pos, step)])
                        self._clauses[step].append([-push, -self.__wall_var(*target)])
                        self._clauses[step].append([-push, -self.__crate_var(*target, step)])

                        # Effects: sokoban steps into crate's old position and the crate moves to the target cell
                        self._clauses[step].append([-push, self.__sokoban_var(*crate_pos, step + 1)])
                        self._clauses[step].append([-push, -self.__sokoban_var(x, y, step + 1)])
                        self._clauses[step].append([-push, self.__crate_var(*target, step + 1)])
                        self._clauses[step].append([-push, -self.__crate_var(*crate_pos, step + 1)])

    def __add_one_action_per_step(self) -> None:
        # Add clauses ensuring exactly one action per step
        for step in range(self._max_steps):
            # At least one action
            action_vars: List[int] = []
            for x in range(self._sokoban_map.height):
                for y in range(self._sokoban_map.width):
                    for direction in Direction:
                        action_vars.append(self.__move_var(x, y, direction, step))
                        action_vars.append(self.__push_var(x, y, direction, step))
            action_vars.append(self.__noop_var(step))
            
            # At most one action
            self._clauses[step].append(action_vars)
            for i1 in range(len(action_vars)):
                for i2 in range(i1 + 1, len(action_vars)):
                    self._clauses[step].append([-action_vars[i1], -action_vars[i2]])
    
    def __add_frame_axioms(self) -> None:
        # Add explanatory frame problem axioms
        for step in range(self._max_steps):
            for x in range(self._sokoban_map.height):
                for y in range(self._sokoban_map.width):
                    # Sokoban moved from cell (x,y)
                    clause = [-self.__sokoban_var(x, y, step), self.__sokoban_var(x, y, step + 1)]
                    for direction in Direction:
                        clause.append(self.__move_var(x, y, direction, step))
                        clause.append(self.__push_var(x, y, direction, step))
                    self._clauses[step].append(clause)
                    
                    # Sokoban moved to cell (x,y)
                    clause = [self.__sokoban_var(x, y, step), -self.__sokoban_var(x, y, step + 1)]
                    for direction in Direction:
                        source = np.subtract((x, y), direction.to_vector())
                        if not self.__valid_coords(*source):
                            continue
                        clause.append(self.__move_var(*source, direction, step))
                        clause.append(self.__push_var(*source, direction, step))
                    self._clauses[step].append(clause)

                    # Crate was moved from cell (x,y)
                    clause = [-self.__crate_var(x, y, step), self.__crate_var(x, y, step + 1)]
                    for direction in Direction:
                        source = np.subtract((x, y), direction.to_vector())
                        if not self.__valid_coords(*source):
                            continue
                        clause.append(self.__push_var(*source, direction, step))
                    self._clauses[step].append(clause)
                    
                    # Crate was moved to cell (x,y)
                    clause = [self.__crate_var(x, y, step), -self.__crate_var(x, y, step + 1)]
                    for direction in Direction:
                        source = np.subtract((x, y), 2 * direction.to_vector())
                        if not self.__valid_coords(*source):
                            continue
                        clause.append(self.__push_var(*source, direction, step))
                    self._clauses[step].append(clause)