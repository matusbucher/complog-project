from __future__ import annotations
from pathlib import Path
import numpy as np
from typing import List, Dict, Tuple, Optional, Union

from logic.logic_program_interface import Action, Direction, LogicProgramInterface, Solution, SolutionStep
from logic.sokoban_map import MapObject, SokobanMap, Ground


PathLike = Union[str, Path]


class LogicProgramOptimized(LogicProgramInterface):
    """
    Generate an optimized planning logic program (CNF) for a Sokoban map.

    Implemented optimizations compared to basic encoding:
    - action overriding: only one action variable per step, with arguments as separate variables
    - sokoban position fluents split into X and Y components
    - more effective clauses (because of new variable structure)

    The encoding creates boolean variables for:
    - static facts: `wall(x,y)` and `storage(x,y)`,
    - fluents per step: `crate(x,y,t)`, `sokobanX(x,t)`, and `sokobanY(y,t)`,
    - actions per step (action overriding): `action(a,t)`, `argX(x,t)`, `argY(y,t)`, and `argD(d,t)`.

    The class has methods to write DIMACS and a human-readable CNF dump.
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
        self._action_map: Optional[Dict[int, Tuple[Action, int]]] = {}
        self._argX_map: Optional[Dict[int, Tuple[int, int]]] = {}
        self._argY_map: Optional[Dict[int, Tuple[int, int]]] = {}
        self._argD_map: Optional[Dict[int, Tuple[Direction, int]]] = {}

        # Caches for variable lookups to avoid creating duplicates
        self._wall_vars: Optional[Dict[Tuple[int, int], int]] = {}
        self._storage_vars: Optional[Dict[Tuple[int, int], int]] = {}
        self._crate_vars: Optional[Dict[Tuple[int, int, int], int]] = {}
        self._sokobanX_vars: Optional[Dict[Tuple[int, int], int]] = {}
        self._sokobanY_vars: Optional[Dict[Tuple[int, int], int]] = {}
        self._action_vars: Optional[Dict[Tuple[Action, int], int]] = {}
        self._argX_vars: Optional[Dict[Tuple[int, int], int]] = {}
        self._argY_vars: Optional[Dict[Tuple[int, int], int]] = {}
        self._argD_vars: Optional[Dict[Tuple[Direction, int], int]] = {}

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

    def set_goal(self, steps: int) -> LogicProgramOptimized:
        if steps > self._max_steps:
            raise ValueError(f"Requested steps {steps} exceed maximum configured steps {self._max_steps}.")
        
        self._goal_step = steps
        self._goal_state = []

        for x in range(self._sokoban_map.height):
            for y in range(self._sokoban_map.width):
                if self._sokoban_map.grid[x][y].ground == Ground.STORAGE:
                    self._goal_state.append(self.__crate_var(x, y, steps))
        return self
    
    def save_dimacs(self, path: PathLike) -> LogicProgramOptimized:      
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
    
    def save_cnf_readable(self, path: PathLike) -> LogicProgramOptimized:
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
        if self._goal_step is None:
            raise ValueError("Goal step not set.")

        actions : List[Action] = [Action.NOOP] * self._goal_step
        argXs : List[int] = [0] * self._goal_step
        argYs : List[int] = [0] * self._goal_step
        argDs : List[Direction] = [Direction.UP] * self._goal_step

        for lit in literals:
            if lit > 0:
                if lit in self._action_map:
                    action, step = self._action_map[lit]
                    actions[step] = action
                elif lit in self._argX_map:
                    x, step = self._argX_map[lit]
                    argXs[step] = x
                elif lit in self._argY_map:
                    y, step = self._argY_map[lit]
                    argYs[step] = y
                elif lit in self._argD_map:
                    direction, step = self._argD_map[lit]
                    argDs[step] = direction
        
        solution: Solution = []
        for step in range(self._goal_step):
            if actions[step] != Action.NOOP:
                solution.append(SolutionStep(actions[step], argXs[step], argYs[step], argDs[step], step))
        
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
    def __sokobanX(cls, x: int, step: int) -> str:
        return f"sokobanX({x},{step})"
    
    @classmethod
    def __sokobanY(cls, y: int, step: int) -> str:
        return f"sokobanY({y},{step})"
    
    @classmethod
    def __action(cls, action: Action, step: int) -> str:
        return f"action({action.to_string()},{step})"
    
    @classmethod
    def __argX(cls, x: int, step: int) -> str:
        return f"argX({x},{step})"
    
    @classmethod
    def __argY(cls, y: int, step: int) -> str:
        return f"argY({y},{step})"
    
    @classmethod
    def __argD(cls, direction: Direction, step: int) -> str:
        return f"argD({direction.to_string()},{step})"
    
    def __new_variable(self, name: str) -> int:
        # Append and return variable index for DIMACS
        self._vars_map.append(name)
        return len(self._vars_map)

    def __wall_var(self, x: int, y: int) -> int:
        if (x, y) not in self._wall_vars:
            self._wall_vars[(x, y)] = self.__new_variable(LogicProgramOptimized.__wall(x, y))
        return self._wall_vars[(x, y)]

    def __storage_var(self, x: int, y: int) -> int:
        if (x, y) not in self._storage_vars:
            self._storage_vars[(x, y)] = self.__new_variable(LogicProgramOptimized.__storage(x, y))
        return self._storage_vars[(x, y)]
    
    def __crate_var(self, x: int, y: int, step: int) -> int:
        if (x, y, step) not in self._crate_vars:
            self._crate_vars[(x, y, step)] = self.__new_variable(LogicProgramOptimized.__crate(x, y, step))
        return self._crate_vars[(x, y, step)]
    
    def __sokobanX_var(self, x: int, step: int) -> int:
        if (x, step) not in self._sokobanX_vars:
            self._sokobanX_vars[(x, step)] = self.__new_variable(LogicProgramOptimized.__sokobanX(x, step))
        return self._sokobanX_vars[(x, step)]
    
    def __sokobanY_var(self, y: int, step: int) -> int:
        if (y, step) not in self._sokobanY_vars:
            self._sokobanY_vars[(y, step)] = self.__new_variable(LogicProgramOptimized.__sokobanY(y, step))
        return self._sokobanY_vars[(y, step)]
    
    def __action_var(self, action: Action, step: int) -> int:
        if (action, step) not in self._action_vars:
            lit = self.__new_variable(LogicProgramOptimized.__action(action, step))
            self._action_vars[(action, step)] = lit
            self._action_map[lit] = (action, step)
        return self._action_vars[(action, step)]
    
    def __argX_var(self, x: int, step: int) -> int:
        if (x, step) not in self._argX_vars:
            lit = self.__new_variable(LogicProgramOptimized.__argX(x, step))
            self._argX_vars[(x, step)] = lit
            self._argX_map[lit] = (x, step)
        return self._argX_vars[(x, step)]
    
    def __argY_var(self, y: int, step: int) -> int:
        if (y, step) not in self._argY_vars:
            lit = self.__new_variable(LogicProgramOptimized.__argY(y, step))
            self._argY_vars[(y, step)] = lit
            self._argY_map[lit] = (y, step)
        return self._argY_vars[(y, step)]
    
    def __argD_var(self, direction: Direction, step: int) -> int:
        if (direction, step) not in self._argD_vars:
            lit = self.__new_variable(LogicProgramOptimized.__argD(direction, step))
            self._argD_vars[(direction, step)] = lit
            self._argD_map[lit] = (direction, step)
        return self._argD_vars[(direction, step)]
    
    def __valid_coords(self, x: int, y: int) -> bool:
        return 0 <= x < self._sokoban_map.height and 0 <= y < self._sokoban_map.width
    
    def __add_init(self) -> None:
        sokobanX = 0
        sokobanY = 0

        # Add initial state of sokoban map
        for x in range(self._sokoban_map.height):
            for y in range(self._sokoban_map.width):
                w = self.__wall_var(x, y)
                st = self.__storage_var(x, y)
                c = self.__crate_var(x, y, 0)

                # Static facts: whether a cell is a floor/wall/storage.
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

                if self._sokoban_map.grid[x][y].map_object is not None:
                    if self._sokoban_map.grid[x][y].map_object == MapObject.CRATE:
                        self._init_state.append(c)
                    else:
                        self._init_state.append(-c)
                        sokobanX = x
                        sokobanY = y
                else:
                    self._init_state.append(-c)
        
        # Sokoban position fluents in step 0
        for x in range(self._sokoban_map.height):
            sx = self.__sokobanX_var(x, 0)
            if x == sokobanX:
                self._init_state.append(sx)
            else:
                self._init_state.append(-sx)
        
        for y in range(self._sokoban_map.width):
            sy = self.__sokobanY_var(y, 0)
            if y == sokobanY:
                self._init_state.append(sy)
            else:
                self._init_state.append(-sy)
    
    def __add_move_effects(self) -> None:
        # Add preconditions and effects of move actions (without pushing any crate)
        for step in range(self._max_steps):
            action = self.__action_var(Action.MOVE, step)

            # Preconditions and effects for sokoban moving vertically
            for x in range(self._sokoban_map.height):
                argX = self.__argX_var(x, step)
                self._clauses[step].append([-action, -argX, self.__sokobanX_var(x, step)])
                for direction in Direction.vertical():
                    argD = self.__argD_var(direction, step)
                    targetX = x + direction.to_vector()[0]
                    if self.__valid_coords(targetX, 0):
                        self._clauses[step].append([-action, -argX, -argD, -self.__sokobanX_var(x, step + 1)])
                        self._clauses[step].append([-action, -argX, -argD, self.__sokobanX_var(targetX, step + 1)])
        
            # Preconditions and effects for sokoban moving horizontally
            for y in range(self._sokoban_map.width):
                argY = self.__argY_var(y, step)
                self._clauses[step].append([-action, -argY, self.__sokobanY_var(y, step)])
                for direction in Direction.horizontal():
                    argD = self.__argD_var(direction, step)
                    targetY = y + direction.to_vector()[1]
                    if self.__valid_coords(0, targetY):
                        self._clauses[step].append([-action, -argY, -argD, -self.__sokobanY_var(y, step + 1)])
                        self._clauses[step].append([-action, -argY, -argD, self.__sokobanY_var(targetY, step + 1)])
        
            # Preconditions that sokoban can only move into free cells
            for x in range(self._sokoban_map.height):
                argX = self.__argX_var(x, step)
                for y in range(self._sokoban_map.width):
                    argY = self.__argY_var(y, step)
                    for direction in Direction:
                        argD = self.__argD_var(direction, step)
                        target = np.add((x, y), direction.to_vector())

                        # Forbid moves that would send sokoban out of map
                        if not self.__valid_coords(*target):
                            self._clauses[step].append([-action, -argX, -argY, -argD])
                            continue

                        # Preconditions: target cell must not be a wall or a crate
                        self._clauses[step].append([-action, -argX, -argY, -argD, -self.__wall_var(*target)])
                        self._clauses[step].append([-action, -argX, -argY, -argD, -self.__crate_var(*target, step)])

    def __add_push_effects(self) -> None:
        # Add preconditions and effects of push actions (sokoban moves together with pushing crate)
        for step in range(self._max_steps):
            action = self.__action_var(Action.PUSH, step)

            # Preconditions and effects for sokoban moving while pushing crates vertically
            for x in range(self._sokoban_map.height):
                argX = self.__argX_var(x, step)
                self._clauses[step].append([-action, -argX, self.__sokobanX_var(x, step)])
                for direction in Direction.vertical():
                    argD = self.__argD_var(direction, step)
                    targetX = x + direction.to_vector()[0]
                    if self.__valid_coords(targetX, 0):
                        self._clauses[step].append([-action, -argX, -argD, -self.__sokobanX_var(x, step + 1)])
                        self._clauses[step].append([-action, -argX, -argD, self.__sokobanX_var(targetX, step + 1)])
            
            # Preconditions and effects for sokoban moving while pushing crates horizontally
            for y in range(self._sokoban_map.width):
                argY = self.__argY_var(y, step)
                self._clauses[step].append([-action, -argY, self.__sokobanY_var(y, step)])
                for direction in Direction.horizontal():
                    argD = self.__argD_var(direction, step)
                    targetY = y + direction.to_vector()[1]
                    if self.__valid_coords(0, targetY):
                        self._clauses[step].append([-action, -argY, -argD, -self.__sokobanY_var(y, step + 1)])
                        self._clauses[step].append([-action, -argY, -argD, self.__sokobanY_var(targetY, step + 1)])
            
            # Preconditions and effects for pushing crates
            for x in range(self._sokoban_map.height):
                for y in range(self._sokoban_map.width):
                    for direction in Direction:
                        argX = self.__argX_var(x, step)
                        argY = self.__argY_var(y, step)
                        argD = self.__argD_var(direction, step)

                        crate_pos = np.add((x, y), direction.to_vector())
                        target = np.add(crate_pos, direction.to_vector())

                        # Forbid pushes that would send crate out of map
                        if not self.__valid_coords(*target):
                            self._clauses[step].append([-action, -argX, -argY, -argD])
                            continue

                        # Preconditions: crate at crate_pos and target cell free
                        self._clauses[step].append([-action, -argX, -argY, -argD, self.__crate_var(*crate_pos, step)])
                        self._clauses[step].append([-action, -argX, -argY, -argD, -self.__wall_var(*target)])
                        self._clauses[step].append([-action, -argX, -argY, -argD, -self.__crate_var(*target, step)])

                        # Effects: the crate moves to the target cell
                        self._clauses[step].append([-action, -argX, -argY, -argD, self.__crate_var(*target, step + 1)])
                        self._clauses[step].append([-action, -argX, -argY, -argD, -self.__crate_var(*crate_pos, step + 1)])

    def __add_mutual_exclusion(self, vars: List[int], step: int) -> None:
        # Add clauses ensuring mutual exclusion among the given variables
        for i1 in range(len(vars)):
            for i2 in range(i1 + 1, len(vars)):
                self._clauses[step].append([-vars[i1], -vars[i2]])

    def __add_one_action_per_step(self) -> None:
        # Add clauses ensuring exactly one action and one of each arguments per step
        for step in range(self._max_steps):
            action_vars: List[int] = [self.__action_var(action, step) for action in Action]
            argX_vars: List[int] = [self.__argX_var(x, step) for x in range(self._sokoban_map.height)]
            argY_vars: List[int] = [self.__argY_var(y, step) for y in range(self._sokoban_map.width)]
            argD_vars: List[int] = [self.__argD_var(direction, step) for direction in Direction]
            
            # At least one action and each type of argument
            self._clauses[step].append(action_vars)
            self._clauses[step].append(argX_vars)
            self._clauses[step].append(argY_vars)
            self._clauses[step].append(argD_vars)

            # At most one action and each type of argument
            self.__add_mutual_exclusion(action_vars, step)
            self.__add_mutual_exclusion(argX_vars, step)
            self.__add_mutual_exclusion(argY_vars, step)
            self.__add_mutual_exclusion(argD_vars, step)
    
    def __all_combinations(self, lists: List[List[int]]) -> List[List[int]]:
        # Helper function to compute Cartesian product of lists of literals
        if not lists:
            return [[]]
        
        result: List[List[int]] = []
        first, *rest = lists

        for item in first:
            for combination in self.__all_combinations(rest):
                if item not in combination:
                    result.append([item] + combination)
        return result
    
    def __add_frame_axioms(self) -> None:
        # Add explanatory frame problem axioms
        for step in range(self._max_steps):
            for x in range(self._sokoban_map.height):
                # Sokoban moved from row x up or down
                clause = [-self.__sokobanX_var(x, step), self.__sokobanX_var(x, step + 1)]
                self._clauses[step].append(clause + [self.__action_var(Action.MOVE, step), self.__action_var(Action.PUSH, step)])
                self._clauses[step].append(clause + [self.__argX_var(x, step)])
                self._clauses[step].append(clause + [self.__argD_var(Direction.UP, step), self.__argD_var(Direction.DOWN, step)])
                    
                # Sokoban moved to row x from up or down
                clause = [self.__sokobanX_var(x, step), -self.__sokobanX_var(x, step + 1)]
                valid_args = []
                for direction in Direction.vertical():
                    sourceX = x - direction.to_vector()[0]
                    if not self.__valid_coords(sourceX, 0):
                        continue
                    valid_args.append([self.__argX_var(sourceX, step), self.__argD_var(direction, step)])
                if valid_args:
                    self._clauses[step].append(clause + [self.__action_var(Action.MOVE, step), self.__action_var(Action.PUSH, step)])
                    for combination in self.__all_combinations(valid_args):
                        self._clauses[step].append(clause + combination)
                
            for y in range(self._sokoban_map.width):
                # Sokoban moved from column y left or right
                clause = [-self.__sokobanY_var(y, step), self.__sokobanY_var(y, step + 1)]
                self._clauses[step].append(clause + [self.__action_var(Action.MOVE, step), self.__action_var(Action.PUSH, step)])
                self._clauses[step].append(clause + [self.__argY_var(y, step)])
                self._clauses[step].append(clause + [self.__argD_var(Direction.LEFT, step), self.__argD_var(Direction.RIGHT, step)])

                # Sokoban moved to column y from left or right
                clause = [self.__sokobanY_var(y, step), -self.__sokobanY_var(y, step + 1)]
                valid_args = []
                for direction in Direction.horizontal():
                    sourceY = y - direction.to_vector()[1]
                    if not self.__valid_coords(0, sourceY):
                        continue
                    valid_args.append([self.__argY_var(sourceY, step), self.__argD_var(direction, step)])
                if valid_args:
                    self._clauses[step].append(clause + [self.__action_var(Action.MOVE, step), self.__action_var(Action.PUSH, step)])
                    for combination in self.__all_combinations(valid_args):
                        self._clauses[step].append(clause + combination)
            
            for x in range(self._sokoban_map.height):
                for y in range(self._sokoban_map.width):
                    # Crate was moved from cell (x,y)
                    clause = [-self.__crate_var(x, y, step), self.__crate_var(x, y, step + 1)]
                    valid_args = []
                    for direction in Direction:
                        source = np.subtract((x, y), direction.to_vector())
                        if self.__valid_coords(*source):
                            valid_args.append((self.__argX_var(source[0], step), self.__argY_var(source[1], step), self.__argD_var(direction, step)))
                    if valid_args:
                        self._clauses[step].append(clause + [self.__action_var(Action.PUSH, step)])
                        for combination in self.__all_combinations(valid_args):
                            self._clauses[step].append(clause + combination)
                    
                    # Crate was moved to cell (x,y)
                    clause = [self.__crate_var(x, y, step), -self.__crate_var(x, y, step + 1)]
                    valid_args = []
                    for direction in Direction:
                        source = np.subtract((x, y), 2 * direction.to_vector())
                        if self.__valid_coords(*source):
                            valid_args.append((self.__argX_var(source[0], step), self.__argY_var(source[1], step), self.__argD_var(direction, step)))
                    if valid_args:
                        self._clauses[step].append(clause + [self.__action_var(Action.PUSH, step)])
                        for combination in self.__all_combinations(valid_args):
                            self._clauses[step].append(clause + combination)