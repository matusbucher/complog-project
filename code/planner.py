from __future__ import annotations
import os
from pathlib import Path
import subprocess
from typing import List, Optional, Tuple, Union

from logic_program_basic import LogicProgramBasic
from logic_program_interface import LogicProgramInterface, Solution
from logic_program_optimized import LogicProgramOptimized
from sokoban_map import SokobanMap


PathLike = Union[str, Path]


class Planner():
    """
    Finds the shortest solution for the given Sokoban map and chosen logic program.

    Wrapper around `LogicProgramInterface`, implementing binary search.
    """

    def __init__(self, sokoban_map: SokobanMap, max_steps: int, solver_path: PathLike, solver_input: PathLike, solver_output: PathLike, optimize: bool = True) -> None:
        """
        Initialize the planner.

        :param sokoban_map: The Sokoban map to solve.
        :param max_steps: The maximum number of steps to consider.
        :param solver_path: Path to the MiniSat binary.
        :param solver_input: Path to the file where the CNF input for MiniSat will be written.
        :param solver_output: Path to the file where the MiniSat output will be written.
        :param optimized: Whether to use the optimized logic program or the basic one.
        """
        if solver_path is None or not os.path.isfile(solver_path):
            raise FileNotFoundError(f"MiniSat binary not found at path: {solver_path!s}")
        
        if solver_input is None or os.path.exists(solver_input) and not os.access(solver_input, os.W_OK):
            raise FileNotFoundError(f"Cannot write to solver input file: {solver_input!s}")
        
        if solver_output is None or os.path.exists(solver_output) and not os.access(solver_output, os.W_OK):
            raise FileNotFoundError(f"Cannot write to solver output file: {solver_output!s}")

        if optimize:
            self._logic_program: LogicProgramInterface = LogicProgramOptimized(sokoban_map, max_steps)
        else:
            self._logic_program: LogicProgramInterface = LogicProgramBasic(sokoban_map, max_steps)
        
        self._solver_path: PathLike = solver_path
        self._solver_input: PathLike = solver_input
        self._solver_output: PathLike = solver_output

    def check_solvability(self) -> bool:
        """
        Check if the problem is solvable within the maximum number of steps.
        """
        self._logic_program.set_goal(self._logic_program.get_max_steps()).save_dimacs(self._solver_input)
        
        result = subprocess.run(
            [self._solver_path, self._solver_input, self._solver_output],
            capture_output=True,
            text=True
        )
        
        return result.returncode == 10

    def find_shortest_solution(self) -> Solution:
        """
        Search for a shortest plan by calling the MiniSat solver repeatedly. If found, return a list of ordered actions.
        """
        least_steps = -1
        left, right = 1, self._logic_program.get_max_steps()

        # Binary search for the shortest plan
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

    def __parse_solution(self) -> Solution:
        solution: List[str] = []
        with open(self._solver_output, "r") as file:
            file.readline()  # Skip "SAT" line
            literals = [int(x) for x in file.readline().strip().split() if x != "0"]
            solution = self._logic_program.extract_solution(literals)
        return solution
    
    def debug_print(self, path: Optional[PathLike] = None) -> Planner:
        pos_vars: List[str] = []
        with open(self._solver_output, "r") as file:
            file.readline()  # Skip "SAT" line
            literals = [int(x) for x in file.readline().strip().split() if x != "0"]
            pos_vars = [self._logic_program.lit_to_str(lit) for lit in literals if lit > 0]
        
        if path is None:
            for var in pos_vars:
                print(var)
            return self
        
        p = Path(path)
        with p.open("w") as file:
            for var in pos_vars:
                file.write(f"{var}\n")
        return self