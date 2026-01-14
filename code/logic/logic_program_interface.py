from __future__ import annotations
from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
from pathlib import Path
from typing import List, Optional, Union, Tuple


class Action(Enum):
    MOVE = 0
    PUSH = 1
    NOOP = 2

    def to_string(self) -> str:
        """
        Return a string name representing the action.
        """
        match self:
            case Action.MOVE:
                return "move"
            case Action.PUSH:
                return "push"
            case Action.NOOP:
                return "noop"
            

class Direction(Enum):
    """
    Enum representing movement directions.
    """
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    @staticmethod
    def vertical() -> List[Direction]:
        """
        Return a list of vertical directions (up and down).
        """
        return [Direction.UP, Direction.DOWN]
    
    @staticmethod
    def horizontal() -> List[Direction]:
        """
        Return a list of horizontal directions (left and right).
        """
        return [Direction.LEFT, Direction.RIGHT]

    def to_string(self) -> str:
        """
        Return a string name representing the direction.
        """
        match self:
            case Direction.UP:
                return "up"
            case Direction.DOWN:
                return "down"
            case Direction.LEFT:
                return "left"
            case Direction.RIGHT:
                return "right"
    
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

        
PathLike = Union[str, Path]
SolutionStep = Tuple[Action, int, int, Direction, int]
Solution = Optional[List[SolutionStep]]


class LogicProgramInterface(ABC):
    @abstractmethod
    def set_goal(self, goal_step: int) -> LogicProgramInterface:
        """
        Set the goal step - that is, at `goal_step`, all crates should be already placed in storage cells.
        
        This results in a satisfiable or unsatisfiable program.
        """
        raise NotImplementedError
    
    @abstractmethod
    def save_dimacs(self, path: PathLike) -> LogicProgramInterface:
        """
        Write the current encoding to `path` in DIMACS format as a MiniSat solver input.
        
        If the goal step is not set yet, the encoding will contain clauses for each step up to the maximum steps.
        """
        raise NotImplementedError
    
    @abstractmethod
    def save_cnf_readable(self, path: PathLike) -> LogicProgramInterface:
        """
        Write a human-readable dump of the CNF to `path`.
        
        If the goal step is not set yet, the dump will contain clauses for each step up to the maximum steps.
        """

    @abstractmethod
    def get_max_steps(self) -> int:
        """
        Return the maximum number of steps the logic program can handle.
        """
        raise NotImplementedError

    @abstractmethod
    def lit_to_str(self, literal: int) -> str:
        """
        Given a literal, return a string representing of the variable.
        """
        raise NotImplementedError
    
    @abstractmethod
    def extract_solution(self, literals: List[int]) -> Solution:
        """
        Given a list of literals from a satisfying assignment, extract the sequence of actions as a solution.
        """
        raise NotImplementedError