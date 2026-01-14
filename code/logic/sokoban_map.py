from __future__ import annotations
from enum import Enum
from typing import List, Optional


class Ground(Enum):
    """
    Enum representing different types of ground on a Sokoban map.
    """
    WALL = 0
    STORAGE = 1
    FLOOR = 2

class MapObject(Enum):
    """
    Enum representing different objects on a Sokoban map.
    """
    CRATE = 0
    SOKOBAN = 1

class Tile:
    """
    Data class representing a tile on a Sokoban map.
    """
    ground: Ground = Ground.FLOOR
    map_object: Optional[MapObject] = None

class SokobanMap:
    """
    Data class representing a Sokoban map.
    """
    height: int
    width: int
    grid: List[List[Tile]]

    def __init__(self, height: int, width: int, grid: List[List[Tile]]):
        self.height = height
        self.width = width
        self.grid = grid

    @staticmethod
    def empty_map(height: int, width: int) -> SokobanMap:
        """
        Create an empty SokobanMap with the given dimensions.
        """
        grid: List[List[Tile]] = [[Tile(ground=Ground.FLOOR, map_object=None) for _ in range(width)] for _ in range(height)]
        return SokobanMap(height=height, width=width, grid=grid)