from __future__ import annotations
from enum import Enum
from typing import List, Optional, Tuple


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

    def __init__(self, height: int, width: int, grid: Optional[List[List[Tile]]] = None):
        if grid is not None:
            self.grid = grid
            if len(grid) != height or any(len(row) != width for row in grid):
                raise ValueError("Grid dimensions do not match specified height and width.")
        else:
            self.grid = [[Tile() for _ in range(width)] for _ in range(height)]
        self.height = height
        self.width = width
    
    def valid_map(self) -> Tuple[bool, bool]:
        """
        Check if the Sokoban map is valid. Returns a tuple (has_one_sokoban, crates_equal_storage).
        """
        sokoban_count = 0
        crate_count = 0
        storage_count = 0

        for row in self.grid:
            for tile in row:
                if tile.map_object == MapObject.SOKOBAN:
                    sokoban_count += 1
                elif tile.map_object == MapObject.CRATE:
                    crate_count += 1
                if tile.ground == Ground.STORAGE:
                    storage_count += 1

        return sokoban_count == 1, crate_count == storage_count