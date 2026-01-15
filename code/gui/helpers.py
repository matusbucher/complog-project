from __future__ import annotations
from enum import Enum
from typing import Optional

from logic.sokoban_map import MapObject, Ground


class Tool(Enum):
    NONE = 0
    WALL = 1
    FLOOR = 2
    STORAGE = 3
    CRATE = 4
    SOKOBAN = 5

    @staticmethod
    def from_map_object(map_object : MapObject) -> Tool:
        if map_object == MapObject.CRATE:
            return Tool.CRATE
        elif map_object == MapObject.SOKOBAN:
            return Tool.SOKOBAN
        else:
            return Tool.NONE
    
    @staticmethod
    def from_ground(ground : Ground) -> Tool:
        if ground == Ground.WALL:
            return Tool.WALL
        elif ground == Ground.FLOOR:
            return Tool.FLOOR
        elif ground == Ground.STORAGE:
            return Tool.STORAGE
        else:
            return Tool.NONE
        

class Settings:
    minisat_binary_file : Optional[str] = None
    minisat_files_folder : Optional[str] = None
    optimizations_enabled : bool = True

    def complete(self) -> bool:
        return self.minisat_binary_file is not None and self.minisat_files_folder is not None