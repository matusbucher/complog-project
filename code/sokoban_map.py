from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union


PathLike = Union[str, Path]
Position = Tuple[int, int]

WALL_CHAR: str = "#"
STORAGE_CHARS: List[str] = ["X", "c", "s"]
CRATE_CHARS: List[str] = ["C", "c"]
SOKOBAN_CHARS: List[str] = ["S", "s"]


class Tile(Enum):
    """
    Enum representing different types of tiles in a Sokoban map.
    """
    WALL = "#"
    STORAGE = "X"
    FLOOR = " "


@dataclass(frozen=True)
class SokobanMap:
    """
    Data class representing a Sokoban map.
    """
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

    @classmethod
    def from_lines(cls, lines: Sequence[str]) -> SokobanMap:
        """
        Parse the given lines and return a SokobanMap.
        """
        grid: List[List[Tile]] = []
        crates: List[Position] = []
        sokoban: Optional[Position] = None

        height = len(lines)
        width = max(len(line) for line in lines)

        for x, line in enumerate(lines):
            row: List[Tile] = []
            for y in range(width):
                ch = line[y] if y < len(line) else " "
                if ch == WALL_CHAR:
                    tile = Tile.WALL
                elif ch in STORAGE_CHARS:
                    tile = Tile.STORAGE
                else:
                    tile = Tile.FLOOR

                if ch in CRATE_CHARS:
                    crates.append((x, y))
                if ch in SOKOBAN_CHARS:
                    if sokoban is not None:
                        raise ParserError(f"Map contains more than one sokoban player position (previous at {sokoban}, additional at {(x, y)})")
                    sokoban = (x, y)

                row.append(tile)
            grid.append(row)
        
        if sokoban is None:
            raise ParserError("Map does not contain a sokoban player start position.")

        return SokobanMap(width=width, height=height, grid=grid, crates=crates, sokoban=sokoban)

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

        return cls.from_lines(lines)