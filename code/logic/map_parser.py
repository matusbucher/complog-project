from pathlib import Path
from typing import Sequence, List, Union

from logic.sokoban_map import Ground, MapObject, SokobanMap, Tile


PathLike = Union[str, Path]

WALL_CHAR: str = "#"
STORAGE_CHARS: List[str] = ["X", "c", "s"]
CRATE_CHARS: List[str] = ["C", "c"]
SOKOBAN_CHARS: List[str] = ["S", "s"]


class MapParserError(ValueError):
    """
    Raised when map parsing encounters invalid or inconsistent state.
    """

class MapParser:
    """
    Parse a sokoban map in text representation into a SokobanMap object.
    """

    @classmethod
    def from_lines(cls, lines: Sequence[str]) -> SokobanMap:
        """
        Parse the given lines and return a SokobanMap.
        """
        grid: List[List[Tile]] = []

        height = len(lines)
        width = max(len(line) for line in lines)

        seen_sokoban: bool = False
        for line in lines:
            row: List[Tile] = []
            for i in range(width):
                tile = Tile()
                ch = line[i] if i < len(line) else " "
                if ch == WALL_CHAR:
                    tile.ground = Ground.WALL
                elif ch in STORAGE_CHARS:
                    tile.ground = Ground.STORAGE
                else:
                    tile.ground = Ground.FLOOR

                if ch in CRATE_CHARS:
                    tile.map_object = MapObject.CRATE
                if ch in SOKOBAN_CHARS:
                    if seen_sokoban:
                        raise MapParserError(f"Map contains more than one sokoban player.")
                    seen_sokoban = True
                    tile.map_object = MapObject.SOKOBAN
                row.append(tile)
            grid.append(row)
        
        if not seen_sokoban:
            raise MapParserError("Map does not contain a sokoban player.")

        return SokobanMap(width=width, height=height, grid=grid)

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