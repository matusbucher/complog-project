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

    @staticmethod
    def from_lines(lines: Sequence[str]) -> SokobanMap:
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

    @staticmethod
    def from_file(path: PathLike) -> SokobanMap:
        """
        Read the file, parse it and return a SokobanMap.
        """
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Map file not found: {path!s}")

        with p.open("r") as file:
            lines = [line.rstrip("\n") for line in file]

        return MapParser.from_lines(lines)
    
    @staticmethod
    def map_to_lines(sokoban_map: SokobanMap) -> List[str]:
        """
        Convert a SokobanMap back to its text representation.
        """
        lines: List[str] = []
        for i in range(sokoban_map.height):
            line_chars: List[str] = []
            for j in range(sokoban_map.width):
                tile = sokoban_map.grid[i][j]
                ch: str = " "
                if tile.ground == Ground.WALL:
                    ch = WALL_CHAR
                elif tile.ground == Ground.STORAGE:
                    if tile.map_object == MapObject.CRATE:
                        ch = "c"
                    elif tile.map_object == MapObject.SOKOBAN:
                        ch = "s"
                    else:
                        ch = "X"
                else:  # Floor
                    if tile.map_object == MapObject.CRATE:
                        ch = "C"
                    elif tile.map_object == MapObject.SOKOBAN:
                        ch = "S"
                    else:
                        ch = " "
                line_chars.append(ch)
            lines.append("".join(line_chars).rstrip())
        return lines