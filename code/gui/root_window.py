import tkinter as tk
from typing import Optional

from logic.logic_program_interface import SolutionStep
from gui.helpers import Settings, Tool
from logic.sokoban_map import SokobanMap
from gui.tool_panel import ToolPanel
from gui.menu_bar import MenuBar
from gui.map_grid import MapGrid
from gui.solution_panel import SolutionPanel


ICON_PATH = "code/assets/sokoban.ico"
CANVAS_SIZE = 600


class RootWindow:
    def __init__(self, window):
        self._window = window
        self._window.title("Sokoban Game Planner")
        self._window.iconbitmap(ICON_PATH)
        
        self._menu_bar = MenuBar(self, self._window)
        
        main_container = tk.Frame(self._window)
        main_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self._editing_panel = ToolPanel(self, main_container)
        self._map_grid = MapGrid(self, main_container, canvas_size=CANVAS_SIZE)
        self._solution_panel = SolutionPanel(self, main_container)

        self._settings : Settings = Settings()
    
    def set_map(self, sokoban_map : SokobanMap) -> None:
        self._map_grid.set_map(sokoban_map)
        self.enable_editing()
    
    def get_map(self) -> Optional[SokobanMap]:
        return self._map_grid.get_map()

    def clear_map(self) -> None:
        self._map_grid.clear_map()
    
    def get_selected_tool(self) -> Tool:
        return self._editing_panel.get_selected_tool()
    
    def disable_editing(self) -> None:
        self._editing_panel.disable_editing()
        self._map_grid.disable_editing()

    def enable_editing(self) -> None:
        self._solution_panel.reset()
        self._editing_panel.enable_editing()
        self._map_grid.enable_editing()
    
    def set_settings(self, settings : Settings) -> None:
        self._settings = settings

    def get_settings(self) -> Settings:
        return self._settings
    
    def open_settings_dialog(self) -> bool:
        self._menu_bar.open_settings_dialog()
    
    def execute_step(self, step : SolutionStep, inverse : bool) -> None:
        self._map_grid.execute_step(step, inverse)