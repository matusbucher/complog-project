import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from typing import Dict, List, Optional, Tuple

from logic.logic_program_interface import Action, Direction, SolutionStep
from gui.helpers import Tool
from gui.images import MapObject, load_map_object_images, load_ground_images, resize_image
from logic.sokoban_map import SokobanMap, Ground


MAX_CELL_SIZE = 100


class MapGrid:
    def __init__(self, root, parent, canvas_size : int):
        self._root = root
        self._canvas_size : int = canvas_size
        self._canvas : tk.Canvas = tk.Canvas(parent, width=canvas_size, height=canvas_size, bg="white")
        self._canvas.pack(side=tk.LEFT, fill=tk.Y, expand=True)
        self._canvas.bind("<Button-1>", lambda event: self.__handle_left_click(event.x, event.y))
        self._canvas.bind("<Button-3>", lambda event: self.__handle_right_click(event.x, event.y))

        self._ground_images : Dict[Ground, Image.Image] = load_ground_images()
        self._object_images : Dict[MapObject, Image.Image] = load_map_object_images()

        self._sokoban_map : Optional[SokobanMap] = None
        self._image_ids : Optional[List[List[List[Optional[int]]]]] = None
        self._resized_ground_images : Optional[Dict[Ground, ImageTk.PhotoImage]] = None
        self._resized_object_images : Optional[Dict[MapObject, ImageTk.PhotoImage]] = None
        self._cell_size : int = 0
        self._start_x : int = 0
        self._start_y : int = 0
        self._end_x : int = 0
        self._end_y : int = 0
        self._cell_back_size : int = 0
        self._cell_front_size : int = 0

    def set_map(self, sokoban_map : SokobanMap) -> None:
        self._sokoban_map = sokoban_map

        self._cell_size = min(MAX_CELL_SIZE, self._canvas_size // self._sokoban_map.width, self._canvas_size // self._sokoban_map.height)
        grid_width = self._sokoban_map.width * self._cell_size
        grid_height = self._sokoban_map.height * self._cell_size
        self._start_x = (self._canvas_size - grid_width) // 2
        self._start_y = (self._canvas_size - grid_height) // 2
        self._end_x = self._start_x + grid_width
        self._end_y = self._start_y + grid_height
        self._cell_back_size = self._cell_size
        self._cell_front_size = int(self._cell_back_size * 0.6)

        self._resized_ground_images = {
            ground: resize_image(img, self._cell_back_size) for ground, img in self._ground_images.items()
        }
        self._resized_object_images = {
            obj: resize_image(img, self._cell_front_size) for obj, img in self._object_images.items()   
        }

        self.__draw_grid()
    
    def get_map(self) -> Optional[SokobanMap]:
        return self._sokoban_map

    def clear_map(self) -> None:
        if self._sokoban_map is None:
            return
        
        self._sokoban_map = SokobanMap(self._sokoban_map.height, self._sokoban_map.width)
        self.__draw_grid()

    def disable_editing(self) -> None:
        self._canvas.unbind("<Button-1>")
        self._canvas.unbind("<Button-3>")
    
    def enable_editing(self) -> None:
        self._canvas.bind("<Button-1>", lambda event: self.__handle_left_click(event.x, event.y))
        self._canvas.bind("<Button-3>", lambda event: self.__handle_right_click(event.x, event.y))

    def execute_step(self, step : SolutionStep, inverse : bool) -> None:
        if self._sokoban_map is None:
            return
        
        if step.action == Action.MOVE:
            self.__execute_move(step.x, step.y, step.direction, inverse)
        elif step.action == Action.PUSH:
            if inverse:
                self.__execute_inverse_push(step.x, step.y, step.direction)
            else:
                self.__execute_push(step.x, step.y, step.direction)


    def __execute_move(self, x : int, y : int, direction : Direction, inverse : bool) -> None:
        from_i, from_j = x, y
        to_i, to_j = np.add((x, y), direction.to_vector())
        if inverse:
            from_i, from_j, to_i, to_j = to_i, to_j, from_i, from_j

        self._sokoban_map.grid[from_i][from_j].map_object = None
        self.__delete_object(from_i, from_j)
        self._sokoban_map.grid[to_i][to_j].map_object = MapObject.SOKOBAN
        self.__draw_object(to_i, to_j, MapObject.SOKOBAN)
    
    def __execute_push(self, x : int, y : int, direction : Direction) -> None:
        to_i, to_j = np.add((x, y), direction.to_vector())
        crate_i, crate_j = np.add((to_i, to_j), direction.to_vector())
        self._sokoban_map.grid[x][y].map_object = None
        self.__delete_object(x, y)
        self.__delete_object(to_i, to_j)
        self._sokoban_map.grid[to_i][to_j].map_object = MapObject.SOKOBAN
        self.__draw_object(to_i, to_j, MapObject.SOKOBAN)
        self._sokoban_map.grid[crate_i][crate_j].map_object = MapObject.CRATE
        self.__draw_object(crate_i, crate_j, MapObject.CRATE)

    def __execute_inverse_push(self, x : int, y : int, direction : Direction) -> None:
        from_i, from_j = np.add((x, y), direction.to_vector())
        crate_i, crate_j = np.add((from_i, from_j), direction.to_vector())
        self._sokoban_map.grid[crate_i][crate_j].map_object = None
        self.__delete_object(crate_i, crate_j)
        self.__delete_object(from_i, from_j)
        self._sokoban_map.grid[from_i][from_j].map_object = MapObject.CRATE
        self.__draw_object(from_i, from_j, MapObject.CRATE)
        self._sokoban_map.grid[x][y].map_object = MapObject.SOKOBAN
        self.__draw_object(x, y, MapObject.SOKOBAN)

    def __handle_right_click(self, x : int, y : int) -> None:
        if self._sokoban_map is None:
            return
        
        indices = self.__get_indices(x, y)
        if indices is None:
            return
        
        i, j = indices
        self._sokoban_map.grid[i][j].map_object = None
        self.__delete_object(i, j)

    def __handle_left_click(self, x : int, y : int) -> None:
        if self._sokoban_map is None:
            return
        
        indices = self.__get_indices(x, y)
        if indices is None:
            return
        
        i, j = indices
        tool = self._root.get_selected_tool()
        if tool == Tool.FLOOR:
            self._sokoban_map.grid[i][j].ground = Ground.FLOOR
            self.__draw_ground(i, j, Ground.FLOOR)
        elif tool == Tool.WALL:
            self._sokoban_map.grid[i][j].ground = Ground.WALL
            self.__draw_ground(i, j, Ground.WALL)
        elif tool == Tool.STORAGE:
            self._sokoban_map.grid[i][j].ground = Ground.STORAGE
            self.__draw_ground(i, j, Ground.STORAGE)
        elif tool == Tool.CRATE:
            self._sokoban_map.grid[i][j].map_object = MapObject.CRATE
            self.__draw_object(i, j, MapObject.CRATE)
        elif tool == Tool.SOKOBAN:
            self._sokoban_map.grid[i][j].map_object = MapObject.SOKOBAN
            self.__draw_object(i, j, MapObject.SOKOBAN)

    def __get_indices(self, x : int, y : int) -> Optional[Tuple[int, int]]:
        if x < self._start_x or x >= self._end_x or y < self._start_y or y >= self._end_y:
            return None
        
        j = (x - self._start_x) // self._cell_size
        i = (y - self._start_y) // self._cell_size
        return (i, j)
        
    def __get_coords(self, i : int, j : int) -> Tuple[int, int]:
        x = self._start_x + j * self._cell_size + self._cell_size // 2
        y = self._start_y + i * self._cell_size + self._cell_size // 2
        return (x, y)
    
    def __delete_object(self, i : int, j : int) -> None:
        if self._image_ids[i][j][1] is not None:
            self._canvas.delete(self._image_ids[i][j][1])
            self._image_ids[i][j][1] = None
        
    def __draw_ground(self, i : int, j : int, ground : Ground) -> None:
        if self._image_ids[i][j][0] is not None:
            self._canvas.delete(self._image_ids[i][j][0])
        
        x, y = self.__get_coords(i, j)
        image = self._canvas.create_image(x, y, anchor=tk.CENTER, image=self._resized_ground_images[ground])
        self._image_ids[i][j][0] = image
        self._canvas.tag_lower(image)

        if ground == Ground.WALL and self._image_ids[i][j][1] is not None:
            self._canvas.delete(self._image_ids[i][j][1])
            self._image_ids[i][j][1] = None

    def __draw_object(self, i : int, j : int, obj : MapObject) -> None:
        if self._sokoban_map.grid[i][j].ground == Ground.WALL:
            return
        
        if self._image_ids[i][j][1] is not None:
            self._canvas.delete(self._image_ids[i][j][1])
        
        x, y = self.__get_coords(i, j)
        image = self._canvas.create_image(x, y, anchor=tk.CENTER, image=self._resized_object_images[obj])
        self._image_ids[i][j][1] = image
        self._canvas.tag_raise(image)
    
    def __draw_grid(self) -> None:
        self._canvas.delete("all")
        
        for x in range(self._start_x, self._end_x + 1, self._cell_size):
            self._canvas.create_line(x, self._start_y, x, self._end_y)
        
        for y in range(self._start_y, self._end_y + 1, self._cell_size):
            self._canvas.create_line(self._start_x, y, self._end_x, y)

        self._image_ids = [[[None, None] for _ in range(self._sokoban_map.width)] for _ in range(self._sokoban_map.height)]

        for i in range(self._sokoban_map.height):
            for j in range(self._sokoban_map.width):
                ground = self._sokoban_map.grid[i][j].ground
                self.__draw_ground(i, j, ground)
                obj = self._sokoban_map.grid[i][j].map_object
                if obj is not None:
                    self.__draw_object(i, j, obj)

        self._canvas.pack()