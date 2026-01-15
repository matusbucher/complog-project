from PIL import ImageTk
import tkinter as tk
from typing import Dict, List

from gui.helpers import Tool
from gui.images import load_clear_image, load_edit_image, load_map_object_images, load_ground_images, resize_image


BUTTON_SIZE = 60
BACKGROUNF_COLOR = "lightgray"
HIGHLIGHT_COLOR = "red"


class ToolPanel:
    def __init__(self, root, parent):
        self._root = root
        self._frame = tk.Frame(parent, bg=BACKGROUNF_COLOR, padx=10, pady=10)
        self._frame.pack(side=tk.LEFT, fill=tk.Y)

        self._selected_tool : Tool = Tool.NONE

        self._tool_images : Dict[Tool, ImageTk.PhotoImage] = {}
        self._tool_images.update({Tool.from_ground(ground): resize_image(img, BUTTON_SIZE) for ground, img in load_ground_images().items()})
        self._tool_images.update({Tool.from_map_object(obj): resize_image(img, BUTTON_SIZE) for obj, img in load_map_object_images().items()})
        self._clear_image = resize_image(load_clear_image(), BUTTON_SIZE)
        self._edit_image = resize_image(load_edit_image(), BUTTON_SIZE)

        self._editing_buttons : List[tk.Button] = []
        self._button_frame_map : Dict[Tool, tk.Frame] = {}
        row_idx = 0
        for tool in Tool:
            if tool == Tool.NONE:
                continue
            frame = tk.Frame(self._frame, background=BACKGROUNF_COLOR, padx=4, pady=4)
            frame.grid(row=row_idx, column=0, pady=5)
            button = tk.Button(frame, image=self._tool_images[tool], relief="flat", command=lambda t=tool: self.__click_on_tool(t))
            button.pack()
            self._editing_buttons.append(button)
            self._button_frame_map[tool] = frame
            row_idx += 1
        
        separator = tk.Frame(self._frame, height=2, bg="gray", bd=1, relief="sunken")
        separator.grid(row=row_idx, column=0, sticky="ew", pady=10)
        
        clear_button = tk.Button(self._frame, image=self._clear_image, relief="flat", command=self.__clear_map)
        clear_button.grid(row=row_idx + 1, column=0, pady=8)
        self._editing_buttons.append(clear_button)

        self._edit_button = tk.Button(self._frame, image=self._edit_image, relief="flat", command=self._root.enable_editing, state=tk.DISABLED)
        self._edit_button.grid(row=row_idx + 2, column=0, pady=8)
    def get_selected_tool(self) -> Tool:
        return self._selected_tool
    
    def disable_editing(self) -> None:
        if self._selected_tool != Tool.NONE:
            self._button_frame_map[self._selected_tool].config(background=BACKGROUNF_COLOR)
            self._selected_tool = Tool.NONE
        for button in self._editing_buttons:
            button.config(state=tk.DISABLED)
        self._edit_button.config(state=tk.NORMAL)

    def enable_editing(self) -> None:
        for button in self._editing_buttons:
            button.config(state=tk.NORMAL)
        self._edit_button.config(state=tk.DISABLED)
    
    def __click_on_tool(self, tool : Tool) -> None:
        if self._selected_tool == Tool.NONE:
            self._button_frame_map[tool].config(background=HIGHLIGHT_COLOR)
            self._selected_tool = tool
            return

        if self._selected_tool == tool:
            self._button_frame_map[tool].config(background=BACKGROUNF_COLOR)
            self._selected_tool = Tool.NONE
            return
        
        self._button_frame_map[self._selected_tool].config(background=BACKGROUNF_COLOR)
        self._button_frame_map[tool].config(background=HIGHLIGHT_COLOR)
        self._selected_tool = tool

    def __clear_map(self) -> None:
        self._root.clear_map()