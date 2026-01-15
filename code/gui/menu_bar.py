import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
from typing import Optional, Tuple

from gui.helpers import Settings
from logic.map_parser import MapParser
from logic.sokoban_map import SokobanMap


SETTING_ICON_PATH = "assets/setting.ico"


class DimensionDialog(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Enter Dimensions")
        self.iconbitmap(SETTING_ICON_PATH)
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        self.result : Optional[Tuple[int, int]] = None

        tk.Label(self, text="Width (max 15):").grid(row=0, column=0, padx=10, pady=5, sticky="e")
        tk.Label(self, text="Height (max 15):").grid(row=1, column=0, padx=10, pady=5, sticky="e")

        self.width_var = tk.StringVar()
        self.height_var = tk.StringVar()

        tk.Entry(self, textvariable=self.width_var, width=5).grid(row=0, column=1, padx=10, pady=5)
        tk.Entry(self, textvariable=self.height_var, width=5).grid(row=1, column=1, padx=10, pady=5)

        button_frame = tk.Frame(self)
        button_frame.grid(row=2, column=0, columnspan=2, pady=10)
        tk.Button(button_frame, text="OK", command=self.on_ok).pack(side="left", padx=5)
        tk.Button(button_frame, text="Cancel", command=self.destroy).pack(side="left", padx=5)

        self.bind("<Return>", lambda e: self.on_ok())
        self.bind("<Escape>", lambda e: self.destroy())

        self.wait_window(self)

    def on_ok(self) -> None:
        try:
            width = int(self.width_var.get())
            height = int(self.height_var.get())
            if not (1 <= width <= 15 and 1 <= height <= 15):
                raise ValueError
            self.result = (width, height)
            self.destroy()
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter dimensions between 1 and 15.")


class SettingsDialog(tk.Toplevel):
    def __init__(self, parent, old_settings: Settings = None):
        super().__init__(parent)
        self.title("Settings Configuration")
        self.iconbitmap(SETTING_ICON_PATH)
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        self.result: Optional[Settings] = None

        tk.Label(self, text="Minisat Binary:").grid(row=0, column=0, padx=10, pady=5, sticky="e")
        self.minisat_binary_var = tk.StringVar(value=old_settings.minisat_binary_file if old_settings.minisat_binary_file else "")
        tk.Entry(self, textvariable=self.minisat_binary_var).grid(row=0, column=1, padx=10, pady=5)
        tk.Button(self, text="...", command=self.pick_minisat_binary_file).grid(row=0, column=2, padx=5)

        tk.Label(self, text="Minisat Input/Output Folder:").grid(row=1, column=0, padx=10, pady=5, sticky="e")
        self.minisat_files_folder_var = tk.StringVar(value=old_settings.minisat_files_folder if old_settings.minisat_files_folder else "")
        tk.Entry(self, textvariable=self.minisat_files_folder_var).grid(row=1, column=1, padx=10, pady=5)
        tk.Button(self, text="...", command=self.pick_minisat_files_folder).grid(row=1, column=2, padx=5)

        tk.Label(self, text="Enable Optimization:").grid(row=3, column=0, padx=10, pady=5, sticky="e")
        self.optimization_var = tk.BooleanVar(value=old_settings.optimizations_enabled)
        tk.Checkbutton(self, variable=self.optimization_var).grid(row=3, column=1, padx=10, pady=5)

        button_frame = tk.Frame(self)
        button_frame.grid(row=4, column=0, columnspan=2, pady=10)
        tk.Button(button_frame, text="OK", command=self.on_ok).pack(side="left", padx=5)
        tk.Button(button_frame, text="Cancel", command=self.destroy).pack(side="left", padx=5)

        self.bind("<Return>", lambda e: self.on_ok())
        self.bind("<Escape>", lambda e: self.destroy())

        self.wait_window(self)

    def pick_minisat_binary_file(self):
        file = filedialog.askopenfilename(title="Select Minisat Binary")
        if file:
            self.minisat_binary_var.set(file)

    def pick_minisat_files_folder(self):
        folder = filedialog.askdirectory(title="Select Minisat Input/Output Folder")
        if folder:
            self.minisat_files_folder_var.set(folder)

    def on_ok(self) -> None:
        minisat_binary_file = self.minisat_binary_var.get()
        minisat_files_folder = self.minisat_files_folder_var.get()
        optimization_enabled = self.optimization_var.get()

        self.result = Settings()
        if minisat_binary_file:
            self.result.minisat_binary_file = minisat_binary_file
        if minisat_files_folder:
            self.result.minisat_files_folder = minisat_files_folder
        self.result.optimizations_enabled = optimization_enabled

        self.destroy()


class MenuBar:
    def __init__(self, root, parent):
        self._root = root
        self._parent : tk.Tk = parent
        self._bar : tk.Menu = tk.Menu(parent)

        self._map_menu : tk.Menu = tk.Menu(self._bar, tearoff=0)
        self._bar.add_cascade(label="Map", menu=self._map_menu)
        self._map_menu.add_command(label="New Map", command=self.__new_map)
        self._map_menu.add_command(label="Import", command=self.__import_map)
        self._map_menu.add_command(label="Export", command=self.__export_map)
        self._map_menu.add_separator()
        self._map_menu.add_command(label="Exit", command=self.__exit)
     
        self._options_menu : tk.Menu = tk.Menu(self._bar, tearoff=0)
        self._bar.add_cascade(label="Options", menu=self._options_menu)
        self._options_menu.add_command(label="Configure", command=self.open_settings_dialog)

        self._help_menu : tk.Menu = tk.Menu(self._bar, tearoff=0)
        self._bar.add_cascade(label="Help", menu=self._help_menu)
        self._help_menu.add_command(label="Manual", command=self.__manual)
        
        parent.config(menu=self._bar)

    def open_settings_dialog(self) -> bool:
        dialog = SettingsDialog(self._parent, self._root.get_settings())
        if dialog.result is None:
            return False
        settings = dialog.result
        self._root.set_settings(settings)
        return settings.complete()

    def __new_map(self) -> None:
        dialog = DimensionDialog(self._parent)
        if dialog.result is None:
            return

        width, height = dialog.result
        self._root.set_map(SokobanMap(height, width))
            
    def __import_map(self) -> None:
        file = filedialog.askopenfilename(
            title="Import Sokoban Map",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
        )

        if not file:
            return
        
        try:
            sokoban_map = MapParser.from_file(file)
            if sokoban_map.width > 15 or sokoban_map.height > 15:
                raise ValueError("Map dimensions exceed maximum allowed size of 15x15.")
            self._root.set_map(sokoban_map)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load map: {e}")
    
    def __export_map(self) -> None:
        current_map = self._root.get_map()
        if current_map is None:
            messagebox.showwarning("No Map", "There is no map to export.")
            return

        file = filedialog.asksaveasfilename(
            title="Export Sokoban Map",
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
        )

        if not file:
            return
        
        try:
            MapParser.map_to_lines(current_map)
            with open(file, "w") as f:
                for line in MapParser.map_to_lines(current_map):
                    f.write(line + "\n")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export map: {e}")
    
    def __exit(self) -> None:
        self._parent.quit()
    
    def __manual(self) -> None:
        raise NotImplementedError