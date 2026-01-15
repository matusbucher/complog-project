import tkinter as tk
from tkinter import messagebox
from typing import Optional

from logic.logic_program_interface import Solution
from logic.planner import Planner


TEXT_HEIGHT = 22
TEXT_WIDTH = 30


class SolutionPanel:
    def __init__(self, root, parent):
        self._root = root
        frame = tk.Frame(parent, bg="lightgray", bd=2, relief=tk.SUNKEN)
        frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        title = tk.Label(frame, text="Solution Plan", bg="lightgray", font=("Arial", 12, "bold"))
        title.pack(pady=5)
        
        self._content = tk.Text(frame, height=TEXT_HEIGHT, width=TEXT_WIDTH, bg="white", state=tk.DISABLED)
        self._content.pack(fill=tk.BOTH, padx=5, pady=5)
        self._content.tag_configure("highlight", background="yellow")

        control_frame = tk.Frame(frame, bg="lightgray")
        control_frame.pack(pady=5)

        step_back_button = tk.Button(control_frame, text="< Step Back", bg="lightblue", font=("Arial", 10, "bold"), command=self.__step_back)
        step_back_button.grid(row=0, column=0, padx=5, pady=5)

        step_next_button = tk.Button(control_frame, text="Step Next >", bg="lightblue", font=("Arial", 10, "bold"), command=self.__step_next)
        step_next_button.grid(row=0, column=1, padx=5, pady=5)

        find_solution_button = tk.Button(control_frame, text="Find Solution", bg="lightgreen", font=("Arial", 12, "bold"), command=self.__find_solution, height=1, width=17)
        find_solution_button.grid(row=1, column=0, columnspan=2, padx=5, pady=5)

        self._shortest_solution_var = tk.BooleanVar(value=True)
        shortest_solution_checkbox = tk.Checkbutton(control_frame, text="Shortest Solution", variable=self._shortest_solution_var, bg="lightgray", font=("Arial", 10))
        shortest_solution_checkbox.grid(row=2, column=0, columnspan=2, padx=5, pady=5)

        self._max_steps_var = tk.IntVar(value=20)
        max_steps_label = tk.Label(control_frame, text="Max Steps:", bg="lightgray", font=("Arial", 10))
        max_steps_label.grid(row=3, column=0, padx=5, pady=2, sticky="e")
        max_steps_entry = tk.Entry(control_frame, textvariable=self._max_steps_var, width=5, font=("Arial", 10))
        max_steps_entry.grid(row=3, column=1, padx=5, pady=2, sticky="w")
        max_steps_slider = tk.Scale(control_frame, from_=1, to=100, orient=tk.HORIZONTAL, length=150, bg="lightgray", variable=self._max_steps_var)
        max_steps_slider.grid(row=4, column=0, columnspan=2, padx=5, pady=2)

        self._solution : Optional[Solution] = None
        self._current_step : int = 0
        self._planner : Optional[Planner] = None

    def reset(self) -> None:
        self._solution = None
        self._current_step = 0
        self._planner = None
        self._content.config(state=tk.NORMAL)
        self._content.delete(1.0, tk.END)
        self._content.config(state=tk.DISABLED)

    def __step_next(self) -> None:
        if self._solution is None or self._current_step >= len(self._solution):
            return
        self._root.execute_step(self._solution[self._current_step], inverse=False)
        self._current_step += 1
        self.__highlight_step(self._current_step)
        
    def __step_back(self) -> None:
        if self._solution is None or self._current_step <= 0:
            return
        self._current_step -= 1
        self._root.execute_step(self._solution[self._current_step], inverse=True)
        self.__highlight_step(self._current_step)

    def __find_solution(self) -> None:
        sokoban_map = self._root.get_map()
        if sokoban_map is None:
            messagebox.showerror("Error", "No map ready.")
            return
        
        if not sokoban_map.valid_map():
            messagebox.showerror("Error", "The map is not valid. Ensure there is exactly one Sokoban.")
            return

        settings = self._root.get_settings()
        if not settings.complete():
            self._root.open_settings_dialog()
            return
            
        self._root.disable_editing()
        self.reset()
        
        try:
            self._planner = Planner(
                sokoban_map=sokoban_map,
                max_steps=self._max_steps_var.get(),
                solver_path=settings.minisat_binary_file,
                solver_input=f"{settings.minisat_files_folder}/input.cnf",
                solver_output=f"{settings.minisat_files_folder}/output.cnf",
                optimize=settings.optimizations_enabled
            )

            if self._shortest_solution_var.get():
                self._solution = self._planner.find_shortest_solution()
            else:
                self._solution = self._planner.find_any_solution()
        except Exception as e:
            messagebox.showerror("Error", f"{e}")
            self._root.enable_editing()
            return
        
        if self._solution is None:
            messagebox.showinfo("No Solution", "No solution found within the given step limit.")
            self._root.enable_editing()
            return
        
        self.__write_solution()
        self.__highlight_step(self._current_step)
        
    def __write_solution(self) -> None:
        self._content.config(state=tk.NORMAL)
        self._content.insert(tk.END, "0: INIT STATE\n")
        for step in self._solution:
            self._content.insert(tk.END, step.to_string() + "\n")
        self._content.config(state=tk.DISABLED)

    def __highlight_step(self, step_index: int) -> None:
        self._content.config(state=tk.NORMAL)
        self._content.tag_remove("highlight", "1.0", tk.END)
        line_start = f"{step_index + 1}.0"
        line_end = f"{step_index + 1}.end"
        self._content.tag_add("highlight", line_start, line_end)
        self._content.see(line_start)
        self._content.config(state=tk.DISABLED)