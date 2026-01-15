"""
GUI planner for Sokoban game solutions. Provides a graphical interface for editing Sokoban maps (together with import/export), founding solutions via an external MiniSat solver, and step-by-step visualization of the found solutions.

Dependencies:
- Python 3.7+
- Pillow library
- NumPy library
- An external MiniSat solver binary
"""

import tkinter as tk

from gui.root_window import RootWindow


if __name__ == '__main__':
    root = tk.Tk()
    app = RootWindow(root)
    root.mainloop()