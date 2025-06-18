import tkinter as tk
import subprocess
import os

# Define script execution functions
def run_script(script_path):
    try:
        subprocess.run(["python", script_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running script: {e}")

# Callbacks
def detect_options():
    clear_frame()
    tk.Button(root, text="Words", command=lambda: run_script("D:/FinalProject(GestureBridge)/Gesture Bridge/Hagrid/Implementation.py")).pack(pady=10)
    tk.Button(root, text="Alphabets", command=lambda: run_script("D:/FinalProject(GestureBridge)/Gesture Bridge/button.py")).pack(pady=10)
    tk.Button(root, text="Back", command=init_main_window).pack(pady=10)

def draw():
    run_script("D:/FinalProject(GestureBridge)/Gesture Bridge/AirCanvas/aircanvas/src/main.py")

# UI Initialization
def init_main_window():
    clear_frame()
    tk.Button(root, text="Detect", command=detect_options, width=20).pack(pady=10)
    tk.Button(root, text="Draw", command=draw, width=20).pack(pady=10)

def clear_frame():
    for widget in root.winfo_children():
        widget.destroy()

# Create main window
root = tk.Tk()
root.title("Sign Language Interface")
root.geometry("300x200")

init_main_window()
root.mainloop()
