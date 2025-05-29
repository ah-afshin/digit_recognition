import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

from PIL import Image, ImageDraw, ImageOps
import torch as t
import numpy as np

from utils import load_model


# GUI constants
CANVAS_SIZE = 280
IMG_SIZE = 28


def preprocess(canvas_image):
    # Convert to grayscale, resize, and invert
    image = canvas_image.convert('L')
    image = ImageOps.invert(image)
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image) / 255.0
    tensor = t.tensor(image, dtype=t.float32).unsqueeze(0).unsqueeze(0)  # [1, 1, 28, 28]
    return tensor


class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Digit Recognizer")

        self.canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg='white')
        self.canvas.grid(row=0, column=0, columnspan=4)

        self.canvas.bind('<B1-Motion>', self.draw)

        self.image = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), "white")
        self.draw_obj = ImageDraw.Draw(self.image)

        tk.Button(root, text="Clear", command=self.clear).grid(row=1, column=0)
        tk.Button(root, text="Load Model", command=self.select_model).grid(row=1, column=1)
        tk.Button(root, text="Predict", command=self.predict).grid(row=1, column=2)
        self.result_label = tk.Label(root, text="Prediction: ?")
        self.result_label.grid(row=1, column=3)

        self.model = None
        self.device = "cuda" if t.cuda.is_available() else "cpu"

    def draw(self, event):
        x, y = event.x, event.y
        r = 8
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill='black')
        self.draw_obj.ellipse([x - r, y - r, x + r, y + r], fill='black')

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), "white")
        self.draw_obj = ImageDraw.Draw(self.image)
        self.result_label.config(text="Prediction: ?")

    def select_model(self):
        path = filedialog.askopenfilename(filetypes=[("PyTorch model", "*.pth")])
        if path:
            try:
                self.model = load_model(path, self.device)
                messagebox.showinfo("Model Loaded", f"Model loaded from:\n{path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model:\n{e}")

    def predict(self):
        if not self.model:
            messagebox.showwarning("No Model", "Please load a model first.")
            return

        tensor = preprocess(self.image).to(self.device)
        with t.inference_mode():
            output = self.model(tensor.view(1, -1))
            prob = t.softmax(output, dim=1)
            pred = prob.argmax().item()

        self.result_label.config(text=f"Prediction: {pred}")


def run_ui():
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()
