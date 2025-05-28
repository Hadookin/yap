import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2


class ImageProcessorApp:
    def __init__(self, master):
        self.master = master
        master.title("Image Processor")

        self.label = tk.Label(master, text="Выберите изображение для обработки:")
        self.label.pack()

        self.select_button = tk.Button(master, text="Выбрать изображение", command=self.select_image)
        self.select_button.pack()

        self.detect_shapes_button = tk.Button(master, text="Обнаружить формы", command=self.detect_shapes)
        self.detect_shapes_button.pack()

        self.image_label = tk.Label(master)
        self.image_label.pack()

        self.image_path = ""
        self.processed_image = None

    def select_image(self):
        self.image_path = filedialog.askopenfilename(title="Выберите изображение",
                                                     filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")])
        if self.image_path:
            print(f"Выбрано изображение: {self.image_path}")
            self.show_image(self.image_path)

    def show_image(self, path):
        image = cv2.imread(path)
        if image is None:
            messagebox.showerror("Ошибка", "Не удалось загрузить изображение. Проверьте путь.")
            return

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)

        self.image_label.config(image=image)
        self.image_label.image = image

    def detect_shapes(self):
        if not self.image_path:
            messagebox.showwarning("Предупреждение", "Пожалуйста, выберите изображение сначала.")
            return

        image = cv2.imread(self.image_path)
        if image is None:
            messagebox.showerror("Ошибка", "Не удалось загрузить изображение. Проверьте путь.")
            return

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            epsilon = 0.04 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) == 3:
                shape = "Triangle"
            elif len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(w) / h
                if 0.95 <= aspect_ratio <= 1.05:
                    shape = "Square"
                else:
                    shape = "Rectangle"
            elif len(approx) == 5:
                shape = "Pentagon"
            elif len(approx) == 6:
                shape = "Hexagon"
            else:
                shape = "Circle"

            cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(image, shape, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        self.processed_image = image
        self.show_detected_image(image)

    def show_detected_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)

        self.image_label.config(image=image)
        self.image_label.image = image


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()
