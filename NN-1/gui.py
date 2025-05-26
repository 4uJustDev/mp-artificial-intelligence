import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from PIL import Image, ImageTk
import numpy as np
import os
from train import train_perceptron
import sys
from io import StringIO


class ConsoleRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.buffer = StringIO()
        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr
        sys.stdout = self
        sys.stderr = self

    def write(self, text):
        self.buffer.write(text)
        self.text_widget.configure(state="normal")
        self.text_widget.insert(tk.END, text)
        self.text_widget.see(tk.END)
        self.text_widget.configure(state="disabled")

    def flush(self):
        pass

    def restore(self):
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr


class ImageRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Распознавание фигур")
        self.root.geometry("1400x800")

        # Configure grid weights for root window
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # Создаем персептрон
        self.perceptron = None
        self.input_size = None

        # Создаем основной контейнер
        self.main_container = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        self.main_container.grid(row=0, column=0, sticky="nsew")

        # Левая панель (дерево файлов)
        self.left_frame = ttk.Frame(self.main_container, width=400)
        self.left_frame.pack_propagate(False)  # Prevent frame from shrinking
        self.main_container.add(
            self.left_frame, weight=0
        )  # Set weight to 0 for fixed width

        # Центральная панель (изображение)
        self.center_frame = ttk.Frame(self.main_container)
        self.main_container.add(self.center_frame, weight=2)

        # Configure center frame grid
        self.center_frame.grid_rowconfigure(0, weight=1)
        self.center_frame.grid_columnconfigure(0, weight=1)

        # Метка для отображения изображения
        self.image_label = ttk.Label(self.center_frame)
        self.image_label.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # Правая панель (результаты)
        self.right_frame = ttk.Frame(self.main_container)
        self.main_container.add(self.right_frame, weight=1)

        # Configure right frame grid
        self.right_frame.grid_rowconfigure(0, weight=1)
        self.right_frame.grid_columnconfigure(0, weight=1)

        # Текстовое поле для результатов с прокруткой
        self.result_text = scrolledtext.ScrolledText(self.right_frame, wrap=tk.WORD)
        self.result_text.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # Настройка левой панели
        self.setup_left_panel()

        # Загрузка изображений
        self.load_images()

        # Запускаем обучение при старте
        self.start_initial_training()

    def setup_left_panel(self):
        # Создаем фрейм для кнопки
        button_frame = ttk.Frame(self.left_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=5)

        # Добавляем кнопку распознавания
        ttk.Button(button_frame, text="Распознать", command=self.recognize_image).pack(
            fill=tk.X, pady=2
        )

        # Создаем фрейм для дерева файлов
        tree_frame = ttk.Frame(self.left_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Создаем стиль для дерева
        style = ttk.Style()
        style.configure(
            "Treeview",
            background="#f0f0f0",
            foreground="black",
            rowheight=25,
            fieldbackground="#f0f0f0",
        )
        style.configure(
            "Treeview.Heading", background="#e0e0e0", foreground="black", relief="flat"
        )
        style.map("Treeview.Heading", background=[("active", "#d0d0d0")])

        # Создаем дерево файлов
        self.recognition_tree = ttk.Treeview(tree_frame, style="Treeview")
        self.recognition_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Добавляем скроллбар для дерева
        tree_scroll = ttk.Scrollbar(
            tree_frame, orient="vertical", command=self.recognition_tree.yview
        )
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.recognition_tree.configure(yscrollcommand=tree_scroll.set)

        # Настраиваем колонки
        self.recognition_tree["columns"] = ("path",)
        self.recognition_tree.column("#0", width=150, minwidth=150)
        self.recognition_tree.column("path", width=100, minwidth=100)
        self.recognition_tree.heading("#0", text="Файлы", anchor=tk.W)
        self.recognition_tree.heading("path", text="Путь", anchor=tk.W)

        # Привязка события выбора файла
        self.recognition_tree.bind("<<TreeviewSelect>>", self.on_recognition_select)

        # Перенаправляем вывод в консоль результатов
        self.recognition_redirector = ConsoleRedirector(self.result_text)

    def load_images(self):
        # Очищаем дерево
        for item in self.recognition_tree.get_children():
            self.recognition_tree.delete(item)

        # Загружаем изображения из папки images
        images_dir = "images"
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)

        # Получаем список всех файлов
        files = os.listdir(images_dir)

        # Фильтруем файлы, исключая обучающие
        test_files = [
            f
            for f in files
            if not any(
                f.lower().startswith(prefix)
                for prefix in ["circle_", "square_", "triangle_", "rectangle_"]
            )
        ]

        # Сортируем файлы по имени
        test_files.sort()

        for filename in test_files:
            if filename.lower().endswith(".bmp"):
                file_path = os.path.join(images_dir, filename)
                self.recognition_tree.insert(
                    "", "end", text=filename, values=(file_path,), tags=("file",)
                )

    def process_image(self, image):
        """Обработка изображения и определение размера входного слоя"""
        # Преобразуем в черно-белое
        image = image.convert("L")
        # Получаем размеры
        width, height = image.size
        # Преобразуем в массив
        img_array = np.array(image)
        # Нормализуем значения
        img_array = (img_array > 128).astype(np.float32)
        # Преобразуем в одномерный массив
        img_array = img_array.flatten()

        # Если размер входного слоя еще не установлен, устанавливаем его
        if self.input_size is None:
            self.input_size = len(img_array)

        return img_array

    def on_recognition_select(self, event):
        selected_items = self.recognition_tree.selection()
        if not selected_items:
            return

        file_path = self.recognition_tree.item(selected_items[0])["values"][0]
        self.display_image(file_path)

    def display_image(self, file_path):
        try:
            image = Image.open(file_path)
            # Получаем размеры фрейма
            frame_width = self.center_frame.winfo_width()
            frame_height = self.center_frame.winfo_height()

            # Вычисляем размеры для отображения с сохранением пропорций
            img_width, img_height = image.size
            ratio = min(frame_width / img_width, frame_height / img_height)
            display_size = (int(img_width * ratio), int(img_height * ratio))

            display_image = image.resize(display_size, Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(display_image)
            self.image_label.configure(image=photo)
            self.image_label.image = photo
            self.current_image = image
        except Exception as e:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Ошибка загрузки изображения: {str(e)}")

    def recognize_image(self):
        if not self.perceptron:
            messagebox.showwarning("Предупреждение", "Персептрон не обучен")
            return

        if not self.current_image:
            messagebox.showwarning("Предупреждение", "Сначала выберите изображение")
            return

        try:
            # Обрабатываем изображение
            img_array = self.process_image(self.current_image)

            # Проверяем размер входного слоя
            if len(img_array) != self.input_size:
                messagebox.showerror(
                    "Ошибка",
                    f"Размер изображения ({len(img_array)}) не соответствует размеру входного слоя ({self.input_size})",
                )
                return

            # Распознаем изображение с деталями
            result = self.perceptron.predict(img_array, return_details=True)

            # Выводим результат
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "Результаты распознавания:\n\n")

            shapes = ["Круг", "Квадрат", "Треугольник", "Прямоугольник"]
            self.result_text.insert(
                tk.END, f"Распознанная фигура: {shapes[result['predicted_class']]}\n\n"
            )

            self.result_text.insert(tk.END, "Выходы нейронов до активации:\n")
            self.result_text.insert(tk.END, str(result["raw_output"]))

            self.result_text.insert(tk.END, "\n\nВыходы нейронов после активации:\n")
            self.result_text.insert(tk.END, str(result["activated_output"]))

            self.result_text.insert(tk.END, "\n\nТекущие веса:\n")
            self.result_text.insert(tk.END, str(self.perceptron.weights))

            self.result_text.insert(tk.END, "\n\nТекущие смещения:\n")
            self.result_text.insert(tk.END, str(self.perceptron.bias))

        except Exception as e:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Ошибка распознавания: {str(e)}")

    def start_initial_training(self):
        """Запуск начального обучения"""
        # Восстанавливаем стандартный вывод
        self.recognition_redirector.restore()

        # Запускаем обучение
        self.perceptron = train_perceptron()

        # Перенаправляем вывод обратно в консоль
        self.recognition_redirector = ConsoleRedirector(self.result_text)

        if self.perceptron:
            self.input_size = self.perceptron.input_size


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageRecognitionApp(root)
    root.mainloop()
