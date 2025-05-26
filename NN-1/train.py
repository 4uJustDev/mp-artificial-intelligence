import os
import numpy as np
from PIL import Image
from perceptron import Perceptron


def load_training_images():
    """Загрузка обучающих изображений из папки images"""
    training_data = []
    training_labels = []
    input_size = None

    # Словарь соответствия префиксов и меток классов
    class_mapping = {"circle": 0, "square": 1, "triangle": 2, "rectangle": 3}

    # Загружаем изображения из папки images
    images_dir = "images"
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
        print("Создана папка images. Пожалуйста, добавьте обучающие изображения.")
        return None, None, None

    # Получаем список всех файлов
    files = os.listdir(images_dir)

    # Фильтруем только файлы с нужными префиксами
    training_files = [
        f
        for f in files
        if any(f.lower().startswith(prefix) for prefix in class_mapping.keys())
    ]

    if not training_files:
        print(
            "Обучающие изображения не найдены. Пожалуйста, добавьте изображения с префиксами: circle_, square_, triangle_, rectangle_"
        )
        return None, None, None

    # Загружаем и обрабатываем изображения
    for filename in training_files:
        # Определяем класс по префиксу
        prefix = next(p for p in class_mapping.keys() if filename.lower().startswith(p))
        label = class_mapping[prefix]

        # Загружаем и обрабатываем изображение
        try:
            image_path = os.path.join(images_dir, filename)
            image = Image.open(image_path)
            image = image.convert("L")  # Преобразуем в черно-белое

            # Преобразуем в массив и нормализуем
            img_array = np.array(image)
            img_array = (img_array > 128).astype(np.float32)
            img_array = img_array.flatten()

            # Устанавливаем размер входного слоя при первой загрузке
            if input_size is None:
                input_size = len(img_array)
            elif len(img_array) != input_size:
                print(
                    f"Ошибка: размер изображения {filename} не соответствует размеру других изображений"
                )
                continue

            training_data.append(img_array)
            training_labels.append(label)
            print(f"Загружено изображение: {filename} (класс: {prefix})")

        except Exception as e:
            print(f"Ошибка при загрузке {filename}: {str(e)}")
            continue

    if not training_data:
        print("Не удалось загрузить ни одного изображения для обучения")
        return None, None, None

    return np.array(training_data), np.array(training_labels), input_size


def train_perceptron():
    """Обучение персептрона на предопределенных изображениях"""
    print("Начало обучения персептрона...")

    # Загружаем обучающие данные
    training_data, training_labels, input_size = load_training_images()
    if training_data is None:
        return None

    # Параметры обучения
    num_classes = 4
    learning_rate = 0.1
    max_epochs = 1000
    error_threshold = 0.001

    print(f"\nПараметры обучения:")
    print(f"Размер входного слоя: {input_size}")
    print(f"Количество классов: {num_classes}")
    print(f"Скорость обучения: {learning_rate}")
    print(f"Максимальное количество эпох: {max_epochs}")
    print(f"Порог ошибки: {error_threshold}\n")

    # Создаем и обучаем персептрон
    perceptron = Perceptron(input_size, num_classes, learning_rate)
    history = perceptron.train(
        training_data, training_labels, max_epochs, error_threshold
    )

    # Выводим результаты обучения
    print("\nРезультаты обучения:")
    for epoch in history:
        print(f"Эпоха {epoch['epoch']}, Ошибка: {epoch['total_error']:.6f}")

    return perceptron


if __name__ == "__main__":
    train_perceptron()
