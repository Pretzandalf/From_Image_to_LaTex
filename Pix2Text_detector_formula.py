from pix2text import MathFormulaDetector
import cv2
from PIL import Image
import numpy as np


def initialize_mfd_detector(model_name='mfd'):
    """
    Инициализация детектора формул через Pix2Text

    Args:
        model_name: название модели ('mfd', 'mfd-advanced-1.5', 'mfd-pro-1.5')

    Returns:
        MathFormulaDetector: инициализированный детектор
    """
    config = {
        'model_name': model_name,
        'model_backend': 'onnx',  # Используем ONNX для лучшей производительности
        'resized_shape': 768,  # Рекомендуемый размер для обработки
    }

    detector = MathFormulaDetector(**config)
    return detector


def detect_formulas_with_pix2text(image_path, detector, confidence_threshold=0.5):
    """
    Детектирование формул с использованием Pix2Text

    Args:
        image_path: путь к изображению
        detector: инициализированный детектор
        confidence_threshold: порог уверенности

    Returns:
        list: список обнаруженных формул с координатами и уверенностью
    """
    # Загрузка изображения в формате, ожидаемом Pix2Text
    if type(image_path) is str:
        image = cv2.imread(image_path)
    else:
        image = image_path

    # Детектирование формул
    detection_results = detector.detect(image)

    # Фильтрация результатов по порогу уверенности
    filtered_results = [
        result for result in detection_results
        if result['score'] >= confidence_threshold
    ]

    return filtered_results


def visualize_detection_results(image_path, detections):
    """
    Визуализация результатов детектирования

    Args:
        image_path: путь к исходному изображению
        detections: результаты детектирования
    """
    # Чтение изображения
    if type(image_path) is str:
        image = cv2.imread(image_path)
    else:
        image = image_path

    # Рисуем bounding boxes для каждой обнаруженной формулы
    for i, detection in enumerate(detections):
        bbox = detection['box']  # Это numpy array формы (4, 2)
        score = detection['score']
        box_type = detection.get('type', 'unknown')

        # Получаем координаты из массива точек
        # bbox содержит 4 точки: [верхний-левый, верхний-правый, нижний-правый, нижний-левый]
        x_coords = bbox[:, 0]  # Все X координаты
        y_coords = bbox[:, 1]  # Все Y координаты

        # Находим минимальные и максимальные координаты
        x1 = int(min(x_coords))
        y1 = int(min(y_coords))
        x2 = int(max(x_coords))
        y2 = int(max(y_coords))

        # Рисуем прямоугольник вокруг формулы
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Добавляем текст с уверенностью и типом
        label = f"{box_type} {i + 1}: {score:.3f}"
        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Показываем изображение
    cv2.imshow('Detected Formulas', image)

    # Ждем нажатия клавиши
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def crop_detected_formulas(image_path, detections, output_dir="cropped_formulas"):
    """
    Обрезка обнаруженных формул в отдельные изображения

    Args:
        image_path: путь к исходному изображению
        detections: результаты детектирования
        output_dir: директория для сохранения обрезанных формул
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    image = cv2.imread(image_path)

    for i, detection in enumerate(detections):
        bbox = detection['bbox']
        x1, y1, x2, y2 = [int(coord) for coord in bbox]

        # Обрезаем область с формулой
        formula_crop = image[y1:y2, x1:x2]

        # Сохраняем обрезанное изображение
        output_path = os.path.join(output_dir, f"formula_{i + 1}_score_{detection['score']:.3f}.png")
        cv2.imwrite(output_path, formula_crop)


def main(image_path = '/Users/yaromirkhrykin/PycharmProjects/MathDetector/Images/telegram-cloud-photo-size-2-5294197420760102264-x.jpg', threshold = 0.2, detector = None):
    if not detector:
        detector = initialize_mfd_detector(model_name='mfd')

    # Детектирование формул
    detections = detect_formulas_with_pix2text(
        image_path,
        detector,
        confidence_threshold = threshold,
    )

    # Визуализация результатов
    visualize_detection_results(image_path, detections,)

    # Обрезка формул в отдельные файлы
    #crop_detected_formulas(image_path, detections, "cropped_formulas")

if __name__ == "__main__":
    main()