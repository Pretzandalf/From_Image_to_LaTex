import cv2
import numpy as np
from PIL import Image
import streamlit as st


def scan_image_from_pil(pil_image, enhance_quality=True):
    """
    Сканирует и обрабатывает PIL изображение

    Args:
        pil_image (PIL.Image): PIL изображение для обработки
        enhance_quality (bool): Улучшать ли качество изображения

    Returns:
        PIL.Image: Обработанное изображение
    """

    # Конвертируем PIL в OpenCV
    image = np.array(pil_image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    print(f"Изображение загружено")
    print(f"Размер: {image.shape[1]}x{image.shape[0]} пикселей")

    # Обрабатываем изображение
    processed_image = image.copy()

    # Находим и выравниваем документ
    processed_image = detect_and_align_document(processed_image)

    if enhance_quality:
        # Улучшаем качество изображения
        processed_image = enhance_image_quality(processed_image)

    # Конвертируем обратно в PIL
    processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
    output_pil = Image.fromarray(processed_image_rgb)

    return output_pil


def detect_and_align_document(image):
    """
    Обнаруживает документ на изображении и выравнивает его
    """
    # Конвертируем в grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Применяем размытие и бинаризацию
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 200)

    # Находим контуры
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    # Ищем прямоугольный контур
    screen_cnt = None
    for contour in contours:
        # Аппроксимируем контур
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        # Если у контура 4 точки, это вероятно наш документ
        if len(approx) == 4:
            screen_cnt = approx
            break

    if screen_cnt is not None:
        # Выравниваем документ
        warped = four_point_transform(image, screen_cnt.reshape(4, 2))
        return warped
    else:
        # Если документ не найден, возвращаем оригинал
        st.warning("Не удалось обнаружить документ на изображении")
        return image


def four_point_transform(image, pts):
    """
    Выполняет перспективное преобразование для выравнивания документа
    """
    # Упорядочиваем точки
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Вычисляем ширину новой картинки
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # Вычисляем высоту новой картинки
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Создаем матрицу преобразования
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # Применяем перспективное преобразование
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


def order_points(pts):
    """
    Упорядочивает точки в порядке: верх-лево, верх-право, низ-право, низ-лево
    """
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def enhance_image_quality(image):
    """
    Улучшает качество изображения
    """
    # Конвертируем в правильное цветовое пространство
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Увеличиваем контраст
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # Применяем CLAHE для улучшения контраста
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # Уменьшаем шум
    image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

    # Резкость
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    image = cv2.filter2D(image, -1, kernel)

    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

# Streamlit интерфейс
def main(pil_image):
    processed_image = scan_image_from_pil(pil_image)
    st.image(processed_image, caption="Обработанное изображение", use_container_width=True)

    return processed_image


# if __name__ == "__main__":
#     main()
