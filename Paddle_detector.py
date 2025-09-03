import os
import streamlit as st
import cv2
import matplotlib.pyplot as plt
import numpy as np
from paddleocr import TextDetection
import random
from shapely.geometry import Polygon
from shapely.ops import unary_union


class TextDetector:
    def __init__(self, model_name="PP-OCRv5_server_det"):
        """Инициализация детектора текста"""
        self.model = TextDetection(model_name=model_name)
        self.image = None
        self.detections = None

    def load_image(self, img_path):
        """Загрузка и подготовка изображения"""

        if isinstance(img_path, str):
            # Загрузка из файла
            image_bgr = cv2.imread(img_path)
        else:
            # PIL Image в OpenCV
            image_rgb = np.array(img_path.convert('RGB'))
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        self.image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return self.image

    def detect_text(self, img_path, batch_size=1):
        """Детекция текста на изображении"""
        self.load_image(img_path)
        output = self.model.predict(self.image, batch_size=batch_size)

        self.detections = {
            'dt_polys': np.array(output[0]['dt_polys']),
            'dt_scores': np.array(output[0]['dt_scores'])
        }

        return self.detections

    def generate_random_color(self):
        """Генерация случайного цвета в формате RGB"""
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    def draw_filled_polygons(self, alpha=0.5):
        """Отрисовка закрашенных полигонов с разными цветами"""
        if self.image is None or self.detections is None:
            raise ValueError("Сначала выполните детекцию текста!")

        # Создаем копию изображения для рисования
        result_image = self.image.copy()

        # Создаем маску для закрашивания
        overlay = result_image.copy()

        for i, (poly, score) in enumerate(zip(self.detections['dt_polys'], self.detections['dt_scores'])):
            poly = poly.astype(int)

            # Генерируем случайный цвет для каждого полигона
            color = self.generate_random_color()

            # Рисуем закрашенный полигон
            cv2.fillPoly(overlay, [poly], color)

            # Рисуем контур полигона
            cv2.polylines(overlay, [poly], isClosed=True, color=(0, 0, 0), thickness=2)

            # Добавляем текст с уверенностью
            x, y = poly[0]
            cv2.putText(overlay, f"{score:.2f}", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Накладываем прозрачный слой
        cv2.addWeighted(overlay, alpha, result_image, 1 - alpha, 0, result_image)

        return result_image

    def merge_intersecting_polygons(self, threshold=0.04):
        """
        Объединяет полигоны, которые пересекаются (по IoU или просто по геометрии).

        Args:
            iou_threshold (float): минимальное значение IoU для объединения

        Returns:
            list: новый список объединённых полигонов (numpy массивы)
        """

        polygons = [Polygon(poly) for poly in self.detections['dt_polys']]
        merged = []

        # Будем объединять полигоны итеративно
        while polygons:
            base_poly = polygons.pop(0)
            has_merge = True

            # Пока находим пересечения с другими полигонами — объединяем
            while has_merge:
                has_merge = False
                new_polygons = []
                for other in polygons:
                    if base_poly.intersects(other):
                        # считаем IoU
                        # inter_area = base_poly.intersection(other).area
                        # union_area = base_poly.union(other).area
                        # iou = inter_area / union_area if union_area > 0.04 else 0
                        # metric = iou

                        area_other = other.area
                        area_base_poly = base_poly.area
                        inter_area = base_poly.intersection(other).area
                        metric = (inter_area/min(area_other, area_base_poly))

                        if metric >= threshold:
                            base_poly = base_poly.union(other)  # объединяем
                            has_merge = True
                        else:
                            new_polygons.append(other)
                    else:
                        new_polygons.append(other)
                polygons = new_polygons

            merged.append(base_poly)

        # Обновим детекции полигонами после объединения
        self.detections['dt_polys'] = [np.array(poly.exterior.coords, dtype=np.int32) for poly in merged]
        self.detections['dt_scores'] = np.ones(len(merged))  # можно поставить 1 или усреднить старые

        return self.detections['dt_polys']

    def extract_polygons(self):
        """
        Группирует полигоны по строкам текста (по Y-координате).
        Если средний Y нового полигона попадает в диапазон предыдущего,
        то он считается частью той же строки.

        Returns:
            dict: {"text_0": [crop1, crop2, ...], "text_1": [...], ...}
        """

        polys = self.detections['dt_polys']

        # сортируем по Y (сверху вниз)
        polys = sorted(polys, key=lambda poly: np.mean(poly[:, 1]))

        lines = {}
        current_line = []
        line_idx = 0

        # границы текущей строки
        _, y0 = polys[0].min(axis=0)
        _, y1 = polys[0].max(axis=0)

        for poly in polys:
            poly = poly.astype(np.int32)

            # средний y этого полигона
            mean_y = np.mean(poly[:, 1])

            if y0 <= mean_y <= y1:
                # тот же ряд
                current_line.append(poly)
                # расширяем границы по вертикали, если нужно
                y0 = min(y0, poly[:, 1].min())
                y1 = max(y1, poly[:, 1].max())
            else:

                current_line = sorted(current_line, key=lambda p: np.mean(p[:, 0]))
                # сохраняем предыдущую строку
                crops = self._extract_from_polys(current_line)
                lines[f"text_{line_idx}"] = crops

                # начинаем новую строку
                line_idx += 1
                current_line = [poly]
                y0, y1 = poly[:, 1].min(), poly[:, 1].max()

        # последняя строка
        if current_line:
            crops = self._extract_from_polys(current_line)
            lines[f"text_{line_idx}"] = crops

        return lines

    def _extract_from_polys(self, polys):
        """Вспомогательная функция: вырезает изображения по полигонам"""
        results = []
        for poly in polys:
            mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [poly], 255)

            out = cv2.bitwise_and(self.image, self.image, mask=mask)
            white_bg = np.full_like(self.image, 255, dtype=np.uint8)
            out = np.where(mask[..., None] == 255, out, white_bg)

            x, y, w, h = cv2.boundingRect(poly)
            cropped = out[y:y + h, x:x + w]
            results.append(cropped)
        return results

    def visualize_detections(self, filled=True, alpha=0.5):
        """Визуализация детекций с отображением в Streamlit"""
        if filled:
            result_image = self.draw_filled_polygons(alpha)
        else:
            result_image = self.image.copy()
            for poly, score in zip(self.detections['dt_polys'], self.detections['dt_scores']):
                poly = poly.astype(int)
                pts = poly.reshape((-1, 1, 2))
                cv2.polylines(result_image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                x, y = poly[0]
                cv2.putText(result_image, f"{score:.2f}", (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Конвертируем BGR в RGB для корректного отображения в Streamlit
        if len(result_image.shape) == 3 and result_image.shape[2] == 3:
            result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        else:
            result_image_rgb = result_image

        st.image(result_image_rgb, caption="Загруженное изображение", use_container_width=True)

        return result_image

def main(img_path, threshold_metric = 0.04):
    detector = TextDetector()

    detections = detector.detect_text(img_path)

    merged_poly = detector.merge_intersecting_polygons(threshold = threshold_metric)

    # Визуализируем с закрашенными областями
    detector.visualize_detections(filled=True, alpha=0.6)

    crops = detector.extract_polygons()

    return crops

# Пример использования
# if __name__ == "__main__":
#     main('/home/pret/Downloads/Pasted image (4).png')