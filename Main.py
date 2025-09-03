from Qwen_recognition import Qwen_recognition_model
from  Paddle_detector import main as text_detector
from Qwen_final import Qwen_recognition_model as final_model
from Document_scan import main as Doc_to_scan
import streamlit as st
import multiprocessing as mp
import numpy as np
import torch
import cv2
import gc
import warnings
warnings.filterwarnings("ignore")

def clear_memory():
    # Принудительная очистка памяти
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    torch.cuda.synchronize()

# # vl промпт
VL_model_instruct = "Тебе будет передан конспект максимально точно распиши рукописный текст на русском языке с этого изображения. Пиши только текст на изображении, это очень важно. Обращай внимание на скобки и матрицы. Верни результат."

#базовая обработка
LLM_first_instruct = r"""
ы хороший ассистент, твоя задача обрабатывать текст, полученный в результате перевода фотографий конспектов по алгебре в текст. Важно оставить слова на изначальном языке (скорее всего русский язык). Ты должен переводить текст в готовый LaTex формат, а именно сделать структуру и перевести математические формулы в формат LaTex, также редактируй слова, которые неправильно были прочтены из изображения. Верни результат
"""

#финальная обработка
LLM_second_instruct = r"""
Ты — ассистент для обработки текстов, полученных из OCR фотографий конспектов по алгебре.
Всегда возвращай результат в виде корректного LaTeX-кода.
Требования: Тебе нужно исправить ошибки в написании LaTex. Сделай все важные импорты (например русского языка \usepackage[english, russian]{babel}), проверь верность написания формул. Все остальное оставь без изменений. Выведи результат
"""


def Image_to_text(image_path):
    with torch.no_grad():
        clear_memory()
        #scan_img = Doc_to_scan(image_path)
        #crops = text_detector(scan_img)
        crops = text_detector(image_path)

        clear_memory()
        recognition_model = Qwen_recognition_model()
        recognition_model.initialize_Qwen_model(VL_model_instruct)
        text_lines = [[] for _ in range(len(crops.keys()))]
        for line_num, line in enumerate(crops.keys()):
            for indx, block_img in enumerate(crops[line]):
                text_on_block = recognition_model.img_to_text(block_img)[0]
                text_lines[line_num].append(text_on_block)

        result = "\n".join("\t".join(row) for row in text_lines)

        del recognition_model
        clear_memory()

        Qwen_final_model = final_model()
        Qwen_final_model.initialize_Qwen_model()
        final_text_LaTex = Qwen_final_model.text_inference(result, LLM_first_instruct)
        final_LaTex = Qwen_final_model.text_inference(final_text_LaTex, LLM_second_instruct)

        del Qwen_final_model
        clear_memory()

        return final_LaTex


#print(Image_to_text("/home/pret/Downloads/Pasted image.png"))
