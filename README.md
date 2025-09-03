# Handwritten Notes to LaTeX Converter

## Overview
This project provides a pipeline for converting handwritten lecture notes into structured LaTeX code.  
The system combines **line detection**, **vision–language models**, and **post-correction with language models** to achieve accurate recognition of mathematical formulas and structured text.

Key features:
- Line-level text detection using **PaddleOCR**.
- Analysis of segmented blocks with a quantized **Qwen-VL model**.
- Two-stage post-processing for error correction and LaTeX syntax normalization.
- Efficient inference on a single GPU with limited memory.

------------------------------------------------------------------------------

Обзор
Данный проект реализует конвейер для преобразования рукописных заметок в структурированный код LaTeX.
Подход сочетает детекцию строк, мультимодельный анализ и двухуровневую коррекцию текста, что позволяет улучшить качество распознавания формул и структурированных записей.

Основные возможности:

- Детекция строк с помощью PaddleOCR.
- Анализ сегментированных блоков моделью Qwen-VL (AWQ).
- Постобработка для исправления ошибок и нормализации LaTeX-формул.
- Возможность запуска на одной видеокарте с ограниченными ресурсами.



👉 **Ссылка на статью:** [Paper.pdf](./Article_.pdf)
