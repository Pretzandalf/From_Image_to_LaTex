# Handwritten Notes to LaTeX Converter

## Overview
This project provides a pipeline for converting handwritten lecture notes into structured LaTeX code.  
The system combines **line detection**, **vision–language models**, and **post-correction with language models** to achieve accurate recognition of mathematical formulas and structured text.

Key features:
- Line-level text detection using **PaddleOCR**.
- Analysis of segmented blocks with a quantized **Qwen-VL model**.
- Two-stage post-processing for error correction and LaTeX syntax normalization.
- Efficient inference on a single GPU with limited memory.


## How to Inference

To try the project locally:

```bash
# 1. Clone the repository
git clone https://github.com/Pretzandalf/From_Image_to_LaTex.git

# 2. Go to the project folder
cd From_Image_to_LaTex

# 3. Run the Streamlit interface
streamlit run streamlit_.py
```

👉 **Link to paper:** [Paper.pdf](./Article_.pdf)
------------------------------------------------------------------------------

Обзор
Данный проект реализует конвейер для преобразования рукописных заметок в структурированный код LaTeX.
Подход сочетает детекцию строк, мультимодельный анализ и двухуровневую коррекцию текста, что позволяет улучшить качество распознавания формул и структурированных записей.

Основные возможности:

- Детекция строк с помощью PaddleOCR.
- Анализ сегментированных блоков моделью Qwen-VL (AWQ).
- Постобработка для исправления ошибок и нормализации LaTeX-формул.
- Возможность запуска на одной видеокарте с ограниченными ресурсами.


```markdown
## Инференс
```
Чтобы запустить проект локально:

```bash
# 1. Склонируйте репозиторий
git clone https://github.com/Pretzandalf/From_Image_to_LaTex.git

# 2. Перейдите в папку проекта
cd From_Image_to_LaTex

# 3. Запустите интерфейс Streamlit
streamlit run streamlit_.py
```

👉 **Ссылка на статью:** [Paper.pdf](./Article_.pdf)
