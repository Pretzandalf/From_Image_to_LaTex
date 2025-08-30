import streamlit as st
from Main import Image_to_text
from PIL import Image
import multiprocessing as mp

# def run_inference(img_bytes):
#     from Main import Image_to_text
#     from PIL import Image
#     import io
#
#     # преобразуем байты обратно в Image
#     image = Image.open(io.BytesIO(img_bytes))
#     image = image.convert("RGB")  # важно для совместимости
#
#     text = Image_to_text(image)
#     return text

def trim_latex(text: str) -> str:
    start = text.find("\\")
    end = text.rfind("\\end{document}")
    if start == -1 or end == -1:
        return text
    return text[start:end + len("\\end{document}")]

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #ee4c2c, #f9a825);
        font-family: "Segoe UI", sans-serif;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        border-radius: 20px;
    }
    .stButton>button {
        background-color: #4A90E2;
        color: white;
        border-radius: 12px;
        padding: 0.6em 1.2em;
        font-size: 16px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #357ABD;
        transform: scale(1.05);
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    /* Увеличиваем текст спиннера */
    div[data-testid="stSpinner"] > div > div > div {
        font-size: 20px !important;  /* размер текста */
        font-weight: bold;
        color: #333;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Заголовок ---
st.title("Конвертация изображения в LaTeX")
st.markdown("Загрузите изображение, и получите текст в формате **LaTeX**.")

# --- Загрузка изображения ---
uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼️ Ваше изображение", use_container_width=True)

    # with st.spinner(" Обрабатываем изображение..."):
    #     # читаем байты для передачи в дочерний процесс
    #     img_bytes = uploaded_file.getvalue()
    #
    #     # запускаем отдельный процесс
    #     ctx = mp.get_context("spawn")
    #     with ctx.Pool(1) as pool:
    #         text = pool.apply(run_inference, (img_bytes,))
    #
    #     text = trim_latex(text)

    with st.spinner("🔄 Обрабатываем изображение..."):
        text = Image_to_text(image)
        text = trim_latex(text)

    # --- Вывод результата ---
    st.success("✅ Готово! Ваш LaTeX код ниже:")

    st.text_area("Результат в LaTeX", text, height=1000)

    # try:
    #     st.latex(text)
    # except:
    #     st.info("⚠️ Не удалось отрендерить LaTeX, но текст доступен выше.")

    # --- Скачивание ---
    st.download_button(
        label="📥 Скачать результат (.txt)",
        data=text,
        file_name="result_text_Latex.txt",
    )
