from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import gc


class Qwen_recognition_model():
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.prompt = None

    def initialize_Qwen_model(self, ):
        """
        Инициализация Qwen2.5-7B-Instruct-AWQ (только текстовая версия).
        """
        self.model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-7B-Instruct-AWQ",
            dtype=torch.float16,
            device_map="cuda"
        )

        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct-AWQ")

    def text_inference(self, text_input: str, prompt : str, max_new_tokens: int = 4096) -> str:
        """
        Генерация текста на основе prompt + входа пользователя.
        """

        # Формируем сообщение в формате chat template
        messages = [
            {
                "role": "system",
                "content": (
                    prompt
                )
            },
            {
                "role": "user",
                "content": f"\n\n{text_input}"
            }
        ]

        # Токенизация
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to("cuda")

        # Генерация
        generated_ids = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

        # Убираем токены входа
        generated_ids_trimmed = generated_ids[:, input_ids.shape[-1]:]

        # Декодирование
        output_text = self.tokenizer.decode(
            generated_ids_trimmed[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        return output_text


if __name__ == "__main__":
    # text_input = конкретный запрос

    instruct = r"""
    **Role:** You are an expert academic assistant specialized in LaTeX typesetting for advanced mathematics, particularly algebra and linear algebra. Your sole purpose is to correct LaTeX syntax errors and ensure perfect formatting.
    
    **Instruction:**
    You will receive text extracted from algebra/linear algebra lecture notes. It is a mix of natural language (Russian/English) and LaTeX code for mathematical expressions.
    
    1.  **Focus Exclusively on LaTeX:** Only analyze text within LaTeX delimiters: `$...$`, `$$...$$`, `\(...\)`, `\[...\]`, `\begin{env}...\end{env}`. Ignore all natural language text completely; do not correct spelling, grammar, or meaning outside of LaTeX.
    
    2.  **Ensure Essential Imports:** The final LaTeX code must be compilable with standard mathematical packages. **You must assume the following packages are always imported in the preamble** and use their commands correctly:
        *   `\usepackage{amsmath}` (for `\matrix`, `\begin{pmatrix}`, `\operatorname`, `\det`, `\tr`, etc.)
        *   `\usepackage{amssymb}` (for advanced symbols like `\mathbb`, `\mathfrak`, `\mathcal`)
        *   `\usepackage{amsfonts}` (for additional fonts)
        *   `\usepackage{bm}` (for bold math symbols `\bm`)
    
        **Correct commands to use these packages' features.** For example:
        *   `det(A)` -> `\det(A)`
        *   `tr(A)` -> `\operatorname{tr}(A)` or `\tr(A)` if `\tr` is defined.
        *   `R` (for real numbers) -> `\mathbb{R}`
        *   `bold{A}` -> `\bm{A}` or `\mathbf{A}`
    
    3.  **Correct LaTeX Errors:** Identify and fix common LaTeX mistakes within these delimiters:
        *   **Missing/Broken Braces:** `a^2 + b^2` -> `a^{2} + b^{2}`
        *   **Incorrect or Missing Commands:** `\aalpha` -> `\alpha`, `matrix` -> `\begin{matrix}`, `→` -> `\rightarrow`
        *   **Missing Backslashes:** `alpha` -> `\alpha`, `rightarrow` -> `\rightarrow`
        *   **Spacing Issues:** Add `\,`, `\:`, or `\;` for correct spacing (e.g., `dx` -> `\,dx`).
        *   **Environment Syntax:** Ensure `\begin{}` and `\end{}` tags are correctly paired and named.
        *   **Unescaped Symbols:** Correct usage of `%`, `&`, `_`, `#` inside and outside math blocks.
    
    4.  **Preserve Formatting & Content:** DO NOT change the overall structure, text, or meaning of the input. Your output must be identical to the input, with **only** the LaTeX code inside the delimiters corrected.
    
    5.  **Output Format:** Return the entire text exactly as received, with only the necessary LaTeX corrections applied. Do not add any introductory text, explanations, or markdown formatting like ```. Do not output the preamble with `\usepackage`. Only correct the inline LaTeX commands.
    
    **Input to Process:**
        """


    text_input = r'''D3-1.
    1	m+n=3mn-3m-3n+4 (1)
    Бинарические операции на M - это отображение o: M x M -> M. Значит, достаточно проверить, что там переводится элементы из R \ {1} x R \ {1} в R \ {1}.
    Умакс, ∫ m ∈ R \ {1} i.e. |m ≠ 2 n ∈ R \ {1}
    Докажем, что \((3mn - 3m - 3n + 4) \in \mathbb{R} \setminus \{m\}\).
    Так как m,n ∈ IR \{0}, то выражение (4) не может выйти за рамки вещественных чисел.
    ⇒ (4) ∈ IR, остается показать, что	(х) ≠ 1.
    Допустим обратное: (y) = 1
    3mn - 3m - 3n + 4 = 1
    3(mn - m - n) = -3
    mn - m - n = -1
    m(n-1) - (n-1) = 0
    (n-1)(m-1) = 0
    \[
    \begin{array}{l}
    n = 1 \\
    m = 1 \quad \text{Противоречие, т.к.} \quad m \neq 1 \quad \Rightarrow \quad (x) \neq 1
    \end{array}
    \]
    \(\Rightarrow\) мон-бинарная операция из множества \(\mathbb{R} \backslash \{0\}\)
    Докажем теперь, что (R \{1\}, o) - группа
    Проверить выполнимость ассоциации групп:
    1) Ассимилятивность
    (a○b)○c = a○(b○c)
    1	2
    (аоб)с = 3(3ab - 3a - 3b + 4)с - 3(3ab - 3a - 3b + 4) - 3с + 4 =
    = gabe - gac - gbe + gac - gab + ga + gb - 12 - 3c + 4 =
    = gabc - g(ac + bc + ab) + gc + ga + gb - 8
    a·(b·c) = 3a (3bc - 3b - 3c + 4) - 3a - 3 (3bc - 3b - 3c + 4) + 4
    = gabc - gab - gac + ga - gbc - gb - gc - 12 + 4 =
    = gabc - g(ac + bc + ab) + gc + ga + gb - 8
                    '''

    model = Qwen_recognition_model()
    model.initialize_Qwen_model()
    print(model.text_inference(text_input, instruct))
