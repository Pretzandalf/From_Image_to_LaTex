from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import torch
import cv2
import gc

class Qwen_recognition_model():
    def __init__(self):
        self.model = None
        self.processor = None

    def initialize_Qwen_model(self, prompt):

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct-AWQ", dtype=torch.float16, device_map="cuda"
        )

        # default processer
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct-AWQ")

        self.prompt = prompt

    def img_to_text(self, image_path,):

        if not type(image_path) is str:
            image_path = Image.fromarray(cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB))

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": self.prompt},
                ],
            }
        ]

        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=2048)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output_text

if __name__ == "__main__":
    path = "Попробуй максимально точно расписать рукописный текст на русском языке с этого изображения. Обращай внимание на скобки и матрицы. Верни результат."
    prompt = "/home/pret/PycharmProjects/Img_to_LaTex/Images_testment/crop_text_17.png"
    model = Qwen_recognition_model()
    model.initialize_Qwen_model(prompt)
    print(model.img_to_text(path))
