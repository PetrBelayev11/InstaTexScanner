from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

PRINTED_MODEL = "microsoft/trocr-base-stage1"
HANDWRITTEN_MODEL = "microsoft/trocr-base-handwritten"


class TrocrEngine:

    def __init__(self, model_name=PRINTED_MODEL):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(self.device)

    def run(self, image_path: str) -> str:
        img = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt").pixel_values.to(self.device)

        with torch.no_grad():
            ids = self.model.generate(inputs, num_beams=4, max_length=256)

        return self.processor.batch_decode(ids, skip_special_tokens=True)[0]
