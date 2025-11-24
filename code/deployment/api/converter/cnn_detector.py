import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import os
from pathlib import Path


class TinyEMNISTDetector:
    """
    Loads CNN trained on EMNIST and uses it as a handwriting detector.
    Idea: handwritten letters trigger high confidence;
          printed text triggers low confidence.
    """

    def __init__(self, model_path=None):
        from .model_cnn import SimpleCNN  # use your existing class

        # Try to find model in different locations
        if model_path is None:
            possible_paths = [
                "models/im2latex_best.pth",
                "converter/models/tiny_cnn_emnist.pth",
                "/app/models/tiny_cnn_emnist.pth",
                os.path.join(os.path.dirname(__file__), "../../models/tiny_cnn_emnist.pth"),
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            if model_path is None:
                raise FileNotFoundError(
                    f"Model file not found. Tried: {possible_paths}"
                )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SimpleCNN(num_classes=27).to(self.device)

        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((64, 256)),
            transforms.ToTensor()
        ])

    def is_handwritten(self, image_path: str) -> bool:
        img = Image.open(image_path).convert("L")
        t = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.model(t)
            probs = F.softmax(out, dim=1).cpu().numpy()[0]

        confidence = np.max(probs)

        # threshold chosen empirically — adjust if needed
        return confidence > 0.25
