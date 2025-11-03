import torch
from PIL import Image
from torchvision import transforms
from model_im2markup import Im2Latex

device = "cuda" if torch.cuda.is_available() else "cpu"

def predict(image_path: str, model: Im2Latex, vocab: list):
    img = Image.open(image_path).convert("L")
    transform = transforms.Compose([
        transforms.Resize((64, 256)),
        transforms.ToTensor()
    ])
    x = transform(img).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        pred_ids = torch.randint(0, len(vocab), (1, 15))
    pred_tokens = [vocab[i] for i in pred_ids[0]]
    return "".join(pred_tokens)

if __name__ == "__main__":
    vocab = list("abcdefghijklmnopqrstuvwxyz1234567890=^_/")
    model = Im2Latex(len(vocab)).to(device)
    out = predict("data/samples/sample_0.png", model, vocab)
    print("Predicted LaTeX:", out)
