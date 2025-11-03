import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from model_im2markup import Im2Latex

class DummyFormulaDataset(Dataset):
    def __init__(self, num_samples=500, seq_len=10, vocab_size=50):
        self.vocab_size = vocab_size
        self.images = torch.rand(num_samples, 1, 64, 256)
        self.labels = torch.randint(0, vocab_size, (num_samples, seq_len))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

device = "cuda" if torch.cuda.is_available() else "cpu"
save_dir = Path("results")
save_dir.mkdir(exist_ok=True)

train_ds = DummyFormulaDataset(400)
val_ds = DummyFormulaDataset(100)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16)

model = Im2Latex(vocab_size=50).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

train_loss, val_loss = [], []

for epoch in range(5):
    model.train()
    running = 0
    for img, tgt in train_loader:
        img, tgt = img.to(device), tgt.to(device)
        optimizer.zero_grad()
        out = model(img, tgt)
        loss = criterion(out.view(-1, out.size(-1)), tgt.view(-1))
        loss.backward()
        optimizer.step()
        running += loss.item()

    model.eval()
    val_running = 0
    with torch.no_grad():
        for img, tgt in val_loader:
            img, tgt = img.to(device), tgt.to(device)
            out = model(img, tgt)
            loss = criterion(out.view(-1, out.size(-1)), tgt.view(-1))
            val_running += loss.item()

    train_loss.append(running / len(train_loader))
    val_loss.append(val_running / len(val_loader))
    print(f"Epoch {epoch+1}: train={train_loss[-1]:.3f}, val={val_loss[-1]:.3f}")

plt.figure()
plt.plot(train_loss, label="Train Loss")
plt.plot(val_loss, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Im2Markup Training Progress")
plt.savefig(save_dir / "training_curve_im2markup.png")
plt.close()

torch.save(model.state_dict(), save_dir / "im2markup_model.pt")
print("Im2Markup training complete. Artifacts saved.")
