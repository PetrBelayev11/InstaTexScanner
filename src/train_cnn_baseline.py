import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm  # Import tqdm for progress bars


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=62):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(64 * 14 * 62, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


device = "cuda" if torch.cuda.is_available() else "cpu"
save_dir = Path("results")
save_dir.mkdir(exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((64, 256)),
    transforms.ToTensor()
])

try:
    train_ds = datasets.EMNIST(root="data", split="letters", train=True, download=True, transform=transform)
    test_ds = datasets.EMNIST(root="data", split="letters", train=False, download=True, transform=transform)
except Exception as e:
    print("EMNIST download failed:", e)
    exit()

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=128)

model = SimpleCNN(num_classes=27).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

train_loss, val_acc = [], []

print("Starting training...")
for epoch in range(5):
    model.train()
    running_loss = 0
    
    # Create progress bar for training batches
    train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/5 [Train]', leave=False)
    
    for x, y in train_pbar:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        # Update progress bar with current loss
        train_pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{running_loss/len(train_pbar):.4f}'
        })

    avg_loss = running_loss / len(train_loader)
    train_loss.append(avg_loss)

    model.eval()
    correct, total = 0, 0
    
    # Create progress bar for validation
    test_pbar = tqdm(test_loader, desc=f'Epoch {epoch+1}/5 [Val]', leave=False)
    
    with torch.no_grad():
        for x, y in test_pbar:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            
            # Update progress bar with current accuracy
            current_acc = correct / total
            test_pbar.set_postfix({
                'acc': f'{current_acc:.3f}'
            })
    
    acc = correct / total
    val_acc.append(acc)
    print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, val_acc={acc:.3f}")

plt.figure()
plt.plot(train_loss, label="Training Loss")
plt.plot(val_acc, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Metric")
plt.legend()
plt.title("CNN Baseline Training Progress")
plt.savefig(save_dir / "training_curve_cnn.png")
plt.close()

torch.save(model.state_dict(), save_dir / "cnn_baseline.pt")
print("Training complete. Model and plot saved.")