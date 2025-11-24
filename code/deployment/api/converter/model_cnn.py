import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    Simple CNN for EMNIST classification.
    Used for handwriting detection.
    Input: batch x 1 x 64 x 256
    Output: batch x num_classes
    """
    
    def __init__(self, num_classes=27):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28x28 -> 14x14
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14 -> 7x7
            
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),  # 32*7*7 = 1568
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)

