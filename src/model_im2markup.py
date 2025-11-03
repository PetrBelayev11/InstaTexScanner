import torch
import torch.nn as nn

class CNNEncoder(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, channels, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.conv(x)

class GRUDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, enc_out, tgt):
        emb = self.embedding(tgt)
        out, _ = self.gru(emb)
        logits = self.fc(out)
        return logits

class Im2Latex(nn.Module):
    def __init__(self, vocab_size=300):
        super().__init__()
        self.encoder = CNNEncoder()
        self.decoder = GRUDecoder(vocab_size)

    def forward(self, img, tgt_seq):
        enc = self.encoder(img)
        logits = self.decoder(enc, tgt_seq)
        return logits
