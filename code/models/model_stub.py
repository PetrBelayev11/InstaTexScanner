

import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    """
    Encodes image into a feature sequence for decoder.
    Input: batch x 1 x 64 x 256
    Output: batch x (sequence_len) x channels
    """

    def __init__(self, channels=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(64, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        )

    def forward(self, x):
        feat = self.net(x)          # B x C x 16 x 64
        B, C, H, W = feat.shape
        feat = feat.permute(0, 2, 3, 1).reshape(B, H * W, C)
        return feat                 # B x (H*W) x C


class GRUDecoder(nn.Module):
    """
    Standard GRU decoder for character prediction.
    """

    def __init__(self, vocab_size, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, enc_feat, tgt_seq):
        emb = self.embedding(tgt_seq)         # B x T x H
        out, _ = self.gru(emb)                # B x T x H
        logits = self.fc(out)                 # B x T x vocab
        return logits


class Im2Latex(nn.Module):
    """
    Complete model used in pipeline.py and inference.py.
    """

    def __init__(self, vocab_size):
        super().__init__()
        self.encoder = CNNEncoder()
        self.decoder = GRUDecoder(vocab_size)

    def forward(self, img, tgt_seq):
        enc = self.encoder(img)                # B x seq_len x C
        logits = self.decoder(enc, tgt_seq)    # B x T x vocab
        return logits
