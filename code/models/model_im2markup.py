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
    GRU decoder with attention to encoder features.
    """

    def __init__(self, vocab_size, hidden_dim=256, enc_dim=128, dropout_p=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.gru = nn.GRU(hidden_dim + enc_dim, hidden_dim, batch_first=True)
        # Multi-head attention over encoder features.
        # We project the decoder hidden state into the encoder space
        # and use it as a query to the image feature sequence.
        self.attention = nn.MultiheadAttention(enc_dim, num_heads=4, batch_first=True)
        self.query_proj = nn.Linear(hidden_dim, enc_dim)
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.enc_dim = enc_dim

    def forward(self, enc_feat, tgt_seq):
        # enc_feat: B x seq_len x enc_dim
        # tgt_seq: B x T
        emb = self.embedding(tgt_seq)  # B x T x H
        emb = self.dropout(emb)

        # Attention over the entire sequence of image features
        # 1) compute queries from target sequence embeddings
        query = self.query_proj(emb)  # B x T x enc_dim
        # 2) multi-head attention: for each step look at the entire feature map
        #    enc_feat acts as both key and value
        attn_output, _ = self.attention(query, enc_feat, enc_feat)  # B x T x enc_dim

        # Concatenate embedding and context from image
        gru_input = torch.cat([emb, attn_output], dim=-1)  # B x T x (H + enc_dim)

        out, _ = self.gru(gru_input)  # B x T x H
        out = self.dropout(out)
        logits = self.fc(out)  # B x T x vocab
        return logits


class Im2Latex(nn.Module):
    """
    Complete model used in pipeline.py and inference.py.
    """

    def __init__(self, vocab_size):
        super().__init__()
        self.encoder = CNNEncoder()
        # Small dropout in the decoder helps to fight overfitting.
        self.decoder = GRUDecoder(vocab_size)

    def forward(self, img, tgt_seq):
        enc = self.encoder(img)                # B x seq_len x C
        logits = self.decoder(enc, tgt_seq)    # B x T x vocab
        return logits

