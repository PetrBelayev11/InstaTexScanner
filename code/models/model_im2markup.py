import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    """
    Encodes image into a feature sequence for decoder.
    Input: batch x 1 x 128 x 512
    Output: batch x (sequence_len) x channels
    """

    def __init__(self, channels=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),  # 128x512 -> 64x256

            nn.Conv2d(64, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),  # 64x256 -> 32x128
        )

    def forward(self, x):
        feat = self.net(x)          # B x C x 32 x 128
        B, C, H, W = feat.shape
        feat = feat.permute(0, 2, 3, 1).reshape(B, H * W, C)
        return feat                 # B x (32*128=4096) x C


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
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, num_layers=2, dropout=0.3):
        super(Im2Latex, self).__init__()
        
        # CNN Encoder for 128x512 images
        self.cnn_encoder = nn.Sequential(
            # Input: [batch, 1, 128, 512]
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # [64, 64, 256]
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # [128, 32, 128]
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # [256, 16, 64]
            
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # [512, 8, 32]
            
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # [512, 4, 32]
            
            nn.AdaptiveAvgPool2d((4, 32))  # Ensure consistent output: [512, 4, 32]
        )
        
        # Rest of the architecture remains the same
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=2)
        self.lstm = nn.LSTM(
            input_size=512 * 4 + embed_dim,  # CNN features + embeddings
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, images, input_tokens):
        # CNN features
        cnn_features = self.cnn_encoder(images)  # [B, 512, 4, 32]
        batch_size = cnn_features.size(0)
        cnn_features = cnn_features.view(batch_size, 512 * 4, -1)  # [B, 2048, 32]
        cnn_features = cnn_features.permute(0, 2, 1)  # [B, 32, 2048]
        
        # Embed tokens
        token_embeddings = self.embedding(input_tokens)  # [B, seq_len, embed_dim]
        
        # Repeat CNN features for each time step
        seq_len = input_tokens.size(1)
        cnn_features = cnn_features.repeat(1, seq_len // 32 + 1, 1)[:, :seq_len, :]
        
        # Combine features
        combined = torch.cat([cnn_features, token_embeddings], dim=-1)
        combined = self.dropout(combined)
        
        # LSTM
        lstm_out, _ = self.lstm(combined)
        
        # Output projection
        output = self.output_proj(lstm_out)
        return output
