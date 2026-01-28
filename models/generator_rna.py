import torch
import torch.nn as nn
import torch.nn.functional as F
from .embedding import NoiseToRNAEmbedding, PositionalEncoding
from .encoder import Encoder, EncoderBlock, MultiHeadAttention, FeedForward

class GeneratorRNA(nn.Module):
    def __init__(self, latent_dim, d_model, num_layers, num_heads, d_ff, lstm_hidden_size=256, lstm_layers=2, dropout=0.1):
        super().__init__()
        self.noise_embedding = NoiseToRNAEmbedding(latent_dim, d_model)
        self.positional_encoding = PositionalEncoding(5000, d_model)  # max_seq_len może być duże
        encoder_blocks = [EncoderBlock(MultiHeadAttention(d_model, num_heads, dropout),
                                       FeedForward(d_model, d_ff, dropout),
                                       d_model, dropout)
                          for _ in range(num_layers)]
        self.encoder = Encoder(d_model, encoder_blocks)
        self.lstm = nn.LSTM(d_model, lstm_hidden_size, lstm_layers,
                            batch_first=True, dropout=dropout if lstm_layers > 1 else 0)
        self.output = nn.Sequential(
            nn.Linear(lstm_hidden_size, lstm_hidden_size // 2),
            nn.LayerNorm(lstm_hidden_size // 2),
            nn.GELU(),
            nn.Linear(lstm_hidden_size // 2, 4)
        )

    def forward(self, noise, seq_len):
        x = self.noise_embedding(noise, seq_len)
        x = self.positional_encoding(x)
        x = self.encoder(x, mask=None)
        x, _ = self.lstm(x)
        logits = self.output(x)
        return F.gumbel_softmax(logits, tau=0.5, hard=True)
