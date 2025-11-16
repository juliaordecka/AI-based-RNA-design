import torch
import torch.nn as nn
import numpy as np

class NoiseToRNAEmbedding(nn.Module):
    """
    Embedding z wektora losowego do sekwencji RNA.
    Obsługuje dynamiczną długość sekwencji.
    """
    def __init__(self, noise_dim, embedding_size):
        super().__init__()
        self.embedding_size = embedding_size
        self.linear = nn.Linear(noise_dim, embedding_size)

    def forward(self, noise, seq_len):
        B = noise.size(0)
        embedded = self.linear(noise)           # [B, embed_dim]
        embedded = embedded.unsqueeze(1)        # [B, 1, embed_dim]
        embedded = embedded.repeat(1, seq_len, 1)  # [B, L, embed_dim]
        embedded = embedded * np.sqrt(self.embedding_size)
        return embedded

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding.
    """
    def __init__(self, max_seq_len, embedding_size):
        super().__init__()
        positions = torch.arange(max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embedding_size, 2).float() * -(np.log(10000.0) / embedding_size))
        pe = torch.zeros(max_seq_len, embedding_size)
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        self.register_buffer('positional_encoding', pe.unsqueeze(0))

    def forward(self, x):
        L = x.size(1)
        return x + self.positional_encoding[:, :L, :]
