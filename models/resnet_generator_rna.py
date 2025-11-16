import torch
import torch.nn as nn
import torch.nn.functional as F
from .embedding import NoiseToRNAEmbedding

class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=5, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = F.gelu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = F.gelu(out)
        out = self.conv2(out)
        out = self.dropout(out)
        return out + residual

class ResNetGeneratorConditional(nn.Module):
    def __init__(self, latent_dim, struct_dim=3, embed_dim=256, n_blocks=4):
        super().__init__()
        self.struct_dim = struct_dim
        self.noise_embedding = NoiseToRNAEmbedding(latent_dim, embed_dim)
        self.struct_proj = nn.Linear(struct_dim, embed_dim)
        kernel_sizes = [3, 5, 7, 9][:n_blocks]
        self.res_blocks = nn.Sequential(*[ResidualBlock(embed_dim, k) for k in kernel_sizes])
        self.out = nn.Conv1d(embed_dim, 4, kernel_size=1)

    def forward(self, noise, cond_struct):
        seq_len = cond_struct.size(1)
        embedded = self.noise_embedding(noise, seq_len)
        cond_emb = self.struct_proj(cond_struct)
        x = embedded + cond_emb
        x = x.permute(0, 2, 1)
        x = self.res_blocks(x)
        x = self.out(x)
        x = x.permute(0, 2, 1)
        return F.gumbel_softmax(x, tau=0.5, hard=True)
