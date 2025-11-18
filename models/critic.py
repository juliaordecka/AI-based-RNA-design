import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class CriticConditional(nn.Module):
    def __init__(self, hidden_size=128, num_layers=1, dropout=0.5, seq_input_dim=4, struct_dim=3):

        super().__init__()
        self.seq_input_dim = seq_input_dim
        self.struct_dim = struct_dim
        self.input_dim = seq_input_dim + struct_dim

        self.hidden_size = hidden_size

        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=self.input_dim, out_channels=hidden_size // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.3),
            nn.Dropout(dropout),
            nn.Conv1d(in_channels=hidden_size // 2, out_channels=hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(0.3),
            nn.Dropout(dropout),
            nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(0.3),
            nn.Dropout(dropout),
        )

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0 if num_layers == 1 else dropout,
            bidirectional=True
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),  # *2 bo BiLSTM
            nn.LeakyReLU(0.3),
            nn.Dropout(dropout),
            spectral_norm(nn.Linear(hidden_size, 1))
        )

    def forward(self, seq, cond_struct=None, mask=None):
        if cond_struct is not None:
            x = torch.cat([seq, cond_struct], dim=-1)  # [B, L, 4+struct_dim]
        else:
            x = seq

        x = x.permute(0, 2, 1)
        x = self.conv_layers(x)
        x = x.permute(0, 2, 1)

        x, _ = self.lstm(x)

        if mask is not None:
            mask = mask.unsqueeze(-1)
            x = x * mask
            lengths = mask.sum(dim=1)
            x = x.sum(dim=1) / (lengths + 1e-8)
        else:
            x = x.mean(dim=1)

        out = self.fc_layers(x)
        return out
