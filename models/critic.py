import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class CriticConditional(nn.Module):
    def __init__(self, hidden_size=128, num_layers=1, dropout=0.5, seq_input_dim=4, struct_dim=3):
        """
        hidden_size: liczba neuronów w LSTM
        num_layers: liczba warstw LSTM
        seq_input_dim: 4 dla one-hot RNA
        struct_dim: liczba cech struktury
        """
        super().__init__()
        self.seq_input_dim = seq_input_dim
        self.struct_dim = struct_dim
        self.input_dim = seq_input_dim + struct_dim  # połączone wejście seq+struct

        self.hidden_size = hidden_size

        # Convolutional layers
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

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0 if num_layers == 1 else dropout,
            bidirectional=True
        )

        # Fully connected
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),  # *2 bo BiLSTM
            nn.LeakyReLU(0.3),
            nn.Dropout(dropout),
            spectral_norm(nn.Linear(hidden_size, 1))
        )

    def forward(self, seq, cond_struct=None, mask=None):
        """
        seq: [B, L, 4] one-hot RNA
        cond_struct: [B, L, struct_dim]
        mask: [B, L] 1 = real, 0 = padding
        """
        if cond_struct is not None:
            x = torch.cat([seq, cond_struct], dim=-1)  # [B, L, 4+struct_dim]
        else:
            x = seq

        # permute do Conv1d
        x = x.permute(0, 2, 1)  # [B, C, L]
        x = self.conv_layers(x)
        x = x.permute(0, 2, 1)  # [B, L, hidden]

        # LSTM
        x, _ = self.lstm(x)  # [B, L, hidden*2]

        # jeśli maska, zeroujemy padding przed agregacją
        if mask is not None:
            mask = mask.unsqueeze(-1)  # [B, L, 1]
            x = x * mask
            lengths = mask.sum(dim=1)  # [B, 1]
            x = x.sum(dim=1) / (lengths + 1e-8)  # średnia po nie-paddingowych nukleotydach
        else:
            x = x.mean(dim=1)

        out = self.fc_layers(x)  # [B,1]
        return out
