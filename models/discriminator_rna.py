import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class DiscriminatorRNA(nn.Module):
    def __init__(self, hidden_size=128, num_layers=1, dropout=0.5):
        super().__init__()
        input_size = 4
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_size, hidden_size // 2, 3, padding=1),
            nn.LeakyReLU(0.3),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_size // 2, hidden_size, 3, padding=1),
            nn.LeakyReLU(0.3),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_size, hidden_size, 3, padding=1),
            nn.LeakyReLU(0.3),
            nn.Dropout(dropout)
        )
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers,
                            batch_first=True, dropout=0 if num_layers == 1 else dropout)
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(0.3),
            nn.Dropout(dropout),
            spectral_norm(nn.Linear(hidden_size // 2, 1))
        )

    def forward(self, x):
        x = x + 0.05 * torch.randn_like(x)
        x = x.permute(0, 2, 1)
        x = self.conv_layers(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = torch.mean(x, dim=1)
        x = self.fc_layers(x)
        return torch.sigmoid(x)
