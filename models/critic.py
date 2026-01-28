import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.utils import spectral_norm


class PairAwareAttention(nn.Module):


    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = spectral_norm(nn.Linear(d_model, d_model))
        self.W_k = spectral_norm(nn.Linear(d_model, d_model))
        self.W_v = spectral_norm(nn.Linear(d_model, d_model))
        self.W_o = spectral_norm(nn.Linear(d_model, d_model))

        self.dropout = nn.Dropout(dropout)
        self.pair_boost = 5.0

    def forward(self, x, pair_matrix=None, mask=None):

        B, L, _ = x.shape

        Q = self.W_q(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if pair_matrix is not None:
            pair_boost = pair_matrix.unsqueeze(1) * self.pair_boost
            scores = scores + pair_boost

        if mask is not None:
            mask_expanded = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask_expanded == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        out = self.W_o(out)

        return out


class CriticConditional(nn.Module):


    def __init__(
            self,
            hidden_size=128,
            num_layers=1,
            dropout=0.2,
            seq_input_dim=4,
            struct_dim=3,
            partner_dim=1,
    ):
        super().__init__()

        self.seq_input_dim = seq_input_dim
        self.struct_dim = struct_dim
        self.partner_dim = partner_dim
        self.input_dim = seq_input_dim + struct_dim + partner_dim
        self.hidden_size = hidden_size

        self.input_proj = spectral_norm(nn.Linear(self.input_dim, hidden_size))

        self.pair_attention = PairAwareAttention(hidden_size, n_heads=4, dropout=dropout)
        self.attn_norm = nn.LayerNorm(hidden_size)


        self.conv1 = spectral_norm(nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1))
        self.conv2 = spectral_norm(nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1))

        self.act = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0 if num_layers == 1 else dropout,
            bidirectional=True,
        )

        self.global_features_dim = 5
        self.global_proj = spectral_norm(nn.Linear(self.global_features_dim, hidden_size // 2))

        self.fc1 = spectral_norm(nn.Linear(hidden_size * 2 + hidden_size // 2, hidden_size))
        self.fc2 = spectral_norm(nn.Linear(hidden_size, 1))

    def build_pair_matrix(self, partners, mask):

        B, L_padded, _ = partners.shape
        device = partners.device

        pair_matrix = torch.zeros(B, L_padded, L_padded, device=device)

        for b in range(B):
            L_orig = int(mask[b].sum().item()) if mask is not None else L_padded

            for i in range(L_orig):
                offset = partners[b, i, 0].item()
                if abs(offset) < 1e-6:
                    continue

                j = int(round(i + offset * L_orig))
                if 0 <= j < L_orig:
                    pair_matrix[b, i, j] = 1.0
                    pair_matrix[b, j, i] = 1.0

        return pair_matrix

    def compute_global_features(self, seq, mask=None):

        B, L, C = seq.size()

        if mask is not None:
            mask_exp = mask.unsqueeze(-1)
            seq_masked = seq * mask_exp
            base_counts = seq_masked.sum(dim=1)
            valid_positions = mask.sum(dim=1, keepdim=True) + 1e-8
            base_dist = base_counts / valid_positions
        else:
            base_dist = seq.mean(dim=1)

        eps = 1e-8
        entropy = -(base_dist * torch.log(base_dist + eps)).sum(dim=1, keepdim=True)
        max_entropy = torch.log(torch.tensor(4.0, device=seq.device))
        entropy = entropy / max_entropy

        global_features = torch.cat([base_dist, entropy], dim=1)
        return global_features

    def forward(self, seq, cond_struct=None, cond_partners=None, mask=None):

        B, L, _ = seq.size()
        device = seq.device

        if mask is None:
            mask = torch.ones(B, L, device=device)

        global_features = self.compute_global_features(seq, mask)
        global_embedded = self.act(self.global_proj(global_features))

        inputs = [seq]
        if cond_struct is not None:
            inputs.append(cond_struct)
        if cond_partners is not None:
            inputs.append(cond_partners)

        x = torch.cat(inputs, dim=-1)

        x = x * mask.unsqueeze(-1)

        x = self.act(self.input_proj(x))

        if cond_partners is not None:
            pair_matrix = self.build_pair_matrix(cond_partners, mask)
        else:
            pair_matrix = None

        attn_out = self.pair_attention(x, pair_matrix, mask)
        x = self.attn_norm(x + attn_out)

        x = x.permute(0, 2, 1)  # (B, C, L)
        x = self.dropout(self.act(self.conv1(x)))
        x = self.dropout(self.act(self.conv2(x)))
        x = x.permute(0, 2, 1)  # (B, L, C)


        with torch.backends.cudnn.flags(enabled=False):
            x, _ = self.lstm(x)

        mask_exp = mask.unsqueeze(-1)
        x = (x * mask_exp).sum(dim=1) / (mask.sum(dim=1, keepdim=True) + 1e-8)


        x = torch.cat([x, global_embedded], dim=1)

        x = self.dropout(self.act(self.fc1(x)))
        return self.fc2(x)



if __name__ == "__main__":
    print("Testing CriticConditional")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    critic = CriticConditional(hidden_size=128).to(device)

    print(f"Parameters: {sum(p.numel() for p in critic.parameters()):,}")

    B, L = 4, 100

    # Fake sequence
    seq = torch.randn(B, L, 4, device=device).softmax(dim=-1)

    # Structure: some paired positions
    struct = torch.zeros(B, L, 3, device=device)
    struct[:, :, 0] = 1  # Default: unpaired
    for i in range(10):
        struct[:, i, 0] = 0
        struct[:, i, 1] = 1  # Open
        struct[:, L - 1 - i, 0] = 0
        struct[:, L - 1 - i, 2] = 1  # Close

    # Partner offsets
    partners = torch.zeros(B, L, 1, device=device)
    for i in range(10):
        j = L - 1 - i
        partners[:, i, 0] = (j - i) / L
        partners[:, j, 0] = (i - j) / L

    mask = torch.ones(B, L, device=device)

    # Forward
    score = critic(seq, struct, partners, mask)
    print(f"Output shape: {score.shape}")  # (4, 1)
    print(f"Scores: {score.squeeze().tolist()}")

    score.sum().backward()
    print("Gradient flow OK")

    print("\nCritic test passed")