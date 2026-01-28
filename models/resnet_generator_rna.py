import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class PairAwareAttention(nn.Module):

    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.pair_boost = 5.0  # How much to boost attention for paired positions

    def forward(self, x, pair_matrix=None):

        B, L, _ = x.shape

        Q = self.W_q(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)  # (B, H, L, d_k)
        K = self.W_k(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # (B, H, L, L)

        if pair_matrix is not None:
            pair_boost = pair_matrix.unsqueeze(1) * self.pair_boost
            scores = scores + pair_boost

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)  # (B, H, L, d_k)
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        out = self.W_o(out)

        return out


class ResidualBlock(nn.Module):

    def __init__(self, channels, kernel_size=5, dropout=0.1):
        super().__init__()
        padding = kernel_size // 2

        self.bn1 = nn.BatchNorm1d(channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = F.gelu(self.bn1(x))
        out = self.conv1(out)
        out = F.gelu(self.bn2(out))
        out = self.conv2(out)
        out = self.dropout(out)
        return out + residual


class ResNetGeneratorConditional(nn.Module):

    def __init__(self, latent_dim=256, struct_dim=3, partner_dim=1, embed_dim=128, n_blocks=4):
        super().__init__()

        self.latent_dim = latent_dim
        self.struct_dim = struct_dim
        self.partner_dim = partner_dim
        self.embed_dim = embed_dim

        self.global_noise_proj = nn.Linear(latent_dim, embed_dim)

        self.local_noise_dim = 32
        self.local_noise_proj = nn.Linear(self.local_noise_dim, embed_dim // 2)

        self.pos_encoding = PositionalEncoding(embed_dim, max_len=5000)

        cond_dim = struct_dim + partner_dim  # 4
        self.cond_to_scale = nn.Sequential(
            nn.Linear(cond_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.cond_to_shift = nn.Sequential(
            nn.Linear(cond_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

        self.input_proj = nn.Linear(embed_dim + embed_dim // 2, embed_dim)

        self.pair_attention = PairAwareAttention(embed_dim, n_heads=4, dropout=0.1)
        self.attn_norm = nn.LayerNorm(embed_dim)

        kernel_sizes = [3, 5, 7, 5][:n_blocks]
        self.res_blocks = nn.ModuleList([
            ResidualBlock(embed_dim, k, dropout=0.1) for k in kernel_sizes
        ])

        self.out_conv = nn.Conv1d(embed_dim, 4, kernel_size=1)
        self.aux_partner = nn.Conv1d(embed_dim, 1, kernel_size=1)

        self.tau = 1.0

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

    def forward(self, noise, cond_struct, cond_partners=None, mask=None, return_logits=False):

        B = noise.size(0)
        L = cond_struct.size(1)
        device = noise.device

        global_emb = self.global_noise_proj(noise)
        global_emb = global_emb.unsqueeze(1).expand(-1, L, -1)

        local_noise = torch.randn(B, L, self.local_noise_dim, device=device)
        local_emb = self.local_noise_proj(local_noise)

        combined = torch.cat([global_emb, local_emb], dim=-1)
        x = self.input_proj(combined)

        x = self.pos_encoding(x)

        if cond_partners is None:
            cond_partners = torch.zeros(B, L, self.partner_dim, device=device)

        cond_input = torch.cat([cond_struct, cond_partners], dim=-1)
        scale = self.cond_to_scale(cond_input)
        shift = self.cond_to_shift(cond_input)

        x = scale * x + shift

        if mask is None:
            mask = torch.ones(B, L, device=device)
        pair_matrix = self.build_pair_matrix(cond_partners, mask)

        attn_out = self.pair_attention(x, pair_matrix)
        x = self.attn_norm(x + attn_out)

        x = x.permute(0, 2, 1)
        for block in self.res_blocks:
            x = block(x)
        x = x.permute(0, 2, 1)

        x = x.permute(0, 2, 1)
        pred_offsets = self.aux_partner(x).permute(0, 2, 1)
        logits = self.out_conv(x).permute(0, 2, 1)

        if return_logits:
            return logits, pred_offsets

        samples = F.gumbel_softmax(logits, tau=self.tau, hard=True)
        return samples, pred_offsets
###############################################################################################

# Test the model
if __name__ == "__main__":
    print("Testing ResNetGeneratorConditional")

    model = ResNetGeneratorConditional(latent_dim=256, embed_dim=128, n_blocks=4)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    B, L = 4, 100
    noise = torch.randn(B, 256)

    struct = torch.zeros(B, L, 3)
    struct[:, :, 0] = 1

    for i in range(10):
        struct[:, i, 0] = 0
        struct[:, i, 1] = 1
        struct[:, L - 1 - i, 0] = 0
        struct[:, L - 1 - i, 2] = 1

    partners = torch.zeros(B, L, 1)
    for i in range(10):
        j = L - 1 - i
        partners[:, i, 0] = (j - i) / L
        partners[:, j, 0] = (i - j) / L

    mask = torch.ones(B, L)

    samples, offsets = model(noise, struct, partners, mask)
    print(f"Samples shape: {samples.shape}")
    print(f"Offsets shape: {offsets.shape}")

    base_counts = samples.sum(dim=[0, 1])
    print(f"Overall base distribution: A={base_counts[0] / base_counts.sum():.2%}, "
          f"C={base_counts[1] / base_counts.sum():.2%}, "
          f"G={base_counts[2] / base_counts.sum():.2%}, "
          f"U={base_counts[3] / base_counts.sum():.2%}")

    print("\nPer-sequence diversity:")
    for b in range(B):
        seq_counts = samples[b].sum(dim=0)
        total = seq_counts.sum()
        print(f"  Seq {b}: A={seq_counts[0] / total:.2%}, C={seq_counts[1] / total:.2%}, "
              f"G={seq_counts[2] / total:.2%}, U={seq_counts[3] / total:.2%}")

    pair_matrix = model.build_pair_matrix(partners, mask)
    print(f"\nPair matrix: {pair_matrix[0].sum().item():.0f} pairs for sequence 0")

    print("\nModel test passed!")