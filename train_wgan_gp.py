import os
import torch
import torch.nn.functional as F
from torch import autograd

def gradient_penalty(critic, real, fake, struct, partners, mask, device):
    B, L, _ = real.size()
    epsilon = torch.rand(B, 1, 1, device=device).expand_as(real)
    interpolated = epsilon * real + (1 - epsilon) * fake
    interpolated.requires_grad_(True)
    crit_out = critic(interpolated, struct, partners, mask)
    grad = autograd.grad(
        outputs=crit_out,
        inputs=interpolated,
        grad_outputs=torch.ones_like(crit_out),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    grad = grad.reshape(B, -1)
    gp = ((grad.norm(2, dim=1) - 1) ** 2).mean()
    return gp


def compute_diversity_penalty(fake_samples, mask):

    B, L, _ = fake_samples.size()
    device = fake_samples.device

    mask_expanded = mask.unsqueeze(-1)
    masked_samples = fake_samples * mask_expanded

    base_counts = masked_samples.sum(dim=[0, 1])
    total_valid = base_counts.sum()

    if total_valid < 1:
        return torch.tensor(0.0, device=device)

    #count nucleotide ratios
    a = base_counts[0] / total_valid
    c = base_counts[1] / total_valid
    g = base_counts[2] / total_valid
    u = base_counts[3] / total_valid

    #penalize deficits of nucleotides - thresholds from statistics
    x = F.relu(0.23 - a) / 0.23  # A deficit
    y = F.relu(0.25 - c) / 0.25  # C deficit
    z = F.relu(0.29 - g) / 0.29  # G deficit
    w = F.relu(0.24 - u) / 0.24  # U deficit

    distrib_loss = (x + y + z + w) / 4.0

    return distrib_loss


def compute_pairing_penalty(fake_samples, partners, mask, lambda_pair=1.0):
    """
    enforces pairing (A-U, G-C) and wobble (G-U)
    encoding:
    A=0, C=1, G=2, U=3
    """
    B, L_padded, _ = fake_samples.size()
    device = fake_samples.device

    base_indices = fake_samples.argmax(dim=-1)

    #valid pairs of equal weight
    complement_matrix = torch.zeros(4, 4, device=device)
    complement_matrix[0, 3] = 1.0  # A-U
    complement_matrix[3, 0] = 1.0  # U-A
    complement_matrix[1, 2] = 1.0  # C-G
    complement_matrix[2, 1] = 1.0  # G-C
    complement_matrix[2, 3] = 1.0  # G-U wobble
    complement_matrix[3, 2] = 1.0  # U-G wobble

    pair_count = 0
    correct_pairs = 0
    gc_pairs = 0
    au_pairs = 0
    gu_pairs = 0

    for b in range(B):
        L_orig = int(mask[b].sum().item())
        if L_orig < 2:
            continue

        for i in range(L_orig):
            offset = partners[b, i, 0].item()
            if abs(offset) < 1e-6:
                continue

            if offset <= 0:
                continue

            partner_idx = int(round(i + offset * L_orig))

            if not (0 <= partner_idx < L_orig):
                continue

            base_i = base_indices[b, i].item()
            base_j = base_indices[b, partner_idx].item()

            is_valid_pair = complement_matrix[int(base_i), int(base_j)].item()

            if is_valid_pair > 0.5:
                correct_pairs += 1
                #G-C: (G=2, C=1) or (C=1, G=2)
                if (base_i == 2 and base_j == 1) or (base_i == 1 and base_j == 2):
                    gc_pairs += 1
                #A-U: (A=0, U=3) or (U=3, A=0)
                elif (base_i == 0 and base_j == 3) or (base_i == 3 and base_j == 0):
                    au_pairs += 1
                #G-U: (G=2, U=3) or (U=3, G=2)
                elif (base_i == 2 and base_j == 3) or (base_i == 3 and base_j == 2):
                    gu_pairs += 1

            pair_count += 1

    if pair_count > 0:
        pair_accuracy = correct_pairs / pair_count
        gc_ratio = gc_pairs / pair_count
        au_ratio = au_pairs / pair_count
        gu_ratio = gu_pairs / pair_count
    else:
        pair_accuracy = 0.0
        gc_ratio = 0.0
        au_ratio = 0.0
        gu_ratio = 0.0

    #differentiable pairing loss
    soft_pair_loss = compute_soft_pairing_penalty(fake_samples, partners, mask, complement_matrix)
    return soft_pair_loss * lambda_pair, pair_accuracy, gc_ratio, au_ratio, gu_ratio


def compute_soft_pairing_penalty(fake_samples, partners, mask, complement_matrix):
    #differentiable version of pairing penalty using probabilities
    B, L_padded, _ = fake_samples.size()
    device = fake_samples.device

    probs = fake_samples

    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    pair_count = 0

    for b in range(B):
        L_orig = int(mask[b].sum().item())
        if L_orig < 2:
            continue

        for i in range(L_orig):
            offset = partners[b, i, 0].item()
            if abs(offset) < 1e-6:
                continue
            if offset <= 0:
                continue

            partner_idx = int(round(i + offset * L_orig))
            if not (0 <= partner_idx < L_orig):
                continue

            prob_i = probs[b, i]
            prob_j = probs[b, partner_idx]

            valid_pair_prob = torch.einsum('i,ij,j->', prob_i, complement_matrix, prob_j)

            total_loss = total_loss + (1.0 - valid_pair_prob)
            pair_count += 1

    if pair_count > 0:
        return total_loss / pair_count
    else:
        return torch.tensor(0.0, device=device, requires_grad=True)


def compute_pair_distribution_loss(fake_samples, partners, mask,
                                   target_gc=0.50, target_au=0.20, target_gu=0.05):
    """
    enforces minimum ratios for each pair type
    penalizes:
    GC < 50% of all pairs
    AU < 20% of all pairs
    GU < 5% of all pairs
    """
    B, L_padded, _ = fake_samples.size()
    device = fake_samples.device

    probs = fake_samples

    #pair type matrices for soft/differentiable computation
    gc_matrix = torch.zeros(4, 4, device=device)
    gc_matrix[2, 1] = 1.0  # G-C
    gc_matrix[1, 2] = 1.0  # C-G

    au_matrix = torch.zeros(4, 4, device=device)
    au_matrix[0, 3] = 1.0  # A-U
    au_matrix[3, 0] = 1.0  # U-A

    gu_matrix = torch.zeros(4, 4, device=device)
    gu_matrix[2, 3] = 1.0  # G-U
    gu_matrix[3, 2] = 1.0  # U-G

    gc_prob_sum = torch.tensor(0.0, device=device, requires_grad=True)
    au_prob_sum = torch.tensor(0.0, device=device, requires_grad=True)
    gu_prob_sum = torch.tensor(0.0, device=device, requires_grad=True)
    pair_count = 0

    for b in range(B):
        L_orig = int(mask[b].sum().item())
        if L_orig < 2:
            continue

        for i in range(L_orig):
            offset = partners[b, i, 0].item()
            if abs(offset) < 1e-6:
                continue
            if offset <= 0:
                continue

            partner_idx = int(round(i + offset * L_orig))
            if not (0 <= partner_idx < L_orig):
                continue

            prob_i = probs[b, i]
            prob_j = probs[b, partner_idx]

            # Probability of each pair type
            gc_prob = torch.einsum('i,ij,j->', prob_i, gc_matrix, prob_j)
            au_prob = torch.einsum('i,ij,j->', prob_i, au_matrix, prob_j)
            gu_prob = torch.einsum('i,ij,j->', prob_i, gu_matrix, prob_j)

            gc_prob_sum = gc_prob_sum + gc_prob
            au_prob_sum = au_prob_sum + au_prob
            gu_prob_sum = gu_prob_sum + gu_prob
            pair_count += 1

    if pair_count == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Compute ratios
    gc_ratio = gc_prob_sum / pair_count
    au_ratio = au_prob_sum / pair_count
    gu_ratio = gu_prob_sum / pair_count

    # Compute deficits (only penalize if below target)
    # a = max(target - actual, 0) / target
    a = F.relu(target_gc - gc_ratio) / target_gc
    b = F.relu(target_au - au_ratio) / target_au
    c = F.relu(target_gu - gu_ratio) / target_gu

    #average deficit
    distrib_loss_3 = (a + b + c) / 3.0

    #penalty for severe total deficit
    distrib_loss_4 = F.relu((a + b + c) - 1.0)

    return distrib_loss_3 + distrib_loss_4


def train_wgan_gp(generator, critic, loader, args, device):

    g_opt = torch.optim.Adam(generator.parameters(), lr=args.lr_g, betas=(0.5, 0.9))
    c_opt = torch.optim.Adam(critic.parameters(), lr=args.lr_c, betas=(0.5, 0.9))

    lambda_gp = getattr(args, 'lambda_gp', 10.0)
    lambda_pair = getattr(args, 'lambda_pair', 0.5)
    lambda_diversity = getattr(args, 'lambda_diversity', 2.0)
    lambda_distrib = getattr(args, 'lambda_distrib', 1.0)

    #target pair ratios (minimum percentage of each pair)
    target_gc = getattr(args, 'target_gc', 0.35)
    target_au = getattr(args, 'target_au', 0.12)
    target_gu = getattr(args, 'target_gu', 0.01)

    total_batches = 0

    initial_tau = 1.0
    min_tau = 0.5
    tau_anneal_epochs = 50

    for epoch in range(args.epochs):
        if epoch < tau_anneal_epochs:
            current_tau = initial_tau - (initial_tau - min_tau) * (epoch / tau_anneal_epochs)
        else:
            current_tau = min_tau
        generator.tau = current_tau

        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        epoch_pair_acc = 0.0
        epoch_gc_ratio = 0.0
        epoch_au_ratio = 0.0
        epoch_gu_ratio = 0.0
        batch_count = 0

        for batch_idx, batch in enumerate(loader):
            total_batches += 1
            batch_count += 1

            seq = batch['seq'].to(device)
            struct = batch['struct'].to(device)
            mask = batch['mask'].to(device)
            partners = batch['partners'].to(device)

            B, L, _ = seq.size()

            #training critic
            for _ in range(args.n_critic):
                c_opt.zero_grad()

                z = torch.randn(B, args.latent_dim, device=device)

                with torch.no_grad():
                    fake_samples, _ = generator(z, struct, partners, mask, return_logits=False)

                c_real = critic(seq, struct, partners, mask)
                c_fake = critic(fake_samples, struct, partners, mask)

                wasserstein = c_real.mean() - c_fake.mean()
                gp = gradient_penalty(critic, seq, fake_samples, struct, partners, mask, device)

                d_loss = -wasserstein + lambda_gp * gp
                d_loss.backward()
                #zakomentowanie gradient clipping
                #torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=10.0)
                c_opt.step()

            #training generator
            g_opt.zero_grad()

            z = torch.randn(B, args.latent_dim, device=device)
            #zmiana
            fake_logits, _ = generator(z, struct, partners, mask, return_logits=True)
            fake_samples = F.gumbel_softmax(fake_logits, tau=current_tau, hard=True)

            c_fake_for_g = critic(fake_samples, struct, partners, mask)

            g_loss_adv = -c_fake_for_g.mean()

            #pairing penalty (for valid base pairs)
            pair_penalty, pair_acc, gc_ratio, au_ratio, gu_ratio = compute_pairing_penalty(
                fake_samples, partners, mask, lambda_pair
            )

            #pair distribution loss (minimum GC/AU/GU ratios)
            distrib_loss = compute_pair_distribution_loss(
                fake_samples, partners, mask, target_gc, target_au, target_gu
            )
            #diversity penalty
            diversity_penalty = compute_diversity_penalty(fake_samples, mask)

            g_loss = (g_loss_adv +
                      pair_penalty +
                      lambda_distrib * distrib_loss +
                      lambda_diversity * diversity_penalty)

            g_loss.backward()

            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=10.0)
            g_opt.step()

            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            epoch_pair_acc += pair_acc
            epoch_gc_ratio += gc_ratio
            epoch_au_ratio += au_ratio
            epoch_gu_ratio += gu_ratio

            #logging
            if (batch_idx + 1) % 1 == 0:
                with torch.no_grad():
                    mask_expanded = mask.unsqueeze(-1)
                    masked_samples = fake_samples * mask_expanded
                    base_counts = masked_samples.sum(dim=[0, 1])
                    total_bases = base_counts.sum()
                    base_probs = base_counts / (total_bases + 1e-8)

                print(f"[Epoch {epoch + 1}] [Batch {batch_idx + 1}/{len(loader)}] "
                      f"[D: {d_loss.item():.4f}] [G: {g_loss.item():.4f}] "
                      f"[W: {wasserstein.item():.4f}] [GP: {gp.item():.4f}] "
                      f"[Pair: {pair_penalty.item():.4f}] [PairAcc: {pair_acc:.2%}] "
                      f"[GC: {gc_ratio:.2%}] [AU: {au_ratio:.2%}] [GU: {gu_ratio:.2%}] "
                      f"[Distrib: {distrib_loss.item():.4f}] "
                      f"[tau: {current_tau:.3f}] "
                      f"[A:{base_probs[0]:.2f} C:{base_probs[1]:.2f} G:{base_probs[2]:.2f} U:{base_probs[3]:.2f}]",
                      flush=True)

            #saving checkpoints
            if total_batches % 10 == 0:
                os.makedirs(args.save_dir, exist_ok=True)
                gen_path = os.path.join(args.save_dir, f"generator_epoch_{epoch + 1}_batch_{total_batches}.pth")
                torch.save(generator.state_dict(), gen_path)
                print(f" Saved checkpoint: {gen_path}", flush=True)