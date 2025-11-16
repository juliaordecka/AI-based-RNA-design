import torch
import os
import csv
from utils.noise_generator import generate_noise

def log_metrics(epoch, batch_idx, num_batches, d_loss, g_loss, critic_real, critic_fake, log_path):
    d_real_mean = torch.mean(critic_real).item()
    d_fake_mean = torch.mean(critic_fake).item()
    wasserstein_distance = d_real_mean - d_fake_mean

    print(
        f"[Epoch {epoch+1}] [Batch {batch_idx+1}/{num_batches}] "
        f"[D loss: {d_loss:.4f}] [G loss: {g_loss:.4f}] "
        f"[Real val: {d_real_mean:.4f}] [Fake val: {d_fake_mean:.4f}] "
        f"[Wasserstein: {wasserstein_distance:.4f}]"
    )
    if log_path:
        with open(log_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                batch_idx, d_loss, g_loss, d_real_mean, d_fake_mean, wasserstein_distance
            ])


def gradient_penalty(critic, real_samples, fake_samples, cond_struct=None, mask=None, device="cpu"):
    """
    Compute gradient penalty for WGAN-GP.
    critic(... ) must return shape (B, 1) or (B,)
    """
    batch_size = real_samples.size(0)
    # random weight for interpolation between real and fake
    epsilon = torch.rand(batch_size, 1, 1, device=device)
    interpolated = epsilon * real_samples + (1 - epsilon) * fake_samples
    interpolated.requires_grad_(True)

    with torch.backends.cudnn.flags(enabled=False):
        mixed_scores = critic(interpolated, cond_struct, mask)  # expect (B,1) or (B,)

    # ensure mixed_scores is shape (B,) for grad calculation
    mixed_scores = mixed_scores.view(-1)

    grad_outputs = torch.ones_like(mixed_scores, device=device)

    gradient = torch.autograd.grad(
        outputs=mixed_scores,
        inputs=interpolated,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    # gradient may be non-contiguous; use reshape (or contiguous().view)
    gradient = gradient.reshape(batch_size, -1)
    gradient_norm = gradient.norm(2, dim=1)
    penalty = torch.mean((gradient_norm - 1) ** 2)
    return penalty


def train_wgan_gp(generator, critic, loader, args, device):
    """
    loader: torch.utils.data.DataLoader returning dicts with keys 'seq','struct','mask'
    """
    generator.train()
    critic.train()

    os.makedirs(args.save_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, "metrics.csv") if args.log_dir else None
    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)
        with open(log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["batch", "d_loss", "g_loss", "critic_real_mean", "critic_fake_mean", "wasserstein_distance"])

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
    optimizer_C = torch.optim.Adam(critic.parameters(), lr=args.lr_c, betas=(0.5, 0.999))

    total_batches = 0
    num_batches = len(loader)

    for epoch in range(args.epochs):
        for i, batch in enumerate(loader):
            total_batches += 1

            real_rna = batch['seq'].float().to(device)        # (B, L, 4)
            cond_struct = batch['struct'].float().to(device)  # (B, L, 3)
            mask = batch['mask'].float().to(device)           # (B, L)
            batch_size = real_rna.size(0)

            # ---------------------
            # Train Critic
            # ---------------------
            z = generate_noise(args.latent_dim, batch_size).to(device)
            fake_rna = generator(z, cond_struct)

            critic_real = critic(real_rna, cond_struct, mask)
            critic_fake = critic(fake_rna.detach(), cond_struct, mask)

            gp = gradient_penalty(critic, real_rna, fake_rna.detach(), cond_struct, mask, device)
            critic_loss = torch.mean(critic_fake) - torch.mean(critic_real) + args.lambda_gp * gp

            optimizer_C.zero_grad()
            critic_loss.backward(retain_graph=True)
            optimizer_C.step()

            # ---------------------
            # Train Generator (every n_critic steps)
            # ---------------------
            generator_loss_val = float('nan')
            if i % args.n_critic == 0:
                z = generate_noise(args.latent_dim, batch_size).to(device)
                fake_rna = generator(z, cond_struct)
                gen_scores = critic(fake_rna, cond_struct, mask)
                generator_loss = -torch.mean(gen_scores)
                optimizer_G.zero_grad()
                generator_loss.backward()
                optimizer_G.step()
                generator_loss_val = generator_loss.item()

            # ---------------------
            # Logging
            # ---------------------
            # make sure to pass a numeric g_loss (not tensor)
            g_loss_for_log = generator_loss_val if isinstance(generator_loss_val, float) else (generator_loss.item() if 'generator_loss' in locals() else float('nan'))
            log_metrics(epoch, i, num_batches, critic_loss.item(), g_loss_for_log, critic_real, critic_fake, log_path)

            # ---------------------
            # Save model periodically
            # ---------------------
            if total_batches % 30 == 0:
                torch.save(generator.state_dict(), os.path.join(args.save_dir, f"generator_epoch_{epoch+1}_batch_{total_batches}.pth"))
