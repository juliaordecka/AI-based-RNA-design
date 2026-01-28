import argparse
import time
import os
import torch
import torch.nn.functional as F
from utils.init_device import init_cuda
from loaders.fasta_data_loader import FastDatasetRNA_Conditional, pad_collate
from torch.utils.data import DataLoader
from models.resnet_generator_rna import ResNetGeneratorConditional
from utils.init_weights import initialize_weights
from models.critic import CriticConditional
from train_wgan_gp import train_wgan_gp


def parse_args():
    parser = argparse.ArgumentParser(description="Run WGAN-GP for RNA sequence generation based on secondary structure")

    parser.add_argument("--data", type=str, default="data/bp_seq_train.fasta",
                        help="Path to the RNA sequence data file")

    parser.add_argument("--max_len", type=int, default=750,
                        help="Maximum sequence length (filters out longer sequences)")

    parser.add_argument("--epochs", type=int, default=400,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--latent_dim", type=int, default=256,
                        help="Dimension of latent noise vector")

    parser.add_argument("--n_critic", type=int, default=5,
                        help="Critic updates per generator update")
    parser.add_argument("--lambda_gp", type=float, default=10.0,
                        help="Gradient penalty weight")
    parser.add_argument("--lambda_pair", type=float, default=1.0,
                        help="Base pairing penalty weight")
    parser.add_argument("--lambda_diversity", type=float, default=1.0,
                        help="Diversity penalty weight")


    parser.add_argument("--lr_g", type=float, default=0.0005,
                        help="Generator learning rate")
    parser.add_argument("--lr_c", type=float, default=0.0001,
                        help="Critic learning rate")

    parser.add_argument("--save_dir", type=str, default="saved_models",
                        help="Directory to save model checkpoints")
    parser.add_argument("--log_dir", type=str, default=None,
                        help="Path to the log file (optional)")

    return parser.parse_args()


def main():
    args = parse_args()

    t0 = time.time()
    print(f"\nWorking directory: {os.getcwd()}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Device count: {torch.cuda.device_count()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"\n{'=' * 40}")
    print("LOADING DATASET")
    print(f"{'=' * 40}")
    print(f"Data file: {args.data}")

    dataset = FastDatasetRNA_Conditional(args.data)

    original_size = len(dataset.data)
    dataset.data = [d for d in dataset.data if d['len'] <= args.max_len]
    filtered_size = len(dataset.data)

    print(f"\n Filtering: max_len={args.max_len}")
    print(f"    Before: {original_size} sequences")
    print(f"    After:    {filtered_size} sequences")

    if filtered_size == 0:
        print("Error: all sequences filtered")
        return


    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=pad_collate)

    try:
        it = iter(loader)
        batch = next(it)
        print(f"\nFirst batch shapes:")
        for k, v in batch.items():
            if hasattr(v, 'shape'):
                print(f"  {k}: {v.shape}")

        paired_pos = (batch['partners'].abs() > 0.01).sum().item()
        total_pos = batch['mask'].sum().item()
        print(f"\nPairing statistics:")
        print(f"  Paired positions: {paired_pos}/{int(total_pos)} ({100 * paired_pos / max(1, total_pos):.1f}%)")

        seq_lengths = batch['mask'].sum(dim=1)
        print(
            f"  Sequence lengths: min={seq_lengths.min().item():.0f}, max={seq_lengths.max().item():.0f}, mean={seq_lengths.mean().item():.1f}")

    except Exception as e:
        print(f"✗ Failed to load batch: {repr(e)}")
        return

    print(f"\n{'=' * 40}")
    print("INITIALIZING MODELS")
    print(f"{'=' * 40}")

    generator = ResNetGeneratorConditional(
        latent_dim=args.latent_dim,
        struct_dim=3,
        partner_dim=1,
    ).to(device)
    initialize_weights(generator)

    critic = CriticConditional(
        struct_dim=3,
        partner_dim=1,
    ).to(device)

    g_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    c_params = sum(p.numel() for p in critic.parameters() if p.requires_grad)
    print(f"Generator parameters: {g_params:,}")
    print(f"Critic parameters: {c_params:,}")

    print(f"\nTesting generator")
    try:
        z = torch.randn(2, args.latent_dim, device=device)
        test_struct = batch['struct'][:2].to(device)
        test_partners = batch['partners'][:2].to(device)
        test_mask = batch['mask'][:2].to(device)

        gen_logits, _ = generator(z, test_struct, test_partners, test_mask, return_logits=True)
        print(f"  Logits shape: {gen_logits.shape}")
        print(f"  Logits range: [{gen_logits.min().item():.2f}, {gen_logits.max().item():.2f}]")

        gen_samples, _ = generator(z, test_struct, test_partners, test_mask, return_logits=False)
        is_onehot = (gen_samples.sum(dim=-1).round() == 1).all()
        print(f"  Samples shape: {gen_samples.shape}")
        print(f"  Valid one-hot: {is_onehot.item()}")

        base_dist = gen_samples.sum(dim=[0, 1])
        base_dist = base_dist / base_dist.sum()
        print(
            f" Test base dist: A={base_dist[0]:.2f} C={base_dist[1]:.2f} G={base_dist[2]:.2f} U={base_dist[3]:.2f}")

    except Exception as e:
        print(f" Generator test failed: {repr(e)}")
        import traceback
        traceback.print_exc()
        return

    print(f"\n")
    print("HYPERPARAMETERS")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Latent dim: {args.latent_dim}")
    print(f"  Max sequence length: {args.max_len}")
    print(f"  n_critic: {args.n_critic}")
    print(f"  lambda_gp: {args.lambda_gp}")
    print(f"  lambda_pair: {args.lambda_pair}")
    print(f"  lambda_diversity: {args.lambda_diversity}")
    print(f"  Pair weights: A-U=1.5 (bonus), G-C=1.0, G-U=0.5 (reduced)")
    print(f"  lr_generator: {args.lr_g}")
    print(f"  lr_critic: {args.lr_c}")
    print(f"  Initial tau: 1.0 (annealing to 0.5)")

    print("STARTING TRAINING")

    train_wgan_gp(generator, critic, loader, args, device)

    print(f"\n")
    print("TRAINING COMPLETE")
    print(f"Total time: {time.time() - t0:.1f}s")

if __name__ == "__main__":
    main()