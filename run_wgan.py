import argparse
from utils.init_device import init_cuda
from loaders.fasta_data_loader import FastDatasetRNA_Conditional, pad_collate
from torch.utils.data import DataLoader
from models.resnet_generator_rna import ResNetGeneratorConditional
from utils.init_weights import initialize_weights
from models.critic import CriticConditional
from train_wgan_gp import train_wgan_gp

def parse_args():
    parser = argparse.ArgumentParser(description="Run WGAN-GP for training RNA sequences")
    parser.add_argument("--data", type=str, default="data/test2.fa", help="Path to the RNA sequence data file")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--n_critic", type=int, default=5)
    parser.add_argument("--lambda_gp", type=float, default=10.0)
    parser.add_argument("--save_dir", type=str, default="saved_models")
    parser.add_argument("--log_dir", type=str, help="Path to the log file (optional)")
    parser.add_argument("--lr_g", type=float, default=0.0005)
    parser.add_argument("--lr_c", type=float, default=0.0001)
    return parser.parse_args()

def main():
    args = parse_args()
    device = init_cuda()

    print("Loading dataset...")
    dataset = FastDatasetRNA_Conditional(args.data)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=pad_collate)
    print(f"Dataset size: {len(dataset)} samples")

    # Generator i krytyk (warunkowane strukturą)
    generator = ResNetGeneratorConditional(args.latent_dim).to(device)
    initialize_weights(generator)
    critic = CriticConditional().to(device)

    # przekazujemy loader zamiast dataset
    train_wgan_gp(generator, critic, loader, args, device)

if __name__ == "__main__":
    main()
