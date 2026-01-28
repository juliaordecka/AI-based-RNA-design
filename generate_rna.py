#!/usr/bin/env python3

import argparse
import torch
from pathlib import Path
from Bio import SeqIO

from models.resnet_generator_rna import ResNetGeneratorConditional


def parse_structure(dotbracket: str):

    L = len(dotbracket)

    table = {'.': [1, 0, 0], '(': [0, 1, 0], ')': [0, 0, 1]}
    roles = []
    for ch in dotbracket:
        roles.append(table.get(ch, [1, 0, 0]))  # Default to dot
    struct_oh = torch.tensor(roles, dtype=torch.float32)

    partner_offset = torch.zeros(L, 1, dtype=torch.float32)
    stack = []
    for i, ch in enumerate(dotbracket):
        if ch == '(':
            stack.append(i)
        elif ch == ')' and stack:
            j = stack.pop()
            offset = (i - j) / L
            partner_offset[j, 0] = offset
            partner_offset[i, 0] = -offset

    return struct_oh, partner_offset


def one_hot_to_rna(seq_oh):
    bases = ['A', 'C', 'G', 'U']
    if seq_oh.dim() == 3:
        seq_oh = seq_oh[0]
    indices = seq_oh.argmax(dim=-1).tolist()
    return ''.join(bases[i] for i in indices)


def parse_fasta_structs(path: Path, mode: str):

    records = list(SeqIO.parse(str(path), "fasta"))
    items = []

    if mode == "struct":
        for rec in records:
            items.append((rec.id, str(rec.seq).strip()))
    elif mode == "paired":
        if len(records) % 2 != 0:
            raise ValueError("paired mode expects even number of FASTA records (sequence, structure pairs).")
        for i in range(0, len(records), 2):
            seq_rec = records[i]
            struct_rec = records[i + 1]
            items.append((seq_rec.id, str(struct_rec.seq).strip()))
    else:
        raise ValueError("mode must be 'struct' or 'paired'")

    return items


def main():
    p = argparse.ArgumentParser(description="Generate RNA sequences for given structures")
    p.add_argument("--checkpoint", required=True, help="Path to generator .pth checkpoint")
    p.add_argument("--input", required=True, help="Input FASTA with structures")
    p.add_argument("--mode", choices=["struct", "paired"], default="struct",
                   help="Input format: 'struct' (each record is structure) or 'paired' (seq, struct pairs)")
    p.add_argument("--n_samples", type=int, default=3, help="Number of sequences per structure")
    p.add_argument("--latent_dim", type=int, default=256, help="Latent dimension (must match training)")
    p.add_argument("--embed_dim", type=int, default=128, help="Embedding dimension (must match training)")
    p.add_argument("--out", default="generated.fa", help="Output FASTA file")
    p.add_argument("--device", default=None, help="Device: 'cpu' or 'cuda' (auto-detect if None)")
    p.add_argument("--batch_size", type=int, default=16, help="Batch size for generation")
    p.add_argument("--strict_load", action="store_true", help="Use strict=True for load_state_dict")
    args = p.parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    items = parse_fasta_structs(Path(args.input), args.mode)
    if not items:
        print("No structures found in input.")
        return
    print(f"Found {len(items)} structures to process")

    print(f"Loading checkpoint: {args.checkpoint}")
    state = torch.load(str(args.checkpoint), map_location=device)

    generator = ResNetGeneratorConditional(
        latent_dim=args.latent_dim,
        struct_dim=3,
        partner_dim=1,
        embed_dim=args.embed_dim,
        n_blocks=4
    )

    try:
        generator.load_state_dict(state, strict=args.strict_load)
        print(f"Loaded checkpoint (strict={args.strict_load})")
    except RuntimeError as e:
        print(f"Warning: load_state_dict failed with strict={args.strict_load}")
        print(f"Error: {e}")
        if not args.strict_load:
            try:
                generator.load_state_dict(state, strict=False)
                print("Loaded with strict=False (some keys may be skipped)")
            except RuntimeError as e2:
                print(f"Failed even with strict=False: {e2}")
                return
        else:
            return

    generator.to(device)
    generator.eval()

    out_f = open(args.out, "w")
    total_generated = 0

    for idx, (header, struct_str) in enumerate(items):
        print(f"Processing [{idx + 1}/{len(items)}]: {header} (length={len(struct_str)})")

        struct_oh, partners = parse_structure(struct_str)
        L = len(struct_str)

        struct_batch = struct_oh.unsqueeze(0).to(device)
        partners_batch = partners.unsqueeze(0).to(device)
        mask_batch = torch.ones(1, L, device=device)

        generated = []
        remaining = args.n_samples

        while remaining > 0:
            cur_batch = min(remaining, args.batch_size)

            z = torch.randn(cur_batch, args.latent_dim, device=device)

            struct_exp = struct_batch.expand(cur_batch, -1, -1)
            partners_exp = partners_batch.expand(cur_batch, -1, -1)
            mask_exp = mask_batch.expand(cur_batch, -1)

            with torch.no_grad():
                samples, _ = generator(z, struct_exp, partners_exp, mask_exp, return_logits=False)

            for i in range(samples.size(0)):
                seq = one_hot_to_rna(samples[i].cpu())
                generated.append(seq)

            remaining -= cur_batch

        for i, seq in enumerate(generated):
            out_f.write(f">{header}_gen{i}\n{seq}\n")

        total_generated += len(generated)

    out_f.close()
    print(f"\nGenerated {total_generated} sequences")
    print(f"Output saved to: {args.out}")


if __name__ == "__main__":
    main()