#!/usr/bin/env python3

import argparse
import torch
from pathlib import Path
from Bio import SeqIO

from models.resnet_generator_rna import ResNetGeneratorConditional

def one_hot_struct(dotbracket: str):
    table = {'.': [1,0,0], '(': [0,1,0], ')': [0,0,1]}
    vecs = [table.get(ch, [1,0,0]) for ch in dotbracket.strip()]
    return torch.tensor(vecs, dtype=torch.float32)  # (L,3)

def one_hot_to_rna(seq_oh):
    bases = ['A','C','G','U']
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
            struct_rec = records[i+1]
            items.append((seq_rec.id, str(struct_rec.seq).strip()))
    else:
        raise ValueError("mode must be 'struct' or 'paired'")
    return items

# ---------------- main ----------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Path to generator .pth")
    p.add_argument("--input", required=True, help="Input FASTA (structures) or paired FASTA")
    p.add_argument("--mode", choices=["struct","paired"], default="struct", help="Input format")
    p.add_argument("--n_samples", type=int, default=3, help="Ile sekwencji wygenerować dla każdej struktury")
    p.add_argument("--latent_dim", type=int, default=256, help="latent dim (dopasuj do treningu)")
    p.add_argument("--out", default="generated.fa", help="Wyjściowy FASTA z wygenerowanymi sekwencjami")
    p.add_argument("--device", default=None, help="cpu or cuda (default autodetect)")
    p.add_argument("--batch_generate", type=int, default=16, help="Ile próbek na batch podczas generacji")
    p.add_argument("--strict_load", action="store_true", help="Użyć strict=True przy load_state_dict (domyślnie False)")
    args = p.parse_args()

    device = torch.device("cuda" if (args.device is None and torch.cuda.is_available()) else (args.device or "cpu"))
    print("Using device:", device)

    items = parse_fasta_structs(Path(args.input), args.mode)
    if not items:
        print("No structures found in input.")
        return

    state = torch.load(str(args.checkpoint), map_location=device)

    generator = ResNetGeneratorConditional(args.latent_dim)
    try:
        generator.load_state_dict(state, strict=args.strict_load)
    except RuntimeError as e:
        print("Warning: load_state_dict failed with strict=", args.strict_load)
        print("Error:", e)
        if not args.strict_load:
            generator.load_state_dict(state, strict=False)
            print("Loaded with strict=False (some keys skipped).")
    generator.to(device)
    generator.eval()

    out_f = open(args.out, "w")

    for header, struct in items:
        cond = one_hot_struct(struct).unsqueeze(0).to(device)
        total_to_gen = args.n_samples
        generated = []

        while total_to_gen > 0:
            cur = min(total_to_gen, args.batch_generate)
            zs = torch.randn(cur, args.latent_dim, device=device)
            with torch.no_grad():
                outs = generator(zs, cond.repeat(cur,1,1))
            for i in range(outs.size(0)):
                seq = one_hot_to_rna(outs[i].cpu())
                generated.append(seq)
            total_to_gen -= cur

        for i, seq in enumerate(generated):
            out_f.write(f">{header}_gen{i}\n{seq}\n")

    out_f.close()
    print("Output saved to", args.out)

if __name__ == "__main__":
    main()
