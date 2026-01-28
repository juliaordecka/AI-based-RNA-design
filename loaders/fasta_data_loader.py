import random
import re
import torch
from torch.utils.data import Dataset

def convert_to_rna(seq: str) -> str:
    return seq.replace("T", "U").replace("t", "u")

def iupac_to_nucleotide(n: str) -> str:
    iupac = {
        'R': ['A', 'G'], 'Y': ['C', 'U'], 'S': ['G', 'C'],
        'W': ['A', 'U'], 'K': ['G', 'U'], 'M': ['A', 'C'],
        'B': ['C', 'G', 'U'], 'D': ['A', 'G', 'U'],
        'H': ['A', 'C', 'U'], 'V': ['A', 'C', 'G'],
        'N': ['A', 'C', 'G', 'U']
    }
    n = n.upper()
    if n in iupac:
        return random.choice(iupac[n])
    return n


def one_hot_seq(sequence: str):
    table = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'U': [0, 0, 0, 1]}
    out = []
    for n in sequence:
        if n not in table:
            n = random.choice(['A', 'C', 'G', 'U'])
        out.append(table[n])
    return torch.tensor(out, dtype=torch.float32)


def one_hot_struct(struct: str):
    out = []
    for s in struct:
        if s in '([{<':
            out.append([0, 1, 0])  # open
        elif s in ')]}>':
            out.append([0, 0, 1])  # close
        else:
            out.append([1, 0, 0])  # dot
    return torch.tensor(out, dtype=torch.float32)


def dotbracket_to_partners(struct: str):
    openers = {'(': ')', '[': ']', '{': '}', '<': '>'}
    closers = {v: k for k, v in openers.items()}
    stacks = {o: [] for o in openers}
    partners = [-1] * len(struct)

    for i, ch in enumerate(struct):
        if ch in openers:
            stacks[ch].append(i)
        elif ch in closers:
            opener = closers[ch]
            if stacks[opener]:
                j = stacks[opener].pop()
                partners[i] = j
                partners[j] = i
    return partners


def partners_to_offsets(partners: list):
    L = len(partners)
    offsets = []
    for i, p in enumerate(partners):
        if p == -1:
            offsets.append([0.0])
        else:
            offsets.append([(p - i) / L])
    return torch.tensor(offsets, dtype=torch.float32)


def split_seq_struct(combined: str):

    struct_chars = set('.()[]{}<>')
    seq_chars = set('ACGUTNRYKMSWBDHVacgutnrykmswbdhv')

    for i, ch in enumerate(combined):
        if ch in struct_chars:
            remaining = combined[i:]
            if all(c in struct_chars for c in remaining):
                return combined[:i], combined[i:]

    match = re.search(r'([.(){}\[\]<>]+)$', combined)
    if match:
        struct = match.group(1)
        seq = combined[:-len(struct)]
        return seq, struct

    return combined, ""

class FastDatasetRNA_Conditional(Dataset):
    """
    Loader for 3-line format:
        >header
        SEQUENCE (may be multi-line)
        STRUCTURE (may be multi-line)
    """

    def __init__(self, file_path):
        self.data = []
        self._load(file_path)

    def _load(self, file_path):
        with open(file_path, 'r') as f:
            content = f.read()

        content = content.replace('\r', '')

        sections = re.split(r'^(?=>)', content, flags=re.MULTILINE)

        for section in sections:
            section = section.strip()
            if not section or not section.startswith('>'):
                continue

            lines = section.split('\n')
            header = lines[0][1:].strip()

            combined = ''.join(line.strip() for line in lines[1:])

            if not combined:
                continue

            seq_raw, struct_raw = split_seq_struct(combined)

            if not seq_raw or not struct_raw:
                print(f"Warning: could not split seq/struct for {header}, skipping")
                continue

            seq_raw = convert_to_rna(seq_raw.upper())
            seq_raw = ''.join(iupac_to_nucleotide(n) for n in seq_raw)

            if len(seq_raw) != len(struct_raw):
                print(f"Warning: length mismatch for {header}: "
                      f"seq={len(seq_raw)}, struct={len(struct_raw)}, skipping")
                continue

            partners = dotbracket_to_partners(struct_raw)

            self.data.append({
                "id": header,
                "seq": seq_raw,
                "struct": struct_raw,
                "partners": partners,
                "len": len(seq_raw)
            })

        print(f"Loaded {len(self.data)} samples from {file_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        return {
            "id": entry["id"],
            "seq": one_hot_seq(entry["seq"]),
            "struct": one_hot_struct(entry["struct"]),
            "partners": partners_to_offsets(entry["partners"]),
            "len": entry["len"]
        }

def pad_collate(batch):
    max_len = max(item["len"] for item in batch)

    seqs, structs, masks, partners = [], [], [], []

    for item in batch:
        L = item["len"]

        pad_seq = torch.zeros(max_len, 4)
        pad_seq[:L] = item["seq"]
        seqs.append(pad_seq)

        pad_struct = torch.zeros(max_len, 3)
        pad_struct[:L] = item["struct"]
        structs.append(pad_struct)

        mask = torch.zeros(max_len)
        mask[:L] = 1
        masks.append(mask)

        pad_part = torch.zeros(max_len, 1)
        pad_part[:L] = item["partners"]
        partners.append(pad_part)

    return {
        "seq": torch.stack(seqs),
        "struct": torch.stack(structs),
        "mask": torch.stack(masks),
        "partners": torch.stack(partners)
    }

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        dataset = FastDatasetRNA_Conditional(sys.argv[1])

        if len(dataset) > 0:
            print(f"\nFirst 3 samples:")
            for i in range(min(3, len(dataset))):
                d = dataset.data[i]
                print(f"  {d['id']}: len={d['len']}")

            lengths = [d['len'] for d in dataset.data]
            print(f"\nLength stats: min={min(lengths)}, max={max(lengths)}, mean={sum(lengths) / len(lengths):.1f}")
    else:
        print("Usage: python fasta_data_loader_simple.py <file.fasta>")