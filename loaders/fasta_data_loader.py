import re
import random
import torch
from torch.utils.data import Dataset
from Bio import SeqIO

def convert_to_rna(seq: str) -> str:
    return seq.replace("T", "U").replace("t", "u")

def iupac_converter(nucleotide: str) -> str:
    nucleotide_map = {
        'R': ['A', 'G'],
        'Y': ['C', 'U'],
        'S': ['G', 'C'],
        'W': ['A', 'U'],
        'K': ['G', 'U'],
        'M': ['A', 'C'],
        'B': ['C', 'G', 'U'],
        'D': ['A', 'G', 'U'],
        'H': ['A', 'C', 'U'],
        'V': ['A', 'C', 'G'],
        'N': ['A', 'C', 'G', 'U']
    }
    if nucleotide in nucleotide_map:
        return random.choice(nucleotide_map[nucleotide])
    return nucleotide

def one_hot_seq(sequence: str):
    table = {'A':[1,0,0,0], 'C':[0,1,0,0], 'G':[0,0,1,0], 'U':[0,0,0,1]}
    # jeśli występuje znak nieznany (np. N) zamieniamy przez losowanie IUPAC wcześniej
    out = []
    for n in sequence:
        if n not in table:
            # w razie czego traktujemy jako N -> zastąp losowym nukleotydem
            n = random.choice(['A','C','G','U'])
        out.append(table[n])
    return torch.tensor(out, dtype=torch.float32)

def one_hot_struct(struct: str):
    table = {'.':[1,0,0], '(': [0,1,0], ')':[0,0,1]}
    out = []
    for s in struct:
        if s not in table:
            # jeśli pojawi się nietypowy znak, traktuj go jako kropkę (nieparowany)
            s = '.'
        out.append(table[s])
    return torch.tensor(out, dtype=torch.float32)


class FastDatasetRNA_Conditional(Dataset):
    """
    Loader for paired (sequence, structure) RNA samples.
    Supports two formats:
      1) pair-of-records: >id\nSEQUENCE\n>id2\nSTRUCTURE\n (two FASTA records per example)
      2) single-record-with-two-lines: >id\nSEQUENCE\nSTRUCTURE\n (one FASTA record; second line is structure)
    It strips all whitespace characters, preserves dot-bracket chars, resolves IUPAC, returns one-hot tensors.
    Padding is done in pad_collate().
    """
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = []
        self._load_pairs()

    def _clean(self, s: str) -> str:
        # usuwa wszystkie whitespace (space, tab, \r, \n)
        return re.sub(r"\s+", "", s)

    def _looks_like_sequence(self, s: str) -> bool:
        return bool(re.fullmatch(r"[ACGUNRYKMSWBDHVacgunrykmswbdhv]+", s))

    def _looks_like_structure(self, s: str) -> bool:
        return bool(re.fullmatch(r"[.\(\)\[\]<>]+", s))

    def _split_seq_struct_if_combined(self, text: str):
        """
        Jeśli text zawiera na końcu fragment składający się tylko z .()[]<>,
        uznajemy to za strukturę i zwracamy (sequence_prefix, structure_suffix).
        W przeciwnym razie zwracamy (None, None).
        """
        m = re.search(r"([.\(\)\[\]<>]+)$", text)
        if m:
            struct = m.group(1)
            seq = text[: -len(struct)]
            if seq and all(ch.upper() in "ACGUNRYKMSWBDHVacgunrykmswbdhv" for ch in seq):
                return seq, struct
        return None, None

    def _load_pairs(self):
        records = list(SeqIO.parse(self.file_path, "fasta"))
        if not records:
            raise ValueError(f"No records found in {self.file_path}")

        # Try to detect format:
        # - easiest: every record is pure sequence or pure structure (two-records-per-example)
        types = []
        cleaned_seqs = []
        for rec in records:
            s = self._clean(str(rec.seq))
            cleaned_seqs.append(s)
            if self._looks_like_sequence(s):
                types.append("seq_only")
            elif self._looks_like_structure(s):
                types.append("struct_only")
            else:
                # mixed or unknown
                types.append("mixed")

        # Case A: alternating seq_only, struct_only -> pairwise
        if len(records) % 2 == 0 and all(t in ("seq_only","struct_only") for t in types):
            # check pattern seq, struct, seq, struct, ...
            ok_pairwise = all(types[i]=="seq_only" and types[i+1]=="struct_only" for i in range(0,len(types),2))
            if ok_pairwise:
                for i in range(0, len(records), 2):
                    seq_raw = cleaned_seqs[i]
                    struct_raw = cleaned_seqs[i+1]
                    # convert IUPAC and T->U
                    seq_raw = convert_to_rna(seq_raw.upper())
                    seq_raw = "".join(iupac_converter(ch) for ch in seq_raw)
                    if len(seq_raw) != len(struct_raw):
                        raise ValueError(f"Sequence and structure length mismatch for {records[i].id} ({len(seq_raw)} != {len(struct_raw)})")
                    self.data.append({
                        "id": records[i].id,
                        "seq": seq_raw,
                        "struct": struct_raw,
                        "len": len(seq_raw)
                    })
                return

        # Case B: each record contains sequence line followed by structure line (combined)
        # Try splitting each record by trailing dot-bracket suffix
        all_split_ok = True
        for idx, s in enumerate(cleaned_seqs):
            seq_part, struct_part = self._split_seq_struct_if_combined(s)
            if seq_part is None:
                all_split_ok = False
                break

        if all_split_ok:
            for rec, s in zip(records, cleaned_seqs):
                seq_part, struct_part = self._split_seq_struct_if_combined(s)
                seq_raw = convert_to_rna(seq_part.upper())
                seq_raw = "".join(iupac_converter(ch) for ch in seq_raw)
                if len(seq_raw) != len(struct_part):
                    raise ValueError(f"Sequence and structure length mismatch for {rec.id} after split ({len(seq_raw)} != {len(struct_part)})")
                self.data.append({
                    "id": rec.id,
                    "seq": seq_raw,
                    "struct": struct_part,
                    "len": len(seq_raw)
                })
            return

        # Fallback: try pairwise by stepping 2 regardless (old behavior) but with helpful error
        if len(records) % 2 == 0:
            for i in range(0, len(records), 2):
                seq_raw = self._clean(str(records[i].seq)).upper()
                struct_raw = self._clean(str(records[i+1].seq))
                seq_raw = convert_to_rna(seq_raw)
                seq_raw = "".join(iupac_converter(ch) for ch in seq_raw)
                if len(seq_raw) != len(struct_raw):
                    raise ValueError(f"Sequence and structure length mismatch for {records[i].id} ({len(seq_raw)} != {len(struct_raw)}) - file format ambiguous")
                self.data.append({
                    "id": records[i].id,
                    "seq": seq_raw,
                    "struct": struct_raw,
                    "len": len(seq_raw)
                })
            return

        # If we get here, format is ambiguous / unsupported
        raise ValueError("Unsupported FASTA layout: loader could not determine pairing. "
                         "Expected either (seq_record, struct_record) pairs or single-records "
                         "where sequence is followed by dot-bracket structure on the same record.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        seq_oh = one_hot_seq(entry["seq"])
        struct_oh = one_hot_struct(entry["struct"])
        return {"id": entry["id"], "seq": seq_oh, "struct": struct_oh, "len": entry["len"]}


def pad_collate(batch):
    max_len = max(item["len"] for item in batch)
    seqs, structs, masks = [], [], []
    for item in batch:
        L = item["len"]
        pad_seq = torch.zeros(max_len, 4)
        pad_seq[:L] = item["seq"]
        pad_struct = torch.zeros(max_len, 3)
        pad_struct[:L] = item["struct"]
        mask = torch.zeros(max_len)
        mask[:L] = 1
        seqs.append(pad_seq)
        structs.append(pad_struct)
        masks.append(mask)
    return {"seq": torch.stack(seqs), "struct": torch.stack(structs), "mask": torch.stack(masks)}
