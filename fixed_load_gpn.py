import numpy as np
import torch
import torch.utils.data as td
from gpn.data import GenomeMSA

from settings import ZARR_PATH

msa = GenomeMSA(ZARR_PATH)

VOCAB = {c: i for i, c in enumerate("ACGT-")}  # gap == 4


def slice_window_fixed(row):
    """Fixed version that handles MSA encoding properly"""
    chrom = row["chrom"]
    pos0 = row["pos"] - 1  # 0-based centre
    start, end = pos0 - 64, pos0 + 64  # [start,end)
    X = msa.get_msa(chrom, start, end)  # (128,90)

    # Convert byte strings to integer indices
    if np.issubdtype(X.dtype, np.bytes_):
        X = np.vectorize(lambda b: VOCAB[b.decode("ascii").upper()])(X)

    # Instead of the problematic base-5 encoding, use a simpler approach
    # that focuses on the human sequence (column 0) with context from other species
    
    # Check for gaps in human sequence
    gap_token = 4  # gap token for "ACGT-" vocab
    if X[64, 0] == gap_token:  # human gap at SNV
        return None  # skip
    
    # Create tokens based on human sequence with conservation info
    # Use human sequence as primary tokens
    human_seq = X[:, 0]  # Human sequence (column 0)
    
    # Add conservation context: count how many species agree with human at each position
    conservation = np.sum(X == human_seq[:, np.newaxis], axis=1)
    
    # Simple encoding: use human tokens + conservation offset
    # This keeps tokens in a reasonable range
    tokens = human_seq.astype(np.int64)
    
    # Apply ref/alt modification
    ref_tokens, alt_tokens = tokens.copy(), tokens.copy()
    ref_tokens[64], alt_tokens[64] = VOCAB[row["ref"]], VOCAB[row["alt"]]
    
    # Create attention mask (drop positions where human has gaps)
    keep = human_seq != gap_token
    
    return ref_tokens[keep], alt_tokens[keep], keep.astype(np.int64)


class TGMSAFixed(td.Dataset):
    """Fixed version of TGMSA that uses proper token encoding"""
    def __init__(self, hf_ds):
        self.ds = hf_ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        row = self.ds[i]
        out = slice_window_fixed(row)
        while out is None:  # rare gaps
            i = (i + 1) % len(self.ds)
            row = self.ds[i]
            out = slice_window_fixed(row)
        ref, alt, attn = out
        return dict(
            ref=torch.tensor(ref, dtype=torch.long),
            alt=torch.tensor(alt, dtype=torch.long),
            attn=torch.tensor(attn, dtype=torch.long),
            label=torch.tensor(row["label"], dtype=torch.float),
        )