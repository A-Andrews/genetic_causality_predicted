import numpy as np
import torch
import torch.utils.data as td
from gpn.data import GenomeMSA

from settings import ZARR_PATH

msa = GenomeMSA(ZARR_PATH)

VOCAB = {c: i for i, c in enumerate("ACGT-")}  # gap == 4


def slice_window(row):
    chrom = row["chrom"]
    pos0 = row["pos"] - 1  # 0-based centre
    start, end = pos0 - 64, pos0 + 64  # [start,end)
    X = msa.get_msa(chrom, start, end)  # (128,90)

    # Older versions of the MSA may yield arrays of byte strings; convert
    # to integer indices expected by the model.
    if np.issubdtype(X.dtype, np.bytes_):
        X = np.vectorize(lambda b: VOCAB[b.decode("ascii")])(X)

    if X[64, 0] == 4:  # human gap at SNV
        return None  # skip
    ref, alt = X.copy(), X.copy()
    ref[64, 0], alt[64, 0] = VOCAB[row["ref"]], VOCAB[row["alt"]]
    keep = X[:, 0] != 4  # drop gap cols
    return ref[keep], alt[keep], keep


class TGMSA(td.Dataset):
    def __init__(self, hf_ds):
        self.ds = hf_ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        row = self.ds[i]
        out = slice_window(row)
        while out is None:  # rare gaps
            i = (i + 1) % len(self.ds)
            row = self.ds[i]
            out = slice_window(row)
        ref, alt, attn = out
        return dict(
            ref=torch.tensor(ref, dtype=torch.long),
            alt=torch.tensor(alt, dtype=torch.long),
            attn=torch.tensor(attn, dtype=torch.long),
            label=torch.tensor(row["label"], dtype=torch.float),
        )
