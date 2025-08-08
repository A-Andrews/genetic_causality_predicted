import argparse
import logging
import time

import numpy as np
import torch
from gpn.data import (  # pip install "gpn @ git+https://github.com/songlab-cal/gpn.git"
    GenomeMSA,
)
from gpn.model import GPNRoFormerModel

from utils import setup_logger

VOCAB = {c: i for i, c in enumerate("ACGT-")}  # A=0,C=1,G=2,T=3,gap=4


def slice_window(msa, chrom, pos1, ref, alt):
    """Return (ref_tensor, alt_tensor, attn_mask) for a 1-based SNV."""
    pos0 = pos1 - 1
    start = pos0 - 64
    end = pos0 + 64  # half-open
    X = msa.get_msa(chrom, start, end)  # (128, 90)

    # Convert byte strings to integer indices if needed
    if X.dtype.kind == "S":  # byte string
        import numpy as np

        X = np.vectorize(lambda b: VOCAB[b.decode("ascii").upper()])(X)

    if X[64, 0] == 4:
        raise ValueError("human gap at SNV column")
    keep = X[:, 0] != 4  # drop gap columns

    # Use only human sequence (column 0) for simple tokenization
    human_seq = X[:, 0]
    Xr, Xa = human_seq.copy(), human_seq.copy()
    Xr[64] = VOCAB[ref]
    Xa[64] = VOCAB[alt]

    return (
        torch.tensor(Xr[keep], dtype=torch.long),
        torch.tensor(Xa[keep], dtype=torch.long),
        torch.tensor(keep, dtype=torch.bool),
    )


def main(args):
    t0 = time.time()
    msa = GenomeMSA(args.msa)
    r, a, m = slice_window(msa, args.chrom, args.pos, args.ref, args.alt)
    logging.info(f"sliced window: ref shape {tuple(r.shape)}, alt {tuple(a.shape)}")

    enc = GPNRoFormerModel.from_pretrained("songlab/gpn-msa-sapiens").to(args.device)
    enc.eval()
    torch.set_grad_enabled(False)

    hr = enc(
        r.unsqueeze(0).to(args.device), attention_mask=m.unsqueeze(0).to(args.device)
    ).last_hidden_state[:, 64, :]
    ha = enc(
        a.unsqueeze(0).to(args.device), attention_mask=m.unsqueeze(0).to(args.device)
    ).last_hidden_state[:, 64, :]

    cos = torch.nn.functional.cosine_similarity(hr, ha).item()
    logging.info(f"Embeddings cosine(ref,alt) = {cos:.4f}")
    logging.info(f"Total wall-clock time: {time.time()-t0:.2f} s")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="GPN-MSA quick test")
    p.add_argument("--msa", required=True, help="path or URL to 89-species Zarr store")
    p.add_argument("--chrom", default="17")
    p.add_argument("--pos", type=int, default=43106464)  # hg38 BRCA1 c.68_69del
    p.add_argument("--ref", default="C")
    p.add_argument("--alt", default="T")
    p.add_argument("--device", default="cpu")
    main(p.parse_args())
    setup_logger(1)  # No seed for this quick test
