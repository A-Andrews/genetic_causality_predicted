import argparse, time, torch, numpy as np
from transformers import AutoModel
from gpn.data import (
    GenomeMSA,
)  # pip install "gpn @ git+https://github.com/songlab-cal/gpn.git"

VOCAB = {c: i for i, c in enumerate("ACGT-")}  # A=0,C=1,G=2,T=3,gap=4


def slice_window(msa, chrom, pos1, ref, alt):
    """Return (ref_tensor, alt_tensor, attn_mask) for a 1-based SNV."""
    pos0 = pos1 - 1
    start = pos0 - 64
    end = pos0 + 64  # half-open
    X = msa.get_msa(chrom, start, end)  # (128, 90)
    if X[64, 0] == 4:
        raise ValueError("human gap at SNV column")
    keep = X[:, 0] != 4  # drop gap columns
    Xr, Xa = X.copy(), X.copy()
    Xr[64, 0] = VOCAB[ref]
    Xa[64, 0] = VOCAB[alt]
    return (
        torch.tensor(Xr[keep], dtype=torch.long),
        torch.tensor(Xa[keep], dtype=torch.long),
        torch.tensor(keep, dtype=torch.bool),
    )


def main(args):
    t0 = time.time()
    msa = GenomeMSA(args.msa)
    print(f"✅  opened MSA store; 90 species = {len(msa.species)==90}")
    r, a, m = slice_window(msa, args.chrom, args.pos, args.ref, args.alt)
    print(f"✅  sliced window: ref shape {tuple(r.shape)}, alt {tuple(a.shape)}")

    enc = AutoModel.from_pretrained(
        "songlab/gpn-msa-sapiens", add_pooling_layer=False
    ).to(args.device)
    enc.eval()
    torch.set_grad_enabled(False)

    hr = enc(
        r.unsqueeze(0).to(args.device), attention_mask=m.unsqueeze(0).to(args.device)
    ).last_hidden_state[:, 64, :]
    ha = enc(
        a.unsqueeze(0).to(args.device), attention_mask=m.unsqueeze(0).to(args.device)
    ).last_hidden_state[:, 64, :]

    cos = torch.nn.functional.cosine_similarity(hr, ha).item()
    print(f"Embeddings cosine(ref,alt) = {cos:.4f}")
    print(f"Total wall-clock time: {time.time()-t0:.2f} s")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="GPN-MSA quick test")
    p.add_argument("--msa", required=True, help="path or URL to 89-species Zarr store")
    p.add_argument("--chrom", default="17")
    p.add_argument("--pos", type=int, default=43106464)  # hg38 BRCA1 c.68_69del
    p.add_argument("--ref", default="C")
    p.add_argument("--alt", default="T")
    p.add_argument("--device", default="cpu")
    main(p.parse_args())
