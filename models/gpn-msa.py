import argparse
import logging
import os
import random
import warnings
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from gpn.model import GPNRoFormerModel
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, WeightedRandomSampler

from data.load_gpn_data import TGMSAFixed as TGMSA
from utils.data_saving import save_args, save_performance_metrics, setup_logger

warnings.filterwarnings(
    "ignore",
    message="Object at .snakemake_timestamp is not recognized as a component of a Zarr hierarchy.",
)


@dataclass
class TrainArgs:
    """Command line arguments for training."""

    max_batches: int | None = None


def parse_args() -> TrainArgs:
    parser = argparse.ArgumentParser(description="Train GPN-MSA classifier")
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Limit the number of batches per epoch for faster experiments.",
    )
    return TrainArgs(**vars(parser.parse_args()))


args = parse_args()

setup_logger(None)  # Initialize logger

MODEL_NAME = "gpn-msa-classifier"  # or any name you want
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# Save CLI arguments (now robust to argparse.Namespace)
save_args(args, MODEL_NAME, TIMESTAMP)

# Deterministic behaviour helps debug training issues
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)

logging.info(f"Using seed {SEED} for reproducibility")

# Load one of the four TraitGym configurations.
# Keep the full dataset in memory for faster filtering during the CV loop.
tg = load_dataset(
    "songlab/TraitGym",
    "complex_traits",
    split="test",
    keep_in_memory=True,
)

logging.info(f"Loaded TraitGym dataset with {len(tg)} records")

# Get a list of all unique chromosomes to use for cross-validation.
# We remove chromosome 'Y' as it is a sex chromosome.
chromosomes = sorted(list(set(tg["chrom"])))
if "Y" in chromosomes:
    chromosomes.remove("Y")
logging.info(f"Using chromosomes for CV: {chromosomes}")

# List to store AUPRC results for each fold
auprc_scores = []
n_epochs = 8
per_fold = []

# Loop through each chromosome to use as the validation set
for test_chrom in chromosomes:
    logging.info("=" * 50)
    logging.info(
        f"Starting Leave-One-Chromosome-Out fold: Holding out chromosome {test_chrom}"
    )

    # Split the dataset into training and validation sets for the current fold
    train_data = tg.filter(lambda r: r["chrom"] != test_chrom)
    val_data = tg.filter(lambda r: r["chrom"] == test_chrom)

    train = TGMSA(train_data)
    val = TGMSA(val_data)

    n_val = len(val)
    n_val_pos = int(np.sum(val.ds["label"]))
    n_val_neg = n_val - n_val_pos
    logging.info(
        f"LOCO fold chr{test_chrom}: val size={n_val} (pos={n_val_pos}, neg={n_val_neg})"
    )

    logging.info(f"Training set size: {len(train)}, Validation set size: {len(val)}")

    # Step 1: Extract all labels from the training dataset
    # We create a new cache path for each fold to avoid using stale data.
    labels_cache_dir = os.path.join("data_consolidation", "cv_labels")
    os.makedirs(labels_cache_dir, exist_ok=True)
    label_cache_path = os.path.join(
        labels_cache_dir, f"train_labels_fold_{test_chrom}.pt"
    )

    try:
        labels = torch.load(label_cache_path)
    except FileNotFoundError:
        logging.info("Labels not found, generating and saving...")
        labels = torch.tensor(train.ds["label"])
        torch.save(labels, label_cache_path)

    logging.info(f"Loaded {len(labels)} labels from {label_cache_path}")

    # Step 2: Compute class counts and weights
    n_pos = (labels == 1).sum().item()
    n_neg = (labels == 0).sum().item()
    weight_for_0 = 1.0 / n_neg
    weight_for_1 = 1.0 / n_pos

    sample_weights = torch.where(labels == 1, weight_for_1, weight_for_0)

    # Step 3: Create the WeightedRandomSampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),  # Or more for even more positive upsampling
        replacement=True,
    )

    # Step 4: Build the DataLoader with the sampler (not shuffle!)
    train_loader = DataLoader(train, batch_size=32, sampler=sampler, drop_last=False)
    val_loader = DataLoader(val, batch_size=32)

    logging.info(
        f"Positives: {n_pos}, Negatives: {n_neg}, pos_weight: {n_neg / n_pos:.2f}"
    )

    # Initialize model, optimizer, scheduler, and loss function for this fold
    enc = GPNRoFormerModel.from_pretrained("songlab/gpn-msa-sapiens")
    for p in enc.parameters():  # freeze for warm-up
        p.requires_grad = False

    class SNPClassifier(nn.Module):
        def __init__(self, encoder, d_model=enc.config.hidden_size):
            super().__init__()
            self.encoder = encoder
            self.cls = nn.Sequential(
                nn.LayerNorm(d_model * 2),
                nn.Linear(d_model * 2, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
            )

        def forward(self, ref, alt, attn):
            extended_attn = self.encoder.get_extended_attention_mask(
                attn, ref.shape, device=ref.device
            )

            h_ref = self.encoder(ref, attention_mask=extended_attn).last_hidden_state[
                :, 64, :
            ]
            h_alt = self.encoder(alt, attention_mask=extended_attn).last_hidden_state[
                :, 64, :
            ]
            x = torch.cat([h_ref, h_alt - h_ref], dim=-1)
            return self.cls(x).squeeze(-1)

    model = SNPClassifier(enc).cuda()
    opt = torch.optim.AdamW(
        [
            {"params": model.cls.parameters(), "lr": 1e-3},
            {"params": enc.parameters(), "lr": 1e-5},
        ],
        weight_decay=1e-2,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([n_neg / n_pos]).cuda())

    # Training loop for the current fold
    for epoch in range(n_epochs):
        model.train()
        if epoch == 3:
            for p in enc.encoder.layer[-4:].parameters():  # unfreeze top 4 layers
                p.requires_grad = True
        running_loss = 0.0
        train_preds, train_labels = [], []

        for batch_idx, batch in enumerate(train_loader):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break
            opt.zero_grad()
            out = model(batch["ref"].cuda(), batch["alt"].cuda(), batch["attn"].cuda())
            loss = lossf(out, batch["label"].cuda())
            loss.backward()
            opt.step()
            running_loss += loss.item()
            probs = torch.sigmoid(out.detach())
            train_preds.append(probs.cpu())
            train_labels.append(batch["label"])
        sched.step()

    # Evaluation for the current fold
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for val_idx, b in enumerate(val_loader):
            if args.max_batches is not None and val_idx >= args.max_batches:
                break
            p = torch.sigmoid(model(b["ref"].cuda(), b["alt"].cuda(), b["attn"].cuda()))
            preds.append(p.cpu())
            labels.append(b["label"])

    if preds and labels:
        preds_t = torch.cat(preds)
        labels_t = torch.cat(labels)
        fold_auprc = average_precision_score(labels_t.numpy(), preds_t.numpy())
        logging.info(f"LOCO-chr{test_chrom} AUPRC = {fold_auprc:.3f} (n_val={n_val})")
        per_fold.append((test_chrom, float(fold_auprc), int(n_val)))


# Calculate and print the final average AUPRC
if not per_fold:
    logging.warning("No per-fold results gathered; cannot compute LOCO.")
else:
    chrs = [c for c, _, _ in per_fold]
    scores = np.array([s for _, s, _ in per_fold], dtype=float)
    weights = np.array(
        [n for _, _, n in per_fold], dtype=float
    )  # number of val examples per chr

    # Size-weighted mean AUPRC (TraitGym-style)
    weighted_mean = np.average(scores, weights=weights)

    # Optional: chromosome-level bootstrap SE of the weighted mean
    def bootstrap_weighted_mean(scores, weights, iters=2000, seed=42):
        rng = np.random.default_rng(seed)
        k = len(scores)
        boots = np.empty(iters, dtype=float)
        for b in range(iters):
            idx = rng.integers(0, k, size=k, endpoint=False)  # resample chromosomes
            boots[b] = np.average(scores[idx], weights=weights[idx])
        return boots.std(ddof=1)

    weighted_se = bootstrap_weighted_mean(scores, weights, iters=2000)

    # Also log the plain (unweighted) mean for reference
    unweighted_mean = scores.mean()
    unweighted_std = scores.std(ddof=1)

    logging.info("=" * 50)
    logging.info("Cross-Validation Complete.")
    logging.info(
        "Per-chrom AUPRC: "
        + ", ".join([f"chr{c}:{s:.3f}(n={n})" for (c, s, n) in per_fold])
    )
    logging.info(
        f"LOCO AUPRC (size-weighted): {weighted_mean:.3f} ± {weighted_se:.3f} (bootstrap SE)"
    )
    logging.info(
        f"LOCO AUPRC (unweighted):    {unweighted_mean:.3f} ± {unweighted_std:.3f} (SD)"
    )
    logging.info("=" * 50)

    performance_metrics = {
        "per_chrom": [
            {"chrom": str(c), "auprc": float(s), "n_val": int(n)}
            for (c, s, n) in per_fold
        ],
        "summary": {
            "weighted_mean_auprc": float(weighted_mean),
            "weighted_se": float(weighted_se),
            "unweighted_mean_auprc": float(unweighted_mean),
            "unweighted_sd": float(unweighted_std),
            "n_total_val": int(weights.sum()),
            "n_chromosomes": int(len(per_fold)),
            # "imbalance_strategy": IMBALANCE_STRATEGY,  # uncomment if you set it
            "epochs": int(n_epochs),
            "max_batches": args.max_batches,
        },
    }

    save_performance_metrics(performance_metrics, MODEL_NAME, TIMESTAMP)
