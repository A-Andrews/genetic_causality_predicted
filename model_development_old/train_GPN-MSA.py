import argparse
import logging
import os
import random
import warnings

import gpn.model
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from gpn.model import GPNRoFormerModel
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, WeightedRandomSampler

from data_consolidation.load_gpn import TGMSAFixed as TGMSA
from utils import setup_logger

warnings.filterwarnings(
    "ignore",
    message="Object at .snakemake_timestamp is not recognized as a component of a Zarr hierarchy.",
)

parser = argparse.ArgumentParser(description="Train GPN-MSA classifier")
parser.add_argument(
    "--max-batches",
    type=int,
    default=None,
    help="Limit the number of batches per epoch for faster experiments.",
)
args = parser.parse_args()

setup_logger(None)  # Initialize logger

# Deterministic behaviour helps debug training issues
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)

logging.info(f"Using seed {SEED} for reproducibility")

# Load one of the four TraitGym configurations.
# Pick the "_full" variant if you want the 9 × matched negatives.
tg = load_dataset(
    "songlab/TraitGym",
    "complex_traits_full",  # or "complex_traits", "mendelian_traits(_full)"
    split="test",  # TraitGym uses "test" for the full label set
)

logging.info(f"Loaded TraitGym dataset with {len(tg)} records")

train = TGMSA(tg.filter(lambda r: r["chrom"] != "21"))  # leave-chr-21-out
val = TGMSA(tg.filter(lambda r: r["chrom"] == "21"))

logging.info(f"Training set size: {len(train)}, Validation set size: {len(val)}")

# Step 1: Extract all labels from your dataset
label_cache_path = os.path.join("data_consolidation", "train_labels.pt")
try:
    labels = torch.load(label_cache_path)
except FileNotFoundError:
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
train_loader = DataLoader(train, batch_size=32, sampler=sampler, drop_last=True)
val_loader = DataLoader(val, batch_size=32)


logging.info(f"Positives: {n_pos}, Negatives: {n_neg}, pos_weight: {n_neg / n_pos:.2f}")

# Debug: Test data loading
("Testing data loader...")
test_batch = next(iter(train_loader))
logging.info(f"Sample batch:")
logging.info(
    f"  ref shape: {test_batch['ref'].shape}, dtype: {test_batch['ref'].dtype}"
)
logging.info(
    f"  alt shape: {test_batch['alt'].shape}, dtype: {test_batch['alt'].dtype}"
)
logging.info(
    f"  attn shape: {test_batch['attn'].shape}, dtype: {test_batch['attn'].dtype}"
)
logging.info(
    f"  label shape: {test_batch['label'].shape}, dtype: {test_batch['label'].dtype}"
)
logging.info(f"  ref sample values: {test_batch['ref'][0][:10]}")
logging.info(f"  alt sample values: {test_batch['alt'][0][:10]}")
logging.info(f"  ref range: [{test_batch['ref'].min()}, {test_batch['ref'].max()}]")
logging.info(f"  alt range: [{test_batch['alt'].min()}, {test_batch['alt'].max()}]")

enc = GPNRoFormerModel.from_pretrained("songlab/gpn-msa-sapiens")  # 86 M params
logging.info(
    f"Loaded GPN-MSA model with {sum(p.numel() for p in enc.parameters())} parameters"
)
logging.info(f"Model architecture: {enc}")

# Debug: Print model configuration
logging.info(f"Model config:")
logging.info(f"  vocab_size: {enc.config.vocab_size}")
logging.info(f"  hidden_size: {enc.config.hidden_size}")
logging.info(
    f"  max_position_embeddings: {getattr(enc.config, 'max_position_embeddings', 'N/A')}"
)

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
        # RoFormerEncoder expects an "extended" attention mask of shape
        # (batch, 1, 1, seq_len) with 0s for tokens to attend to and large
        # negative values for positions that should be masked.  The
        # `GPNRoFormerModel` wrapper does not internally expand the 2D mask
        # that we construct in the data loader, which previously led to a
        # broadcast error once the batch dimension differed from the sequence
        # length (e.g. batch size 32 vs. sequence length 128).
        #
        # Expand the mask here using the helper provided by the underlying
        # pretrained model so that it matches the expected shape.
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
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=6)
lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([n_neg / n_pos]).cuda())

for epoch in range(8):
    model.train()
    # 1) warm-up epochs 0-2 with encoder frozen
    if epoch == 3:
        for p in enc.encoder.layer[-4:].parameters():  # unfreeze top 4 layers
            p.requires_grad = True
    running_loss = 0.0
    train_preds, train_labels = [], []
    for batch_idx, batch in enumerate(train_loader):
        print(f"Mean label in batch: {batch['label'].mean().item():.3f}")
        if args.max_batches is not None and batch_idx >= args.max_batches:
            break
        opt.zero_grad()

        # Debug: Check input tensor properties
        logging.info(f"Batch {batch_idx}:")
        logging.info(
            f"  ref shape: {batch['ref'].shape}, range: [{batch['ref'].min()}, {batch['ref'].max()}]"
        )
        logging.info(
            f"  alt shape: {batch['alt'].shape}, range: [{batch['alt'].min()}, {batch['alt'].max()}]"
        )
        logging.info(
            f"  attn shape: {batch['attn'].shape}, dtype: {batch['attn'].dtype}"
        )

        # Check for invalid token indices
        if (
            batch["ref"].max() >= enc.config.vocab_size
            or batch["alt"].max() >= enc.config.vocab_size
        ):
            logging.info(
                f"WARNING: Token indices exceed vocab_size ({enc.config.vocab_size})"
            )
            logging.info(
                f"  ref max: {batch['ref'].max()}, alt max: {batch['alt'].max()}"
            )

        try:
            out = model(batch["ref"].cuda(), batch["alt"].cuda(), batch["attn"].cuda())
            loss = lossf(out, batch["label"].cuda())
            loss.backward()
            logging.info(f"cls grad norm:, {model.cls[1].weight.grad.norm().item()}")
            for i, layer in enumerate(enc.encoder.layer[-4:]):
                grad_norm = (
                    layer.output.dense.weight.grad.norm().item()
                    if layer.output.dense.weight.grad is not None
                    else 0.0
                )
                logging.info(f"Layer {i} grad norm: {grad_norm:.4f}")
            opt.step()
            logging.info(f"cls grad mean: {model.cls[1].weight.mean().item()}")
            running_loss += loss.item()

            # Track learning dynamics
            probs = torch.sigmoid(out.detach())
            train_preds.append(probs.cpu())
            train_labels.append(batch["label"])
            if batch_idx == 0:
                logging.info(
                    f"  pred range: [{probs.min().item():.4f}, {probs.max().item():.4f}]"
                )
                if model.cls[1].weight.grad is not None:
                    logging.info(
                        f"  cls grad norm: {model.cls[1].weight.grad.norm().item():.4f}"
                    )
        except RuntimeError as e:
            logging.error(f"CUDA error in batch {batch_idx}: {e}")
            logging.error(f"Skipping batch due to error...")
            continue
    sched.step()

    # Training metrics
    if train_preds and train_labels:
        train_auprc = average_precision_score(
            torch.cat(train_labels).numpy(), torch.cat(train_preds).numpy()
        )
        logging.info(
            f"epoch {epoch}: train loss {running_loss/len(train_preds):.4f}, AUPRC = {train_auprc:.3f}"
        )

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
        auprc = average_precision_score(labels_t.numpy(), preds_t.numpy())
        logging.info(f"epoch {epoch}: LOCO-chr21 AUPRC = {auprc:.3f}")
        logging.info(
            f"  val pred range: [{preds_t.min().item():.4f}, {preds_t.max().item():.4f}]"
        )
