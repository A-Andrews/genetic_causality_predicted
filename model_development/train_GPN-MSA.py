import gpn.model
import torch
import torch.nn as nn
from datasets import load_dataset
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader
from gpn.model import GPNRoFormerModel

from fixed_load_gpn import TGMSAFixed as TGMSA

# Load one of the four TraitGym configurations.
# Pick the "_full" variant if you want the 9 × matched negatives.
tg = load_dataset(
    "songlab/TraitGym",
    "complex_traits_full",  # or "complex_traits", "mendelian_traits(_full)"
    split="test",  # TraitGym uses "test" for the full label set
)

train = TGMSA(tg.filter(lambda r: r["chrom"] != "21"))  # leave-chr-21-out
val = TGMSA(tg.filter(lambda r: r["chrom"] == "21"))

train_loader = DataLoader(train, batch_size=32, shuffle=True, drop_last=True)
val_loader = DataLoader(val, batch_size=32)

# Debug: Test data loading
("Testing data loader...")
test_batch = next(iter(train_loader))
print(f"Sample batch:")
print(f"  ref shape: {test_batch['ref'].shape}, dtype: {test_batch['ref'].dtype}")
print(f"  alt shape: {test_batch['alt'].shape}, dtype: {test_batch['alt'].dtype}")
print(f"  attn shape: {test_batch['attn'].shape}, dtype: {test_batch['attn'].dtype}")
print(f"  label shape: {test_batch['label'].shape}, dtype: {test_batch['label'].dtype}")
print(f"  ref sample values: {test_batch['ref'][0][:10]}")
print(f"  alt sample values: {test_batch['alt'][0][:10]}")
print(f"  ref range: [{test_batch['ref'].min()}, {test_batch['ref'].max()}]")
print(f"  alt range: [{test_batch['alt'].min()}, {test_batch['alt'].max()}]")

enc = GPNRoFormerModel.from_pretrained("songlab/gpn-msa-sapiens")  # 86 M params

# Debug: Print model configuration
print(f"Model config:")
print(f"  vocab_size: {enc.config.vocab_size}")
print(f"  hidden_size: {enc.config.hidden_size}")
print(f"  max_position_embeddings: {getattr(enc.config, 'max_position_embeddings', 'N/A')}")

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
        h_ref = self.encoder(ref, attention_mask=attn).last_hidden_state[:, 64, :]
        h_alt = self.encoder(alt, attention_mask=attn).last_hidden_state[:, 64, :]
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
lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]).cuda())

for epoch in range(8):
    model.train()
    # 1) warm-up epochs 0-2 with encoder frozen
    if epoch == 3:
        for p in enc.encoder.layer[-4:].parameters():  # unfreeze top 4 layers
            p.requires_grad = True
    for batch_idx, batch in enumerate(train_loader):
        opt.zero_grad()
        
        # Debug: Check input tensor properties
        print(f"Batch {batch_idx}:")
        print(f"  ref shape: {batch['ref'].shape}, range: [{batch['ref'].min()}, {batch['ref'].max()}]")
        print(f"  alt shape: {batch['alt'].shape}, range: [{batch['alt'].min()}, {batch['alt'].max()}]")
        print(f"  attn shape: {batch['attn'].shape}, dtype: {batch['attn'].dtype}")
        
        # Check for invalid token indices
        if batch['ref'].max() >= enc.config.vocab_size or batch['alt'].max() >= enc.config.vocab_size:
            print(f"WARNING: Token indices exceed vocab_size ({enc.config.vocab_size})")
            print(f"  ref max: {batch['ref'].max()}, alt max: {batch['alt'].max()}")
        
        try:
            out = model(batch["ref"].cuda(), batch["alt"].cuda(), batch["attn"].cuda())
            loss = lossf(out, batch["label"].cuda())
            loss.backward()
            opt.step()
        except RuntimeError as e:
            print(f"CUDA error in batch {batch_idx}: {e}")
            print(f"Skipping batch due to error...")
            continue
    sched.step()

    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for b in val_loader:
            p = torch.sigmoid(model(b["ref"].cuda(), b["alt"].cuda(), b["attn"].cuda()))
            preds.append(p.cpu())
            labels.append(b["label"])
    auprc = average_precision_score(torch.cat(labels), torch.cat(preds))
    print(f"epoch {epoch}: LOCO-chr21 AUPRC = {auprc:.3f}")
