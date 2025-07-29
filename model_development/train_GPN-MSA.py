import gpn.model
import torch.nn as nn
from datasets import load_dataset
from sklearn.metrics import average_precision_score
from transformers import AutoModel

from data_consolidation.load_gpn import TGMSA

# Load one of the four TraitGym configurations.
# Pick the "_full" variant if you want the 9 × matched negatives.
tg = load_dataset(
    "songlab/TraitGym",
    "complex_traits_full",  # or "complex_traits", "mendelian_traits(_full)"
    split="test",  # TraitGym uses "test" for the full label set
)

train = TGMSA(tg.filter(lambda r: r["chrom"] != "21"))  # leave-chr-21-out
val = TGMSA(tg.filter(lambda r: r["chrom"] == "21"))

enc = AutoModel.from_pretrained(
    "songlab/gpn-msa-sapiens", add_pooling_layer=False
)  # 86 M params
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
    for batch in train_loader:
        opt.zero_grad()
        out = model(batch["ref"].cuda(), batch["alt"].cuda(), batch["attn"].cuda())
        loss = lossf(out, batch["label"].cuda())
        loss.backward()
        opt.step()
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
