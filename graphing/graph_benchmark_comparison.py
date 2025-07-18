import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from settings import TRAITGYM_PATH


timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

ALLOWED_DATASETS = {"complex_traits_matched_9", "mendelian_traits_matched_9"}
METRIC_FOLDERS = ("AUPRC_by_chrom_weighted_average", "AUPRC")
OUTDIR = Path("graphs") / pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
OUTDIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------
# Small helper: robustly read the score (3rd column) from a 1-row CSV
# ------------------------------------------------------------------
def read_score(csv_path: Path) -> float:
    """Return the 3rd column as float, skipping any non-numeric rows."""
    with csv_path.open(newline="") as fh:
        for row in csv.reader(fh):
            if len(row) < 3:
                continue  # blank or malformed line
            try:
                return float(row[2])  # third field is the score
            except ValueError:
                continue  # row is header → keep looking
    raise RuntimeError(f"Could not parse score from {csv_path}")


# ------------------------------------------------------------------
# Collect all rows that belong to the two matched-9 datasets
# ------------------------------------------------------------------
records = []
for metric in METRIC_FOLDERS:
    for csv_file in Path(TRAITGYM_PATH).rglob(f"{metric}/all/*.csv"):
        dataset = csv_file.parts[-4]  # …/<dataset>/<metric>/all/<file>
        if dataset not in ALLOWED_DATASETS:
            continue

        records.append(
            {
                "dataset": dataset,
                "metric": metric,
                "model": csv_file.stem,
                "score": read_score(csv_file),  # ← uses robust parser
            }
        )

if not records:
    raise SystemExit(
        "No CSVs from the matched-9 datasets were found – "
        "check TRAITGYM_PATH and that the Hugging-Face download ran."
    )

# ------------------------------------------------------------------
# Plot – horizontal bars, numeric x-axis, values at bar ends
# ------------------------------------------------------------------
df = pd.DataFrame(records)
for (dataset, metric), sub in df.groupby(["dataset", "metric"]):
    sub = sub.sort_values("score", ascending=False)

    plt.figure(figsize=(10, 0.25 * len(sub) + 2))
    bars = plt.barh(sub["model"], sub["score"])
    plt.gca().invert_yaxis()
    plt.xlabel(metric.replace("_", " "))
    plt.title(f"TraitGym – {dataset}  ({metric})")

    # write the exact score next to each bar
    for bar, score in zip(bars, sub["score"]):
        plt.text(
            bar.get_width() + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.3f}",
            va="center",
            ha="left",
            fontsize=8,
        )

    plt.tight_layout()
    out = OUTDIR / f"{dataset}_{metric}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")

print(f"All figures are in  {OUTDIR}/")
