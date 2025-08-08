import json
import os
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd

from utils.settings import BAR_COLOURS, EDGE_COLOUR


def plot_chromosome_performance(
    chrom_metrics: pd.Series | pd.DataFrame,
    model_name: str,
    params: Dict,
    timestamp: str,
    folder: str = "graphs",
    metric: str = "auprc",
    errors: pd.Series | None = None,
) -> str:

    out_dir = os.path.join(folder, "per_chromosome_performances", model_name, timestamp)
    os.makedirs(out_dir, exist_ok=True)

    if isinstance(chrom_metrics, pd.DataFrame):
        values = chrom_metrics[metric]
    else:
        values = chrom_metrics

    fig, ax = plt.subplots(figsize=(8, 4))
    yerr = None
    if errors is not None:
        yerr = errors.reindex(values.index).fillna(0).values
    ax.bar(
        values.index.astype(str),
        values.values,
        yerr=yerr,
        color=BAR_COLOURS[0],
        edgecolor="none",
        ecolor=EDGE_COLOUR,
        capsize=3,
    )
    ax.set_xlabel("Chromosome")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{model_name} {metric.upper()} by Chromosome")
    plt.tight_layout()

    fig_path = os.path.join(out_dir, f"chromosome_{metric}.png")
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)

    values_df = values.to_frame(metric)
    if errors is not None:
        values_df["variance"] = errors.reindex(values.index)
    values_df.to_csv(os.path.join(out_dir, f"chromosome_{metric}.csv"))

    with open(os.path.join(out_dir, "params.json"), "w") as f:
        json.dump({"model": model_name, "params": params}, f, indent=2)

    return fig_path
