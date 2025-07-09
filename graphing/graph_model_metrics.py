import os
import json
from typing import Dict
import matplotlib.pyplot as plt
import pandas as pd

from graphing.graph_utils import BAR_COLOURS, EDGE_COLOUR

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def plot_model_metrics(
    metrics: Dict[str, float],
    model_name: str,
    params: Dict,
    timestamp: str,
    folder: str = "graphs",
):
    """Plot bar chart of model performance metrics and save artefacts."""
    out_dir = os.path.join(folder, timestamp)
    os.makedirs(out_dir, exist_ok=True)

    metrics_series = pd.Series(metrics)

    fig, ax = plt.subplots(figsize=(6, 4))
    metrics_series.plot(kind="bar", color=BAR_COLOURS[0], edgecolor="none", ax=ax)
    ax.set_ylabel("Score")
    ax.set_xlabel("Metric")
    ax.set_title(f"{model_name} Performance")
    plt.tight_layout()

    fig_path = os.path.join(out_dir, "performance_metrics.png")
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)

    metrics_series.to_csv(os.path.join(out_dir, "performance_metrics.csv"))

    with open(os.path.join(out_dir, "params.json"), "w") as f:
        json.dump({"model": model_name, "params": params}, f, indent=2)

    return fig_path
