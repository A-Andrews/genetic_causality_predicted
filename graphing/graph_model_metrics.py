import json
import os
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
    errors: Dict[str, float] | None = None,
):
    """Plot bar chart of model performance metrics and save artefacts.

    Parameters
    ----------
    metrics
        Mapping of metric name to score (typically mean across folds).
    errors
        Optional mapping of metric name to error (e.g. standard deviation).
        When provided, error bars are drawn using ``matplotlib``'s ``yerr``.
    """
    out_dir = os.path.join(folder, timestamp)
    os.makedirs(out_dir, exist_ok=True)

    metrics_series = pd.Series(metrics)

    fig, ax = plt.subplots(figsize=(6, 4))
    yerr = None
    if errors is not None:
        yerr = pd.Series(errors).reindex(metrics_series.index).values
    ax.bar(
        metrics_series.index,
        metrics_series.values,
        yerr=yerr,
        color=BAR_COLOURS[0],
        edgecolor="none",
        ecolor=EDGE_COLOUR,
        capsize=3,
    )
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
