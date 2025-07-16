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
        yerr = pd.Series(errors).reindex(metrics_series.index).fillna(0).values
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

    metrics_df = metrics_series.to_frame("metric")
    if errors is not None:
        metrics_df["variance"] = pd.Series(errors).reindex(metrics_series.index)
    metrics_df.to_csv(os.path.join(out_dir, "performance_metrics.csv"))

    with open(os.path.join(out_dir, "params.json"), "w") as f:
        json.dump({"model": model_name, "params": params}, f, indent=2)

    return fig_path


def plot_chromosome_performance(
    chrom_metrics: pd.Series | pd.DataFrame,
    model_name: str,
    params: Dict,
    timestamp: str,
    *,
    folder: str = "graphs",
    metric: str = "auprc",
    errors: pd.Series | None = None,
) -> str:
    """Plot performance for each held-out chromosome.

    Parameters
    ----------
    chrom_metrics
        Either a :class:`~pandas.Series` of metric values indexed by chromosome
        or a :class:`~pandas.DataFrame` where ``metric`` selects the desired
        column.
    metric
        Metric name to display when ``chrom_metrics`` is a dataframe.
    errors
        Optional standard error values for each chromosome.
    """
    out_dir = os.path.join(folder, timestamp)
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
