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


def plot_feature_importance(
    feature_importance: pd.Series,
    model_name: str,
    params: Dict,
    timestamp: str,
    folder: str = "graphs",
    top_n: int = 20,
    errors: pd.Series | None = None,
):
    """Plot feature importances using Oxford colours and save artefacts.

    Parameters
    ----------
    feature_importance
        Importance scores indexed by feature name.
    errors
        Optional error values corresponding to ``feature_importance``. When
        provided, horizontal error bars are shown.
    """
    out_dir = os.path.join(folder, timestamp)
    os.makedirs(out_dir, exist_ok=True)

    feature_importance = feature_importance.sort_values(ascending=False).head(top_n)
    if errors is not None:
        errors = errors.reindex(feature_importance.index).fillna(0)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(
        feature_importance.index[::-1],
        feature_importance.values[::-1],
        xerr=None if errors is None else errors.values[::-1],
        color=BAR_COLOURS[0],
        edgecolor="none",
        ecolor=EDGE_COLOUR,
        capsize=3,
    )
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.set_title(f"{model_name} Feature Importance")
    plt.tight_layout()

    fig_path = os.path.join(out_dir, "feature_importance.png")
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)

    feature_importance.to_csv(os.path.join(out_dir, "feature_importance.csv"))

    with open(os.path.join(out_dir, "params.json"), "w") as f:
        json.dump({"model": model_name, "params": params}, f, indent=2)

    return fig_path


def plot_permutation_importance(
    perm_importance: pd.Series,
    model_name: str,
    params: Dict,
    timestamp: str,
    folder: str = "graphs",
    top_n: int = 20,
    errors: pd.Series | None = None,
):
    """Plot permutation importances using Oxford colours and save artefacts."""
    out_dir = os.path.join(folder, timestamp)
    os.makedirs(out_dir, exist_ok=True)

    perm_importance = perm_importance.sort_values(ascending=False).head(top_n)
    if errors is not None:
        errors = errors.reindex(perm_importance.index).fillna(0)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(
        perm_importance.index[::-1],
        perm_importance.values[::-1],
        xerr=None if errors is None else errors.values[::-1],
        color=BAR_COLOURS[0],
        edgecolor="none",
        ecolor=EDGE_COLOUR,
        capsize=3,
    )
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.set_title(f"{model_name} Permutation Importance")
    plt.tight_layout()

    fig_path = os.path.join(out_dir, "permutation_importance.png")
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)

    perm_importance.to_csv(os.path.join(out_dir, "permutation_importance.csv"))

    with open(os.path.join(out_dir, "params.json"), "w") as f:
        json.dump({"model": model_name, "params": params}, f, indent=2)

    return fig_path
