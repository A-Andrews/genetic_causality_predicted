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
):
    """Plot feature importances using Oxford colours and save artefacts."""
    out_dir = os.path.join(folder, timestamp)
    os.makedirs(out_dir, exist_ok=True)

    feature_importance = feature_importance.sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(8, 6))
    feature_importance[::-1].plot(
        kind="barh", color=BAR_COLOURS[0], edgecolor="none", ax=ax
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
