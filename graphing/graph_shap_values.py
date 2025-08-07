import json
import os
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import shap

from settings import BAR_COLOURS, EDGE_COLOUR

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def plot_shap_values(
    model,
    X: pd.DataFrame,
    model_name: str,
    params: Dict,
    timestamp: str,
    folder: str = "graphs",
    top_n: int = 20,
    errors: pd.Series | None = None,
):
    """Compute and plot SHAP values with Oxford colours.

    Parameters
    ----------
    errors
        Optional error values to display as horizontal error bars.
    """
    out_dir = os.path.join(folder, timestamp)
    os.makedirs(out_dir, exist_ok=True)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_abs = pd.Series(abs(shap_values).mean(axis=0), index=X.columns)
    shap_abs = shap_abs.sort_values(ascending=False).head(top_n)
    if errors is not None:
        errors = errors.reindex(shap_abs.index).fillna(0)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(
        shap_abs.index[::-1],
        shap_abs.values[::-1],
        xerr=None if errors is None else errors.values[::-1],
        color=BAR_COLOURS[0],
        edgecolor="none",
        ecolor=EDGE_COLOUR,
        capsize=3,
    )
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_ylabel("Feature")
    ax.set_title(f"{model_name} SHAP Importance")
    plt.tight_layout()

    fig_path = os.path.join(out_dir, "shap_importance.png")
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)

    pd.DataFrame(shap_values, columns=X.columns).to_csv(
        os.path.join(out_dir, "shap_values.csv"), index=False
    )
    shap_df = shap_abs.to_frame("importance")
    if errors is not None:
        shap_df["variance"] = errors.reindex(shap_abs.index)
    shap_df.to_csv(os.path.join(out_dir, "shap_importance.csv"))

    with open(os.path.join(out_dir, "params.json"), "w") as f:
        json.dump({"model": model_name, "params": params}, f, indent=2)

    return fig_path
