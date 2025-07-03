import json
import logging
import os
from dataclasses import asdict
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.utils import resample


def evaluate(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute common binary classification metrics."""
    metrics = {}
    if len(np.unique(y_true)) < 2:
        logging.warning("Only one class present in y_true; metrics may be meaningless")
        ap = 1.0
        roc = 1.0
        acc = 1.0
        f1 = 1.0
    else:
        ap = average_precision_score(y_true, y_pred)
        roc = roc_auc_score(y_true, y_pred)
        preds_binary = (y_pred >= 0.5).astype(int)
        acc = accuracy_score(y_true, preds_binary)
        f1 = f1_score(y_true, preds_binary)

    metrics["auprc"] = ap
    metrics["roc_auc"] = roc
    metrics["accuracy"] = acc
    metrics["f1"] = f1
    return metrics


def oversample_minority(
    X: pd.DataFrame, y: pd.Series
) -> Tuple[pd.DataFrame, pd.Series]:
    """Randomly oversample the minority class."""
    df = X.copy()
    df["label"] = y
    counts = df["label"].value_counts()
    if len(counts) < 2:
        return X, y
    minority = counts.idxmin()
    majority = counts.idxmax()
    if counts[minority] == counts[majority]:
        return X, y

    minority_df = df[df["label"] == minority]
    majority_df = df[df["label"] == majority]
    minority_upsampled = resample(
        minority_df,
        replace=True,
        n_samples=len(majority_df),
        random_state=42,
    )
    df_upsampled = pd.concat([majority_df, minority_upsampled])
    return df_upsampled.drop(columns=["label"]), df_upsampled["label"]


def compute_feature_importance(model, feature_names: pd.Index) -> pd.Series:
    """Return feature importance from a fitted model."""
    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=feature_names)
        return imp.sort_values(ascending=False)
    logging.warning("Model does not expose feature_importances_.")
    return pd.Series(dtype=float)


def compute_permutation_importance(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    scoring: str = "roc_auc",
    n_repeats: int = 10,
    random_state: int = 42
) -> pd.Series:
    """Return permutation importance for a fitted model."""
    result = permutation_importance(
        model,
        X,
        y,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring=scoring,
    )
    imp = pd.Series(result.importances_mean, index=X.columns)
    return imp.sort_values(ascending=False)

leaky_cols = [
    "chrom",
    "pos",
    "ref",
    "alt",
    "SNP",
    "trait",
    "CHR",
    "BP",
    "CM",
    "genetic_dist",
    "variant_id",
    "label",
    "pip",
]


def prepare_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare dataframe for modelling by dropping leak-prone columns."""

    logging.info("Data loaded successfully.")
    logging.info(f"Data shape: {data.shape}")
    logging.info(f"Columns: {data.columns.tolist()}")
    logging.info(f"Column types:\n{data.dtypes}")
    logging.info(
        f"Object columns:\n{data.select_dtypes(include='object').columns.tolist()}"
    )

    X = data.drop(columns=["label"])
    y = data["label"]

    X = X.drop(columns=[col for col in leaky_cols if col in X.columns])

    return X, y


def save_args(args, directory: str) -> None:
    """Persist CLI arguments for reproducibility."""

    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "cli_args.json"), "w") as f:
        json.dump(asdict(args), f, indent=2)
