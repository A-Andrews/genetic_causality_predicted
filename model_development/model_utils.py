import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Tuple

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

from graphing.graph_importances import plot_feature_importance
from graphing.graph_model_metrics import plot_model_metrics
from graphing.graph_shap_values import plot_shap_values


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
    random_state: int = 42,
    sample_size: int | None = 5000,
) -> pd.Series:
    """Return permutation importance for a fitted model.

    To keep runtime manageable on very large datasets, a random subset of
    ``sample_size`` rows is used when ``sample_size`` is not ``None``.
    """
    if len(np.unique(y)) < 2:
        logging.warning(
            "Only one class present in y; using f1 for permutation importance"
        )
        scoring = "f1"

    if sample_size is not None and len(X) > sample_size:
        X, y = resample(
            X,
            y,
            n_samples=sample_size,
            replace=False,
            random_state=random_state,
        )

    result = permutation_importance(
        model,
        X,
        y,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring=scoring,
        n_jobs=-1,
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


def chromosome_holdout_cv(
    data: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    build_model: Callable[..., Any],
) -> List[float]:
    """Run chromosome hold-out cross-validation."""
    chromosomes = sorted(data["chrom"].unique())
    cv_scores = []
    for chrom in chromosomes:
        train_mask = data["chrom"] != chrom
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[~train_mask], y[~train_mask]

        model = build_model(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
        )

        # XGBoost can handle DataFrames directly while TabNet requires ndarrays
        x_val_pred = X_val if hasattr(model, "get_booster") else X_val.values
        y_pred = model.predict_proba(x_val_pred)[:, 1]
        metrics = evaluate(y_val, y_pred)
        cv_scores.append(metrics["auprc"])
        logging.info(
            "Chromosome %s - AUPRC: %.3f | ROC-AUC: %.3f | Positives: %d",
            chrom,
            metrics["auprc"],
            metrics["roc_auc"],
            (y_val == True).sum(),
        )

    logging.info(
        "Mean chromosome CV AUPRC: %.3f +/- %.3f",
        np.mean(cv_scores),
        np.std(cv_scores),
    )
    return cv_scores


def train_final_model(
    X: pd.DataFrame,
    y: pd.Series,
    build_model: Callable[..., Any],
    model_name: str,
    args: dataclass,
) -> Any:
    """Train final model and save artefacts."""
    model = build_model(X, y)
    params = asdict(args)

    feature_imp = compute_feature_importance(model, X.columns)
    logging.info(
        "Top 10 features by model importance:\n%s", feature_imp.head(10).to_string()
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    fi_path = plot_feature_importance(feature_imp, model_name, params, timestamp)
    shap_path = plot_shap_values(model, X, model_name, params, timestamp)

    x_pred = X if hasattr(model, "get_booster") else X.values
    y_pred = model.predict_proba(x_pred)[:, 1]
    metrics = evaluate(y, y_pred)
    metrics_path = plot_model_metrics(metrics, model_name, params, timestamp)

    for path in [fi_path, shap_path, metrics_path]:
        save_args(args, os.path.dirname(path))

    return model
