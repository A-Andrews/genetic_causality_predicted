import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.utils import resample

from graphing.graph_importances import (
    plot_feature_importance,
    plot_permutation_importance,
)
from graphing.graph_model_metrics import plot_chromosome_performance, plot_model_metrics
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
) -> Tuple[pd.Series, pd.Series]:
    """Return permutation importance for a fitted model.

    To keep runtime manageable on very large datasets, a random subset of
    ``sample_size`` rows is used when ``sample_size`` is not ``None``.
    Returns
    -------
    Tuple[pd.Series, pd.Series]
        Mean and standard deviation of the permutation importance for each
        feature.
    """
    logging.info(
        "Permutation importance on data shape %s with label distribution %s",
        X.shape,
        y.value_counts().to_dict(),
    )
    if len(np.unique(y)) < 2:
        logging.warning(
            "Only one class present in y; skipping permutation importance. Distribution: %s",
            y.value_counts().to_dict(),
        )
        empty = pd.Series(dtype=float)
        return empty, empty

    if sample_size is not None and len(X) > sample_size:
        X, y = resample(
            X,
            y,
            n_samples=sample_size,
            replace=False,
            random_state=random_state,
        )
        logging.info(
            "Sampling %d rows for permutation importance. New distribution %s",
            len(X),
            y.value_counts().to_dict(),
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
    imp_mean = pd.Series(result.importances_mean, index=X.columns)
    imp_std = pd.Series(result.importances_std, index=X.columns)
    imp_mean = imp_mean.sort_values(ascending=False)
    imp_std = imp_std.reindex(imp_mean.index)
    return imp_mean, imp_std


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
    *,
    n_runs: int = 1,
    random_state: int | None = None,
    collect_importance: bool = False,
    return_chrom_metrics: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
    """Run chromosome hold-out cross-validation."""
    rng = np.random.default_rng(random_state)
    chromosomes = sorted(data["chrom"].unique())
    run_metrics = []
    fi_runs = [] if collect_importance else None
    chrom_runs = [] if return_chrom_metrics else None
    for run in range(n_runs):
        logging.info("Starting CV run %d/%d", run + 1, n_runs)
        run_seed = None if random_state is None else int(rng.integers(0, 1_000_000))
        fold_metrics = []
        fold_weights = []
        fi_list = [] if collect_importance else None
        for chrom in chromosomes:
            train_mask = data["chrom"] != chrom
            X_train, y_train = X[train_mask], y[train_mask]
            X_val, y_val = X[~train_mask], y[~train_mask]

            model = build_model(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                random_state=run_seed,
            )

            # XGBoost can handle DataFrames directly while TabNet requires ndarrays
            x_val_pred = X_val if hasattr(model, "get_booster") else X_val.values
            y_pred = model.predict_proba(x_val_pred)[:, 1]
            metrics = evaluate(y_val, y_pred)
            fold_metrics.append(metrics)
            fold_weights.append(len(y_val))
            if collect_importance:
                fi = compute_feature_importance(model, X.columns)
                fi_list.append(fi)
            logging.info(
                "Run %d, Chromosome %s - AUPRC: %.3f | ROC-AUC: %.3f | Positives: %d",
                run + 1,
                chrom,
                metrics["auprc"],
                metrics["roc_auc"],
                (y_val == True).sum(),
            )

        fold_df = pd.DataFrame(fold_metrics, index=chromosomes)
        fold_sizes = pd.Series(fold_weights, index=chromosomes)
        weighted_mean = fold_df.mul(fold_sizes, axis=0).sum() / fold_sizes.sum()
        run_metrics.append(weighted_mean)
        if collect_importance:
            fi_runs.append(pd.concat(fi_list, axis=1).mean(axis=1))
        if return_chrom_metrics:
            chrom_runs.append(fold_df)

    metrics_df = pd.DataFrame(run_metrics)
    logging.info(
        "Mean CV AUPRC over %d runs: %.3f +/- %.3f",
        n_runs,
        metrics_df["auprc"].mean(),
        metrics_df["auprc"].std() / np.sqrt(n_runs),
    )
    fi_df = pd.concat(fi_runs, axis=1) if collect_importance else None
    chrom_mean = None
    chrom_err = None
    if return_chrom_metrics:
        chrom_concat = pd.concat(chrom_runs, keys=range(n_runs))
        chrom_mean = chrom_concat.groupby(level=1).mean()
        chrom_err = chrom_concat.groupby(level=1).std().div(np.sqrt(n_runs))
    return metrics_df, fi_df, chrom_mean, chrom_err


def train_final_model(
    X: pd.DataFrame,
    y: pd.Series,
    build_model: Callable[..., Any],
    model_name: str,
    args: dataclass,
    *,
    metric_errors: Dict[str, float] | None = None,
    fi_errors: pd.Series | None = None,
    timestamp: str | None = None,
) -> Any:
    """Train final model and save artefacts."""
    model = build_model(X, y)
    params = asdict(args)

    feature_imp = compute_feature_importance(model, X.columns)
    logging.info(
        "Top 10 features by model importance:\n%s", feature_imp.head(10).to_string()
    )

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    fi_path = plot_feature_importance(
        feature_imp, model_name, params, timestamp, errors=fi_errors
    )
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_err = pd.Series(abs(shap_values).std(axis=0), index=X.columns)
    shap_path = plot_shap_values(
        model,
        X,
        model_name,
        params,
        timestamp,
        errors=shap_err,
    )

    perm_imp, perm_err = compute_permutation_importance(model, X, y)

    logging.info(
        "Top 10 features by permutation importance:\n%s",
        perm_imp.head(10).to_string(),
    )
    pi_path = plot_permutation_importance(
        perm_imp, model_name, params, timestamp, errors=perm_err
    )

    x_pred = X if hasattr(model, "get_booster") else X.values
    y_pred = model.predict_proba(x_pred)[:, 1]
    metrics = evaluate(y, y_pred)
    metrics_path = plot_model_metrics(
        metrics, model_name, params, timestamp, errors=metric_errors
    )

    for path in [fi_path, shap_path, pi_path, metrics_path]:
        save_args(args, os.path.dirname(path))

    return model
