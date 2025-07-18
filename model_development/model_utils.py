import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Callable, Dict, Tuple

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
    X: pd.DataFrame,
    y: pd.Series,
    *,
    random_state: int | None = None,
    n_samples: int = 1,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Randomly oversample the minority class using bootstrapping

    If ``n_samples`` is greater than one, multiple independent bootstrapped
    datasets are drawn and concatenated to produce a larger, balanced dataset.

    Parameters
    ----------
    random_state:
        Seed for reproducible sampling.
    n_samples:
        Number of independent bootstrap samples to draw.
    """
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
        random_state=random_state,
    )
    rng = np.random.default_rng(random_state)
    minority_samples = []
    for _ in range(n_samples):
        seed = None if random_state is None else int(rng.integers(0, 1_000_000))
        minority_sample = resample(
            minority_df,
            replace=True,
            n_samples=len(majority_df),
            random_state=seed,
        )
        sample_df = pd.concat([majority_df, minority_sample])
        minority_samples.append(minority_sample)

    all_minority = pd.concat(minority_samples, ignore_index=True)
    df_upsampled = pd.concat([majority_df, all_minority], ignore_index=True)
    return df_upsampled.drop(columns=["label"]), df_upsampled["label"]


def downsample_majority(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    fraction: float,
    random_state: int | None = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Randomly downsample the majority (negative) class.

    Parameters
    ----------
    fraction:
        Fraction of majority class samples to keep. Values ``>=1`` disable
        downsampling.
    random_state:
        Seed for reproducible sampling.
    """
    if fraction >= 1:
        return X, y

    df = X.copy()
    df["label"] = y
    counts = df["label"].value_counts()
    if len(counts) < 2:
        return X, y
    majority = counts.idxmax()
    minority = counts.idxmin()

    majority_df = df[df["label"] == majority]
    minority_df = df[df["label"] == minority]
    n_samples = int(len(majority_df) * fraction)
    majority_down = resample(
        majority_df,
        replace=False,
        n_samples=n_samples,
        random_state=random_state,
    )
    df_down = pd.concat([majority_down, minority_df], ignore_index=True)
    return df_down.drop(columns=["label"]), df_down["label"]


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
        Mean and standard error of the permutation importance for each
        feature. The standard error is computed from the standard deviation
        across ``n_repeats`` permutations.
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
    imp_se = pd.Series(result.importances_std / np.sqrt(n_repeats), index=X.columns)
    imp_mean = imp_mean.sort_values(ascending=False)
    imp_se = imp_se.reindex(imp_mean.index)
    return imp_mean, imp_se


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
    compute_shap: bool = False,
    compute_permutation: bool = False,
    return_chrom_metrics: bool = False,
    bootstrap_samples: int = 1,
    bootstrap: bool = False,
    neg_frac: float = 1.0,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame | None,
    pd.DataFrame | None,
    pd.DataFrame | None,
    pd.DataFrame | None,
    pd.DataFrame | None,
    pd.DataFrame | None,
]:
    """
    Run chromosome hold-out cross-validation.

    Parameters
    ----------
    n_runs:
        Number of repeated CV runs. Each run uses a different random seed so
        bootstrapped sampling produces independent training sets when
        ``bootstrap`` is ``True``.
    random_state:
        Base random seed for reproducible runs.
    bootstrap:
        If ``True``, oversample the minority class in each training fold using
        bootstrapping.
    bootstrap_samples:
        Number of independent bootstrap samples to draw when ``bootstrap`` is
        ``True``. Each sample is concatenated with a copy of the majority class
        to create a larger training set.
    neg_frac:
        Fraction of negative examples to keep in each training fold. Values
        greater than or equal to ``1`` disable downsampling.
    compute_shap:
        If ``True``, compute SHAP values for the validation fold of each
        chromosome and aggregate them across runs.
    compute_permutation:
        If ``True``, compute permutation importance for each validation fold
        and aggregate the results across runs.
    """
    rng = np.random.default_rng(random_state)
    chromosomes = sorted(data["chrom"].unique())
    run_metrics = []
    fi_runs = [] if collect_importance else None
    shap_runs = [] if compute_shap else None
    perm_runs_mean = [] if compute_permutation else None
    perm_runs_se = [] if compute_permutation else None
    chrom_runs = [] if return_chrom_metrics else None
    for run in range(n_runs):
        logging.info("Starting CV run %d/%d", run + 1, n_runs)
        run_seed = None if random_state is None else int(rng.integers(0, 1_000_000))
        fold_metrics = []
        fold_weights = []
        fi_list = [] if collect_importance else None
        shap_list = [] if compute_shap else None
        perm_list_mean = [] if compute_permutation else None
        perm_list_se = [] if compute_permutation else None
        for chrom in chromosomes:
            train_mask = data["chrom"] != chrom
            X_train, y_train = X[train_mask], y[train_mask]
            if neg_frac < 1.0:
                fold_seed = (
                    None if random_state is None else int(rng.integers(0, 1_000_000))
                )
                X_train, y_train = downsample_majority(
                    X_train,
                    y_train,
                    fraction=neg_frac,
                    random_state=fold_seed,
                )
            if bootstrap:
                fold_seed = (
                    None if random_state is None else int(rng.integers(0, 1_000_000))
                )
                X_train, y_train = oversample_minority(
                    X_train,
                    y_train,
                    random_state=fold_seed,
                    n_samples=bootstrap_samples,
                )
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
            if compute_shap:
                x_val_shap = X_val if hasattr(model, "get_booster") else X_val.values
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(x_val_shap)
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]
                shap_mean = pd.Series(np.abs(shap_values).mean(axis=0), index=X.columns)
                shap_list.append(shap_mean)
            if compute_permutation:
                X_imp, y_imp = (X_val, y_val)
                if len(np.unique(y_val)) < 2:
                    logging.warning(
                        "Chromosome %s validation has a single class; using training data for permutation importance",
                        chrom,
                    )
                    X_imp, y_imp = X_train, y_train
                pi_mean, pi_se = compute_permutation_importance(
                    model,
                    X_imp,
                    y_imp,
                )
                perm_list_mean.append(pi_mean)
                perm_list_se.append(pi_se)
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
            fi_df = pd.concat(fi_list, axis=1)
            fi_df.columns = chromosomes
            fi_weighted = fi_df.mul(fold_sizes, axis=1).sum(axis=1) / fold_sizes.sum()
            fi_runs.append(fi_weighted)
        if compute_permutation:
            pi_mean_df = pd.concat(perm_list_mean, axis=1)
            pi_mean_df.columns = chromosomes
            pi_mean_weighted = (
                pi_mean_df.mul(fold_sizes, axis=1).sum(axis=1) / fold_sizes.sum()
            )
            perm_runs_mean.append(pi_mean_weighted)
            pi_se_df = pd.concat(perm_list_se, axis=1)
            pi_se_df.columns = chromosomes
            pi_se_weighted = (
                pi_se_df.mul(fold_sizes, axis=1).sum(axis=1) / fold_sizes.sum()
            )
            perm_runs_se.append(pi_se_weighted)
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
    shap_df = pd.concat(shap_runs, axis=1) if compute_shap else None
    perm_mean_df = pd.concat(perm_runs_mean, axis=1) if compute_permutation else None
    perm_se_df = pd.concat(perm_runs_se, axis=1) if compute_permutation else None
    return metrics_df, fi_df, chrom_mean, chrom_err, shap_df, perm_mean_df, perm_se_df


def plot_cv_results(
    metrics_df: pd.DataFrame,
    fi_df: pd.DataFrame | None,
    chrom_mean: pd.DataFrame | None,
    chrom_err: pd.DataFrame | None,
    perm_df: pd.DataFrame | None,
    perm_err_df: pd.DataFrame | None,
    args,
    model_name: str,
    timestamp: str,
    *,
    folder: str = "graphs/cv",
) -> str:
    """Plot cross-validation results and persist CLI arguments."""

    if len(metrics_df) > 1:
        metric_errors = metrics_df.std().div(np.sqrt(len(metrics_df))).to_dict()
    elif chrom_mean is not None:
        metric_errors = chrom_mean.std().div(np.sqrt(len(chrom_mean))).to_dict()
    else:
        metric_errors = {m: np.nan for m in metrics_df.columns}

    fi_errors = None
    if fi_df is not None and fi_df.shape[1] > 1:
        fi_errors = fi_df.std(axis=1).div(np.sqrt(fi_df.shape[1]))
    perm_errors = perm_err_df.mean(axis=1) if perm_err_df is not None else None

    cv_path = plot_model_metrics(
        metrics_df.mean().to_dict(),
        f"{model_name} CV",
        asdict(args),
        timestamp,
        folder=folder,
        errors=metric_errors,
    )
    if fi_df is not None:
        plot_feature_importance(
            fi_df.mean(axis=1),
            model_name,
            asdict(args),
            timestamp,
            folder=folder,
            errors=fi_errors,
        )
    if perm_df is not None:
        plot_permutation_importance(
            perm_df.mean(axis=1),
            model_name,
            asdict(args),
            timestamp,
            folder=folder,
            errors=perm_errors,
        )

    save_args(args, os.path.dirname(cv_path))

    if chrom_mean is not None:
        plot_chromosome_performance(
            chrom_mean["auprc"],
            model_name,
            asdict(args),
            timestamp,
            folder=folder,
            errors=None if chrom_err is None else chrom_err["auprc"],
        )

    return cv_path
