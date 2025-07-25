import argparse
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from model_utils import chromosome_holdout_cv, plot_cv_results, prepare_data

import data_consolidation.data_loading as data_loading
from utils import setup_logger


@dataclass
class TrainArgs:
    """Command line arguments for training."""

    n_estimators: int = 100
    learning_rate: float = 0.3
    max_depth: int = 6
    random_state: int = 42
    n_runs: int = 1
    bootstrap: bool = False
    bootstrap_samples: int = 1
    neg_frac: float = 1.0
    compute_shap: bool = False
    compute_permutation: bool = False
    use_graph_annotations: bool = False
    use_per_snp: bool = False


def compute_scale_pos_weight(y: pd.Series) -> float:
    """Return ``scale_pos_weight`` for imbalanced datasets."""
    pos = np.sum(y == True)
    neg = np.sum(y == False)
    return (neg / pos) if pos > 0 else 1.0


def build_xgb_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    args: dataclass,
    *,
    eval_set: List[Tuple[np.ndarray, np.ndarray]] | None = None,
    random_state: int | None = None,
) -> xgb.XGBClassifier:
    """Return a fitted :class:`xgboost.XGBClassifier`."""

    model = xgb.XGBClassifier(
        eval_metric="logloss",
        enable_categorical=True,
        base_score=0.5,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        scale_pos_weight=compute_scale_pos_weight(y_train),
        random_state=args.random_state if random_state is None else random_state,
    )

    model.fit(X_train, y_train, eval_set=eval_set)
    return model


def parse_args() -> TrainArgs:
    parser = argparse.ArgumentParser(description="Train XGBoost model")
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.3)
    parser.add_argument("--max_depth", type=int, default=6)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--n_runs", type=int, default=1)
    parser.add_argument("--bootstrap", type=bool, default=False)
    parser.add_argument("--bootstrap_samples", type=int, default=1)
    parser.add_argument("--neg_frac", type=float, default=1.0)
    parser.add_argument("--compute_shap", type=bool, default=False)
    parser.add_argument("--compute_permutation", type=bool, default=False)
    parser.add_argument("--use_graph_annotations", type=bool, default=False)
    parser.add_argument("--use_per_snp", type=bool, default=False)
    return TrainArgs(**vars(parser.parse_args()))


def main() -> None:
    args = parse_args()
    setup_logger(seed=args.random_state)
    logging.info("Training arguments: %s", args)

    data = data_loading.load_all_chromosomes(
        include_graph=args.use_graph_annotations,
        include_per_snp=args.use_per_snp,
    )

    X, y = prepare_data(data)

    def build_model(X_train, y_train, *, eval_set=None, random_state=None):
        return build_xgb_classifier(
            X_train,
            y_train,
            args,
            eval_set=eval_set,
            random_state=random_state,
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    (
        cv_metrics,
        fi_df,
        chrom_mean,
        chrom_err,
        _,
        perm_df,
        perm_err_df,
    ) = chromosome_holdout_cv(
        data,
        X,
        y,
        build_model,
        n_runs=args.n_runs,
        random_state=args.random_state,
        collect_importance=True,
        compute_shap=args.compute_shap,
        compute_permutation=args.compute_permutation,
        return_chrom_metrics=True,
        bootstrap=args.bootstrap,
        bootstrap_samples=args.bootstrap_samples,
        neg_frac=args.neg_frac,
    )
    plot_cv_results(
        cv_metrics,
        fi_df,
        chrom_mean,
        chrom_err,
        perm_df,
        perm_err_df,
        args,
        "XGBoost",
        timestamp,
    )


if __name__ == "__main__":
    main()
