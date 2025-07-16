import argparse
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime

import numpy as np
from model_utils import (
    chromosome_holdout_cv,
    compute_feature_importance,
    compute_permutation_importance,
    prepare_data,
    save_args,
)
from pytorch_tabnet.tab_model import TabNetClassifier

import data_consolidation.data_loading as data_loading
from graphing.graph_importances import plot_feature_importance
from graphing.graph_model_metrics import plot_chromosome_performance, plot_model_metrics
from utils import setup_logger


@dataclass
class TrainArgs:
    """Command line arguments for training."""

    n_d: int = 8
    n_a: int = 8
    n_steps: int = 3
    gamma: float = 1.5
    lambda_sparse: float = 1e-3
    max_epochs: int = 50
    batch_size: int = 1024
    patience: int = 15
    num_workers: int = 0
    random_state: int = 42
    n_runs: int = 1
    bootstrap: bool = False
    bootstrap_samples: int = 1
    compute_shap: bool = False


def parse_args() -> TrainArgs:
    parser = argparse.ArgumentParser(description="Train TabNet model")
    parser.add_argument("--n_d", type=int, default=8)
    parser.add_argument("--n_a", type=int, default=8)
    parser.add_argument("--n_steps", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=1.5)
    parser.add_argument("--lambda_sparse", type=float, default=1e-3)
    parser.add_argument("--max_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--n_runs", type=int, default=1)
    parser.add_argument("--bootstrap", type=bool, default=False)
    parser.add_argument("--bootstrap_samples", type=int, default=1)
    parser.add_argument("--compute_shap", type=bool, default=False)
    return TrainArgs(**vars(parser.parse_args()))


def main() -> None:
    args = parse_args()
    setup_logger(seed=args.random_state)
    logging.info("Training arguments: %s", args)

    data = data_loading.load_all_chromosomes()
    X, y = prepare_data(data)
    categorical_cols = X.select_dtypes(include=["category"]).columns.tolist()
    cat_idxs = [X.columns.get_loc(col) for col in categorical_cols]
    cat_dims = []
    for col in categorical_cols:
        X[col] = X[col].cat.add_categories("missing").fillna("missing")
        X[col] = X[col].cat.codes
        max_val = X[col].max()
        if (X[col] < 0).any():
            raise ValueError(f"Negative code found in column {col}, check encoding.")
        cat_dims.append(max_val + 1)
    for col, dim in zip(categorical_cols, cat_dims):
        print(f"{col}: max code = {X[col].max()}, embedding dim = {dim}")

    def build_model(
        X_train, y_train, *, eval_set=None, random_state=None
    ) -> TabNetClassifier:

        model = TabNetClassifier(
            n_d=args.n_d,
            n_a=args.n_a,
            n_steps=args.n_steps,
            gamma=args.gamma,
            lambda_sparse=args.lambda_sparse,
            seed=args.random_state if random_state is None else random_state,
            cat_idxs=cat_idxs,
            cat_dims=cat_dims,
            cat_emb_dim=3,
        )
        if eval_set is not None:
            eval_set = [(X.values, y.values) for X, y in eval_set]
        model.fit(
            X_train.values,
            y_train.values,
            eval_set=eval_set,
            max_epochs=args.max_epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        return model

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    cv_metrics, fi_df, chrom_mean, chrom_err, _ = chromosome_holdout_cv(
        data,
        X,
        y,
        build_model,
        n_runs=args.n_runs,
        random_state=args.random_state,
        collect_importance=True,
        compute_shap=args.compute_shap,
        return_chrom_metrics=True,
        bootstrap=args.bootstrap,
        bootstrap_samples=args.bootstrap_samples,
    )
    metric_errors = cv_metrics.std().div(np.sqrt(len(cv_metrics))).to_dict()
    fi_errors = (
        fi_df.std(axis=1).div(np.sqrt(fi_df.shape[1])) if fi_df is not None else None
    )

    plot_dir = "graphs/cv"
    cv_path = plot_model_metrics(
        cv_metrics.mean().to_dict(),
        "TabNet CV",
        asdict(args),
        timestamp,
        folder=plot_dir,
        errors=metric_errors,
    )
    if fi_df is not None:
        plot_feature_importance(
            fi_df.mean(axis=1),
            "TabNet",
            asdict(args),
            timestamp,
            folder=plot_dir,
            errors=fi_errors,
        )
    save_args(args, os.path.dirname(cv_path))

    if chrom_mean is not None:
        plot_chromosome_performance(
            chrom_mean["auprc"],
            "TabNet",
            asdict(args),
            timestamp,
            errors=None if chrom_err is None else chrom_err["auprc"],
        )


if __name__ == "__main__":
    main()
