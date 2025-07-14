import argparse
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from dataclasses import dataclass

from model_utils import (
    chromosome_holdout_cv,
    compute_feature_importance,
    compute_permutation_importance,
    prepare_data,
    train_final_model,
)
from pytorch_tabnet.tab_model import TabNetClassifier

import data_consolidation.data_loading as data_loading
from graphing.graph_importances import plot_feature_importance
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

    def build_model(X_train, y_train, *, eval_set=None) -> TabNetClassifier:

        model = TabNetClassifier(
            n_d=args.n_d,
            n_a=args.n_a,
            n_steps=args.n_steps,
            gamma=args.gamma,
            lambda_sparse=args.lambda_sparse,
            seed=args.random_state,
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

    cv_metrics, fi_df = chromosome_holdout_cv(
        data,
        X,
        y,
        build_model,
        collect_importance=True,
    )
    metric_errors = cv_metrics.std().to_dict()
    fi_errors = fi_df.std(axis=1) if fi_df is not None else None

    train_final_model(
        X,
        y,
        build_model,
        "TabNet",
        args,
        metric_errors=metric_errors,
        fi_errors=fi_errors,
    )


if __name__ == "__main__":
    main()
