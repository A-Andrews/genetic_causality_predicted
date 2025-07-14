import argparse
import logging
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from model_utils import chromosome_holdout_cv, prepare_data, train_final_model

import data_consolidation.data_loading as data_loading
from utils import setup_logger


@dataclass
class TrainArgs:
    """Command line arguments for training."""

    n_estimators: int = 100
    learning_rate: float = 0.3
    max_depth: int = 6
    random_state: int = 42


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
        random_state=args.random_state,
    )

    model.fit(X_train, y_train, eval_set=eval_set)
    return model


def parse_args() -> TrainArgs:
    parser = argparse.ArgumentParser(description="Train XGBoost model")
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.3)
    parser.add_argument("--max_depth", type=int, default=6)
    parser.add_argument("--random_state", type=int, default=42)
    return TrainArgs(**vars(parser.parse_args()))


def main() -> None:
    args = parse_args()
    setup_logger(seed=args.random_state)
    logging.info("Training arguments: %s", args)

    data = data_loading.load_all_chromosomes()

    X, y = prepare_data(data)

    def build_model(X_train, y_train, *, eval_set=None):
        return build_xgb_classifier(
            X_train,
            y_train,
            args,
            eval_set=eval_set,
        )

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
        "XGBoost",
        args,
        metric_errors=metric_errors,
        fi_errors=fi_errors,
    )


if __name__ == "__main__":
    main()
