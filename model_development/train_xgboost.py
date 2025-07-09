import argparse
import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
from model_utils import (
    compute_feature_importance,
    compute_permutation_importance,
    evaluate,
    oversample_minority,
    prepare_data,
    save_args,
)

import data_consolidation.data_loading as data_loading
from graphing.graph_importances import plot_feature_importance
from graphing.graph_model_metrics import plot_model_metrics
from graphing.graph_shap_values import plot_shap_values
from utils import setup_logger


@dataclass
class TrainArgs:
    """Command line arguments for training."""

    n_estimators: int = 100
    learning_rate: float = 0.3
    max_depth: int = 6
    random_state: int = 42


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

    logging.info("Starting chromosome hold-out cross-validation")
    chromosomes = sorted(data["chrom"].unique())
    cv_scores = []

    for chrom in chromosomes:
        train_mask = data["chrom"] != chrom
        X_train_cv, y_train_cv = X[train_mask], y[train_mask]
        X_val_cv, y_val_cv = X[~train_mask], y[~train_mask]

        # X_train_cv, y_train_cv = oversample_minority(X_train_cv, y_train_cv)

        pos = np.sum(y_train_cv == True)
        neg = np.sum(y_train_cv == False)
        spw = (neg / pos) if pos > 0 else 1.0

        model = xgb.XGBClassifier(
            eval_metric="logloss",
            enable_categorical=True,
            base_score=0.5,
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            max_depth=args.max_depth,
            scale_pos_weight=spw,
            random_state=args.random_state,
        )
        model.fit(X_train_cv, y_train_cv)

        y_pred_cv = model.predict_proba(X_val_cv)[:, 1]
        metrics_cv = evaluate(y_val_cv, y_pred_cv)
        cv_scores.append(metrics_cv["auprc"])
        logging.info(
            "Chromosome %s - AUPRC: %.3f | ROC-AUC: %.3f | Positives: %d",
            chrom,
            metrics_cv["auprc"],
            metrics_cv["roc_auc"],
            (y_val_cv == True).sum(),
        )

    logging.info(
        "Mean chromosome CV AUPRC: %.3f +/- %.3f",
        np.mean(cv_scores),
        np.std(cv_scores),
    )

    pos_total = np.sum(y == True)
    neg_total = np.sum(y == False)
    spw_total = (neg_total / pos_total) if pos_total > 0 else 1.0

    final_model = xgb.XGBClassifier(
        eval_metric="logloss",
        enable_categorical=True,
        base_score=0.5,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        scale_pos_weight=spw_total,
        random_state=args.random_state,
    )

    final_model.fit(X, y)

    feature_imp = compute_feature_importance(final_model, X.columns)
    logging.info(
        "Top 10 features by model importance:\n%s", feature_imp.head(10).to_string()
    )

    # perm_imp = compute_permutation_importance(final_model, X, y)
    # logging.info(
    #     "Top 10 features by permutation importance:\n%s",
    #     perm_imp.head(10).to_string(),
    # )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    fi_path = plot_feature_importance(feature_imp, "XGBoost", asdict(args), timestamp)
    shap_path = plot_shap_values(final_model, X, "XGBoost", asdict(args), timestamp)
    y_pred_final = final_model.predict_proba(X)[:, 1]
    metrics_final = evaluate(y, y_pred_final)
    metrics_path = plot_model_metrics(metrics_final, "XGBoost", asdict(args), timestamp)

    for path in [fi_path, shap_path, metrics_path]:
        save_args(args, os.path.dirname(path))


if __name__ == "__main__":
    main()
