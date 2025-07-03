import argparse
import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

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
from graphing.graph_shap_values import plot_shap_values
from utils import setup_logger


@dataclass
class TrainArgs:
    """Command line arguments for training."""

    max_depth: int | None = None
    random_state: int = 42


def parse_args() -> TrainArgs:
    parser = argparse.ArgumentParser(description="Train DecisionTree model")
    parser.add_argument("--max_depth", type=int, default=None)
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

        model = DecisionTreeClassifier(
            max_depth=args.max_depth,
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

    final_model = DecisionTreeClassifier(
        max_depth=args.max_depth,
        random_state=args.random_state,
    )

    final_model.fit(X, y)

    feature_imp = compute_feature_importance(final_model, X.columns)
    logging.info(
        "Top 10 features by model importance:\n%s", feature_imp.head(10).to_string()
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    fi_path = plot_feature_importance(
        feature_imp, "DecisionTree", asdict(args), timestamp
    )
    shap_path = plot_shap_values(
        final_model, X, "DecisionTree", asdict(args), timestamp
    )

    for path in [fi_path, shap_path]:
        save_args(args, os.path.dirname(path))


if __name__ == "__main__":
    main()
