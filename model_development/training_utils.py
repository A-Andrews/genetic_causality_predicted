import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd
from model_utils import compute_feature_importance, evaluate, save_args

from graphing.graph_importances import plot_feature_importance
from graphing.graph_shap_values import plot_shap_values


def chromosome_holdout_cv(
    data: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    build_model: Callable[[pd.DataFrame, pd.Series], Any],
) -> List[float]:
    """Run chromosome hold-out cross-validation."""
    chromosomes = sorted(data["chrom"].unique())
    cv_scores = []
    for chrom in chromosomes:
        train_mask = data["chrom"] != chrom
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[~train_mask], y[~train_mask]

        model = build_model(X_train, y_train)
        y_pred = model.predict_proba(X_val)[:, 1]
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
    build_model: Callable[[pd.DataFrame, pd.Series], Any],
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

    for path in [fi_path, shap_path]:
        save_args(args, os.path.dirname(path))

    return model
