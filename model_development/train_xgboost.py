import logging

import numpy as np
import pandas as pd
import xgboost as xgb
from model_utils import (
    compute_feature_importance,
    compute_permutation_importance,
    evaluate,
    oversample_minority,
)

import data_consolidation.data_loading as data_loading
from settings import BASELINELD_PATH, PLINK_PATH, TRAITGYM_PATH
from utils import setup_logger

setup_logger(seed=42)  # Initialize logger with a seed for reproducibility

data = data_loading.load_all_chromosomes()

logging.info("Data loaded successfully.")
logging.info(f"Data shape: {data.shape}")
logging.info(f"Columns: {data.columns.tolist()}")
logging.info(f"Column types:\n{data.dtypes}")
logging.info(
    f"Object columns:\n{data.select_dtypes(include='object').columns.tolist()}"
)

X = data.drop(columns=["label"])  # Features
y = data["label"]  # Target variable

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
]
X = X.drop(columns=[col for col in leaky_cols if col in X.columns])

# ---------------------------------------------------------------------------
# K-Fold Cross-Validation with optional oversampling
# ---------------------------------------------------------------------------

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
        n_estimators=100,
        scale_pos_weight=spw,
        random_state=42,
    )
    model.fit(X_train_cv, y_train_cv)

    y_pred_cv = model.predict_proba(X_val_cv)[:, 1]
    metrics_cv = evaluate(y_val_cv, y_pred_cv)
    cv_scores.append(metrics_cv["auprc"])
    logging.info(
        f"Chromosome {chrom} - AUPRC: {metrics_cv['auprc']:.3f} | ROC-AUC: {metrics_cv['roc_auc']:.3f} | Positives: {(y_val_cv == True).sum()}"
    )

logging.info(
    f"Mean chromosome CV AUPRC: {np.mean(cv_scores):.3f} +/- {np.std(cv_scores):.3f}"
)

pos_total = np.sum(y == True)
neg_total = np.sum(y == False)
spw_total = (neg_total / pos_total) if pos_total > 0 else 1.0

final_model = xgb.XGBClassifier(
    eval_metric="logloss",
    enable_categorical=True,
    base_score=0.5,
    n_estimators=100,
    scale_pos_weight=spw_total,
    random_state=42,
)
final_model.fit(X, y)

feature_imp = compute_feature_importance(final_model, X.columns)
logging.info(
    "Top 10 features by model importance:\n%s", feature_imp.head(10).to_string()
)

perm_imp = compute_permutation_importance(final_model, X, y)
logging.info(
    "Top 10 features by permutation importance:\n%s", perm_imp.head(10).to_string()
)
