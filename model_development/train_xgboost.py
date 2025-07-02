import logging

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.utils import resample

import data_consolidation.data_loading as data_loading
from settings import BASELINELD_PATH, PLINK_PATH, TRAITGYM_PATH
from utils import setup_logger


def evaluate(y_true: pd.Series, y_pred: np.ndarray) -> dict:
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


def oversample_minority(X, y):
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
