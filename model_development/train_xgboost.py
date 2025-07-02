import logging

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score
from sklearn.utils import resample

import data_consolidation.data_loading as data_loading
from settings import BASELINELD_PATH, PLINK_PATH, TRAITGYM_PATH
from utils import setup_logger


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

# Chromosome to hold out for validation
HOLDOUT_CHROM = 22

X = data.drop(columns=["label"])  # Features
y = data["label"]  # Target variable

train_mask = data["chrom"] != HOLDOUT_CHROM

X_train, y_train = X[train_mask], y[train_mask]
X_val, y_val = X[~train_mask], y[~train_mask]

y_train_counts = dict(zip(*np.unique(y_train, return_counts=True)))
y_val_counts = dict(zip(*np.unique(y_val, return_counts=True)))

logging.info(f"Train label distribution: {y_train_counts}")
logging.info(f"Validation label distribution: {y_val_counts}")

pos_count = np.sum(y_train == True)
neg_count = np.sum(y_train == False)
scale_pos_weight = (neg_count / pos_count) if pos_count > 0 else 1.0

# Train XGBoost model
model = xgb.XGBClassifier(
    eval_metric="aucpr",
    enable_categorical=True,
    base_score=0.5,
    scale_pos_weight=scale_pos_weight,
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=3,
)
model.fit(
    X_train,
    y_train,
)

# Predict
y_pred = model.predict_proba(X_val)[:, 1]
unique_preds, pred_counts = np.unique(y_pred, return_counts=True)
pred_distribution = dict(zip(unique_preds, pred_counts))
logging.info(f"Prediction Distribution: {pred_distribution}")
logging.info(f"Prediction min/max: {y_pred.min():.5f} / {y_pred.max():.5f}")
positive_probs = y_pred[y_val.values == True]
logging.info(f"Positive preds: {positive_probs.tolist()}")

# Evaluate
ap = average_precision_score(y_val, y_pred)
logging.info(f"AUPRC: {ap:.3f} (positives in val: {y_val_counts.get(True, 0)})")

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
        scale_pos_weight=spw,
    )
    model.fit(X_train_cv, y_train_cv)

    y_pred_cv = model.predict_proba(X_val_cv)[:, 1]
    ap_cv = average_precision_score(y_val_cv, y_pred_cv)
    cv_scores.append(ap_cv)
    logging.info(
        f"Chromosome {chrom} AUPRC: {ap_cv:.3f} (positives in val: {(y_val_cv == True).sum()})"
    )

logging.info(
    f"Mean chromosome CV AUPRC: {np.mean(cv_scores):.3f} +/- {np.std(cv_scores):.3f}"
)
