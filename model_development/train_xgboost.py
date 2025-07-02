import logging

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, train_test_split
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

X = data.drop(columns=["label"])  # Features
y = data["label"]  # Target variable


# Split data
X_train, X_val, y_train, y_val = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

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

logging.info("Starting 5-fold cross-validation with oversampling")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
    X_train_cv, y_train_cv = X.iloc[train_idx], y.iloc[train_idx]
    X_val_cv, y_val_cv = X.iloc[val_idx], y.iloc[val_idx]

    X_train_cv, y_train_cv = oversample_minority(X_train_cv, y_train_cv)

    model = xgb.XGBClassifier(
        eval_metric="logloss",
        enable_categorical=True,
        base_score=0.5,
    )
    model.fit(X_train_cv, y_train_cv)

    y_pred_cv = model.predict_proba(X_val_cv)[:, 1]
    ap_cv = average_precision_score(y_val_cv, y_pred_cv)
    cv_scores.append(ap_cv)
    logging.info(
        f"Fold {fold} AUPRC: {ap_cv:.3f} (positives in val: {(y_val_cv == True).sum()})"
    )

logging.info(f"Mean CV AUPRC: {np.mean(cv_scores):.3f} +/- {np.std(cv_scores):.3f}")
