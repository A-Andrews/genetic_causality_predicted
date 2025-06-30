import logging

import numpy as np
import xgboost as xgb
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

import data_consolidation.data_loading as data_loading
from settings import BASELINELD_PATH, PLINK_PATH, TRAITGYM_PATH
from utils import setup_logger

setup_logger(seed=42)  # Initialize logger with a seed for reproducibility

data = data_loading.load_data(chromosome=11)  # Load data for chromosome 1

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

# Train XGBoost model
model = xgb.XGBClassifier(
    eval_metric="logloss",
    enable_categorical=True,
    base_score=0.5,
)
model.fit(X_train, y_train)

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
