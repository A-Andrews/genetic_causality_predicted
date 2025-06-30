import logging

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

# Train XGBoost model
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", enable_categorical=True)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict_proba(X_val)[:, 1]

# Evaluate
ap = average_precision_score(y_val, y_pred)
print(f"AUPRC: {ap:.3f}")
