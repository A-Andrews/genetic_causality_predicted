import data_consolidation.data_loading as data_loading
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score

# Load LD annotations
ld_annotations = data_loading.load_baselineLD_annotations("data/baselineLD_annotations.tsv.gz")
# Load BIM file
bim_file = data_loading.load_bim_file("data/variants.bim")
# Merge LD and BIM data
merged_data = data_loading.merge_ld_bim(ld_annotations, bim_file)
# Load TraitGym dataset
traitgym_data = data_loading.load_traitgym_data("data/traitgym_dataset.parquet", split="test")
# Merge TraitGym data with LD annotations
merged_traitgym_data = data_loading.merge_varient_features(traitgym_data, merged_data)

X = merged_traitgym_data.drop(columns=['label'])  # Features
y = merged_traitgym_data['label']  # Target variable

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Train XGBoost model
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
model.fit(X_train, y_train)

# Predict
y_pred = model.predict_proba(X_val)[:, 1]

# Evaluate
ap = average_precision_score(y_val, y_pred)
print(f"AUPRC: {ap:.3f}")