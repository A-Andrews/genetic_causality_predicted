import argparse
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime

from model_utils import compute_feature_importance, prepare_data, save_args
from pytorch_tabnet.tab_model import TabNetClassifier
from training_utils import chromosome_holdout_cv

import data_consolidation.data_loading as data_loading
from graphing.graph_importances import plot_feature_importance
from utils import setup_logger


@dataclass
class TrainArgs:
    """Command line arguments for training."""

    n_d: int = 8
    n_a: int = 8
    n_steps: int = 3
    gamma: float = 1.5
    lambda_sparse: float = 1e-3
    max_epochs: int = 50
    batch_size: int = 1024
    random_state: int = 42


def parse_args() -> TrainArgs:
    parser = argparse.ArgumentParser(description="Train TabNet model")
    parser.add_argument("--n_d", type=int, default=8)
    parser.add_argument("--n_a", type=int, default=8)
    parser.add_argument("--n_steps", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=1.5)
    parser.add_argument("--lambda_sparse", type=float, default=1e-3)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--random_state", type=int, default=42)
    return TrainArgs(**vars(parser.parse_args()))


def main() -> None:
    args = parse_args()
    setup_logger(seed=args.random_state)
    logging.info("Training arguments: %s", args)

    data = data_loading.load_all_chromosomes()
    categorical_cols = data.select_dtypes(include=["category"]).columns.tolist()
    cat_idxs = [data.columns.get_loc(col) for col in categorical_cols]
    cat_dims = [data[col].nunique() for col in categorical_cols]
    for col in categorical_cols:
        data[col] = data[col].cat.codes
    X, y = prepare_data(data)

    def build_model(X_train, y_train) -> TabNetClassifier:

        model = TabNetClassifier(
            n_d=args.n_d,
            n_a=args.n_a,
            n_steps=args.n_steps,
            gamma=args.gamma,
            lambda_sparse=args.lambda_sparse,
            seed=args.random_state,
            cat_idxs=cat_idxs,
            cat_dims=cat_dims,
            cat_emb_dim=3,
        )
        model.fit(X_train.values, y_train.values, args.max_epochs, args.batch_size)
        return build_model

    chromosome_holdout_cv(
        data,
        X,
        y,
        build_model,
    )

    final_model = build_model()
    final_model.fit(
        X.values,
        y.values,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
    )

    feature_imp = compute_feature_importance(final_model, X.columns)
    logging.info(
        "Top 10 features by model importance:\n%s",
        feature_imp.head(10).to_string(),
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    fi_path = plot_feature_importance(feature_imp, "TabNet", asdict(args), timestamp)

    for path in [fi_path]:
        save_args(args, os.path.dirname(path))


if __name__ == "__main__":
    main()
