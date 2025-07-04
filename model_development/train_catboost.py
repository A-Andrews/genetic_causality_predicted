import argparse
import logging
from dataclasses import asdict, dataclass

import numpy as np
from catboost import CatBoostClassifier
from model_utils import prepare_data
from training_utils import chromosome_holdout_cv, train_final_model

import data_consolidation.data_loading as data_loading
from utils import setup_logger


@dataclass
class TrainArgs:
    n_estimators: int = 200
    learning_rate: float = 0.1
    depth: int = 6
    random_state: int = 42


def parse_args() -> TrainArgs:
    parser = argparse.ArgumentParser(description="Train CatBoost model")
    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--random_state", type=int, default=42)
    return TrainArgs(**vars(parser.parse_args()))


def main() -> None:
    args = parse_args()
    setup_logger(seed=args.random_state)
    logging.info("Training arguments: %s", args)

    data = data_loading.load_all_chromosomes()
    X, y = prepare_data(data)
    cat_features = X.select_dtypes(include=["category"]).columns.tolist()

    def build_model(X_train, y_train):
        model = CatBoostClassifier(
            iterations=args.n_estimators,
            learning_rate=args.learning_rate,
            depth=args.depth,
            loss_function="Logloss",
            random_seed=args.random_state,
            verbose=False,
            cat_features=cat_features,
        )
        model.fit(X_train, y_train, verbose=False)
        return model

    chromosome_holdout_cv(data, X, y, build_model)
    train_final_model(X, y, build_model, "CatBoost", args)


if __name__ == "__main__":
    main()
