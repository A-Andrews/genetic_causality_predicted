import argparse
import logging
from dataclasses import asdict, dataclass

import lightgbm as lgb
import numpy as np
from model_utils import prepare_data
from training_utils import chromosome_holdout_cv, train_final_model

import data_consolidation.data_loading as data_loading
from utils import setup_logger


@dataclass
class TrainArgs:
    n_estimators: int = 200
    learning_rate: float = 0.1
    max_depth: int = -1
    random_state: int = 42


def parse_args() -> TrainArgs:
    parser = argparse.ArgumentParser(description="Train LightGBM model")
    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--max_depth", type=int, default=-1)
    parser.add_argument("--random_state", type=int, default=42)
    return TrainArgs(**vars(parser.parse_args()))


def main() -> None:
    args = parse_args()
    setup_logger(seed=args.random_state)
    logging.info("Training arguments: %s", args)

    data = data_loading.load_all_chromosomes()
    X, y = prepare_data(data)

    def build_model(X_train, y_train):
        pos = np.sum(y_train == True)
        neg = np.sum(y_train == False)
        spw = (neg / pos) if pos > 0 else 1.0
        model = lgb.LGBMClassifier(
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            max_depth=args.max_depth,
            scale_pos_weight=spw,
            random_state=args.random_state,
        )
        model.fit(X_train, y_train)
        return model

    chromosome_holdout_cv(data, X, y, build_model)
    train_final_model(X, y, build_model, "LightGBM", args)


if __name__ == "__main__":
    main()
