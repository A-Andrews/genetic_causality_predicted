import argparse
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score

from data.load_and_merge_data import load_all_chromosomes
from data.prepare_data import prepare_data
from utils.data_saving import setup_logger


@dataclass
class TrainArgs:
    """Command line arguments for training."""

    n_estimators: int = 100
    learning_rate: float = 0.3
    max_depth: int = 6
    random_state: int = 42
    use_graph_annotations: bool = False


def compute_scale_pos_weight(y: pd.Series) -> float:
    """Return ``scale_pos_weight`` for imbalanced datasets."""
    pos = np.sum(y == True)
    neg = np.sum(y == False)
    return (neg / pos) if pos > 0 else 1.0


def chromosome_holdout_cv(
    data: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    args: TrainArgs,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame | None,
    pd.DataFrame | None,
    pd.DataFrame | None,
]:
    n_runs = 5
    rng = np.random.default_rng(args.random_state)
    chromosomes = sorted(data["chrom"].unique())
    run_auprcs = []
    fi_runs = []
    chrom_runs = []
    for run in range(n_runs):
        logging.info("Starting CV run %d/%d", run + 1, n_runs)
        fold_auprcs = []
        fold_weights = []
        fi_list = []
        for chrom in chromosomes:
            train_mask = data["chrom"] != chrom
            X_train, y_train = X[train_mask], y[train_mask]
            X_val, y_val = X[~train_mask], y[~train_mask]

            model = xgb.XGBClassifier(
                eval_metric="logloss",
                enable_categorical=True,
                base_score=0.5,
                n_estimators=args.n_estimators,
                learning_rate=args.learning_rate,
                max_depth=args.max_depth,
                scale_pos_weight=compute_scale_pos_weight(y_train),
                random_state=args.random_state,
            )

            model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

            # XGBoost can handle DataFrames directly while TabNet requires ndarrays
            y_pred = model.predict_proba(X_val)[:, 1]

            auprc = average_precision_score(y_val, y_pred)

            fold_auprcs.append(auprc)
            fold_weights.append(len(y_val))

            fi = pd.Series(model.feature_importances_, X.columns)
            fi.sort_values(ascending=False)

            fi_list.append(fi)

            logging.info(
                "Run %d, Chromosome %s - AUPRC: %.3f | ROC-AUC: %.3f | Positives: %d",
                run + 1,
                chrom,
                auprc,
                (y_val == True).sum(),
            )

        fold_df = pd.DataFrame(fold_auprcs, index=chromosomes)
        fold_sizes = pd.Series(fold_weights, index=chromosomes)
        weighted_mean = fold_df.mul(fold_sizes, axis=0).sum() / fold_sizes.sum()
        run_auprcs.append(weighted_mean)

        fi_df = pd.concat(fi_list, axis=1)
        fi_df.columns = chromosomes
        fi_weighted = fi_df.mul(fold_sizes, axis=1).sum(axis=1) / fold_sizes.sum()
        fi_runs.append(fi_weighted)

        chrom_runs.append(fold_df)

    metrics_df = pd.DataFrame(run_auprcs)
    logging.info(
        "Mean CV AUPRC over %d runs: %.3f +/- %.3f",
        n_runs,
        metrics_df["auprc"].mean(),
        metrics_df["auprc"].std() / np.sqrt(n_runs),
    )
    fi_df = pd.concat(fi_runs, axis=1)
    chrom_mean = None
    chrom_err = None

    chrom_concat = pd.concat(chrom_runs, keys=range(n_runs))
    chrom_mean = chrom_concat.groupby(level=1).mean()
    chrom_err = chrom_concat.groupby(level=1).std().div(np.sqrt(n_runs))

    return metrics_df, fi_df, chrom_mean, chrom_err


def parse_args() -> TrainArgs:
    parser = argparse.ArgumentParser(description="Train XGBoost model")
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.3)
    parser.add_argument("--max_depth", type=int, default=6)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--use_graph_annotations", type=bool, default=False)
    return TrainArgs(**vars(parser.parse_args()))


def main() -> None:
    args = parse_args()
    setup_logger(seed=args.random_state)
    logging.info("Training arguments: %s", args)

    data = load_all_chromosomes(
        include_graph=args.use_graph_annotations,
    )

    X, y = prepare_data(data)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    auprc_df, fi_df, chrom_mean, chrom_err = chromosome_holdout_cv(
        data,
        X,
        y,
        args,
    )

    # (
    #     cv_metrics,
    #     fi_df,
    #     chrom_mean,
    #     chrom_err,
    #     _,
    #     perm_df,
    #     perm_err_df,
    # ) = chromosome_holdout_cv(
    #     data,
    #     X,
    #     y,
    #     build_model,
    #     n_runs=args.n_runs,
    #     random_state=args.random_state,
    #     collect_importance=True,
    #     compute_shap=args.compute_shap,
    #     compute_permutation=args.compute_permutation,
    #     return_chrom_metrics=True,
    #     bootstrap=args.bootstrap,
    #     bootstrap_samples=args.bootstrap_samples,
    #     neg_frac=args.neg_frac,
    # )
    # plot_cv_results(
    #     cv_metrics,
    #     fi_df,
    #     chrom_mean,
    #     chrom_err,
    #     perm_df,
    #     perm_err_df,
    #     args,
    #     "XGBoost",
    #     timestamp,
    # )


if __name__ == "__main__":
    main()
