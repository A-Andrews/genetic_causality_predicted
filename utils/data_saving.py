import json
import logging
import os
from dataclasses import asdict

from utils.settings import SAVE_RESULTS_PATH


def setup_logger(seed):
    """set useful logger set-up"""
    logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)

    if seed is not None:
        logging.info(f"Seed: {seed}")


def save_args(args, model_name, timestamp) -> None:
    """Persist CLI arguments for reproducibility."""
    output_dir = os.path.join(SAVE_RESULTS_PATH, model_name, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "cli_args.json"), "w") as f:
        json.dump(asdict(args), f, indent=2)


def save_performance_metrics(performance_metrics, model_name, timestamp) -> None:
    """Save AUPRC scores for each chromosome."""
    output_dir = os.path.join(SAVE_RESULTS_PATH, model_name, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    with open(
        os.path.join(output_dir, "chromosome_performance.json"),
        "w",
    ) as f:
        json.dump(performance_metrics, f, indent=2)
    return
