import json
import logging
import os
from dataclasses import asdict


def setup_logger(seed):
    """set useful logger set-up"""
    logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)

    if seed is not None:
        logging.info(f"Seed: {seed}")


def save_args(args, directory: str) -> None:
    """Persist CLI arguments for reproducibility."""

    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "cli_args.json"), "w") as f:
        json.dump(asdict(args), f, indent=2)

def save_performance_metrics(metrics: dict, directory: str) -> None:
    return
