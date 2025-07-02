import logging


def setup_logger(seed):
    """set useful logger set-up"""
    logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)

    if seed is not None:
        logging.info(f"Seed: {seed}")
