import logging

def setup_logger(seed):
    """set useful logger set-up"""
    logging.basicConfig(
        format="%(asctime)s %(message)s", encoding="utf-8", level=logging.INFO
    )
    # logging.debug(f"Pytorch version: {torch.__version__}")
    if seed is not None:
        logging.info(f"Seed: {seed}")