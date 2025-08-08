import pandas as pd

from utils.settings import TRAITGYM_PATH


def load_traitgym_data():
    """
    Load TraitGym dataset splits

    Returns:
    """
    df = pd.read_parquet(TRAITGYM_PATH)
    return df
