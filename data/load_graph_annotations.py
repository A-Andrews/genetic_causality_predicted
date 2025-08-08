import os
import pandas as pd
import logging

from utils.settings import GRAPH_ANNOTATIONS_PATH

def load_graph_annotations(chromosome):
    """
    Load graph annotations from a predefined file path.

    Returns:
    pandas.DataFrame
        DataFrame containing graph annotations.
    """
    file_path = os.path.join(
        GRAPH_ANNOTATIONS_PATH, f"ldsc_SNPs_sumstats.chr{chromosome}.tree_stats.txt.gz"
    )
    df = pd.read_csv(file_path, sep="\t", compression="gzip")
    df = df.drop(columns=["#hg38_chr", "hg38_pos"], errors="ignore")

    logging.info(f"Loaded graph annotations for chromosome {chromosome} from {file_path}")
    return df
