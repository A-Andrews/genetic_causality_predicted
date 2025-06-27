import os
import pandas as pd
from settings import BASELINELD_PATH, PLINK_PATH, TRAITGYM_PATH

def load_baselineLD_annotations(file_path):
    """
    Load baseline LD annotations from a predefined file path.
    
    Returns:
    """
    df = pd.read_csv(file_path, sep='\t', compression='gzip')
    return df

def load_bim_file(file_path):
    """
    """
    bim_df = pd.read_csv(file_path, sep='\t', header=None, names=bim_cols)
    return bim_df

def merge_ld_bim(ld_df, bim_df):
    """
    Merge LD and BIM dataframes on SNP column.
    
    Returns:
    """
    merged_df = pd.merge(ld_df, bim_df, on='SNP', how='inner')
    return merged_df

def load_traitgym_data(file_path, split):
    """
    Load TraitGym dataset splits

    Returns:
    """
    df = pd.read_parquet(file_path)
    return df

def merge_varient_features(traitgym_df, annotation_df):
    """
    """
    merged_df = pd.merge(ld_df, bim_df, on=['chrom', 'pos', 'ref', 'alt'], how='inner')
    return merged_df