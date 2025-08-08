import gzip
import pandas as pd
import logging
from pathlib import Path
from utils.settings import SNP_BINARY_PATH

def load_single_trait_file(file_path, trait):
    """Load a single ``*.binary.gz`` file as a dataframe.

    Parameters
    ----------
    file_path : path-like
        Path to the gzipped file.
    trait : str
        Name of the trait being loaded.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ``chrom`` and ``pos`` plus a boolean column
        for ``trait`` indicating SNP presence.
    """
    with gzip.open(file_path, "rt") as fh:
        df = pd.read_csv(
            fh,
            sep="\t",
            usecols=[1, 2],
            header=None,
            names=["chrom", "pos"],
        )

    df.dropna(inplace=True)
    df["chrom"] = (
        df["chrom"].astype(str).str.lower().str.replace("chr", "", regex=False)
    )
    df["pos"] = df["pos"].astype("int32")
    df.drop_duplicates(subset=["chrom", "pos"], inplace=True)
    df[trait] = True
    return df


def load_baseline_annotations():
    """
    Build a SNP × trait presence/absence matrix from every *.binary.gz file
    found directly inside `dir_path`. The first column of each file is ignored
    since it only contains a variant identifier.

    Returns
    -------
    pandas.DataFrame
        Columns:
            chrom  – chromosome without the 'chr' prefix (categorical)
            pos    – base-pair position (int32)
            <trait1>, <trait2>, … – boolean flag indicating SNP presence
    """
    dir_path = Path(SNP_BINARY_PATH)
    file_paths = sorted(fp for fp in dir_path.glob("*.binary.gz") if fp.is_file())

    if not file_paths:
        raise FileNotFoundError(f"No .binary.gz files found in {dir_path}")

    master_df = None
    trait_order = []

    for fp in file_paths:
        trait = fp.stem.replace(".binary", "")
        trait_order.append(trait)
        df = load_single_trait_file(fp, trait)
        logging.info(f"Loaded {len(df)} SNPs for trait '{trait}' from file {fp}")
        if master_df is None:
            master_df = df
        else:
            master_df = pd.merge(master_df, df, on=["chrom", "pos"], how="outer")
            master_df.fillna(False, inplace=True)
    if master_df is None:
        return pd.DataFrame(columns=["chrom", "pos"])

    for trait in trait_order:
        master_df[trait] = master_df[trait].astype(bool)

    master_df["pos"] = master_df["pos"].astype("int32")

    chrom_numeric = pd.to_numeric(master_df["chrom"], errors="coerce")
    master_df["_chrom_numeric"] = chrom_numeric
    master_df.sort_values(by=["_chrom_numeric", "chrom", "pos"], inplace=True)
    master_df.drop(columns="_chrom_numeric", inplace=True)
    master_df["chrom"] = pd.Categorical(master_df["chrom"])

    for trait in trait_order:
        master_df[trait] = pd.Series(
            master_df[trait], dtype=pd.SparseDtype("bool", False)
        )

    master_df.reset_index(drop=True, inplace=True)

    return master_df
