import gzip
import logging
import os
from pathlib import Path

import pandas as pd

from settings import (
    BASELINELD_PATH,
    GRAPH_ANNOTATIONS_PATH,
    PLINK_PATH,
    SNP_BINARY_PATH,
    TRAITGYM_PATH,
)

def _load_single_trait_file(file_path, trait):
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
    df["chrom"] = df["chrom"].astype(str).str.lower().str.replace("chr", "", regex=False)
    df["pos"] = df["pos"].astype("int32")
    df.drop_duplicates(subset=["chrom", "pos"], inplace=True)
    df[trait] = True
    return df


def load_trait_directory(dir_path):
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
    dir_path = Path(dir_path)
    file_paths = sorted(fp for fp in dir_path.glob("*.binary.gz") if fp.is_file())

    if not file_paths:
        raise FileNotFoundError(f"No .binary.gz files found in {dir_path}")

    master_df = None
    trait_order = []

    for fp in file_paths:
        trait = fp.stem.replace(".binary", "")
        trait_order.append(trait)
        df = _load_single_trait_file(fp, trait)
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
        master_df[trait] = pd.Series(master_df[trait], dtype=pd.SparseDtype("bool", False))

    master_df.reset_index(drop=True, inplace=True)

    return master_df


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
    return df


def merge_graph_annotations(graph_df, df):
    graph_df = graph_df.rename(columns={"SNP_ID": "SNP"})
    merged_df = pd.merge(df, graph_df, on="SNP", how="inner")
    return merged_df


def load_baselineLD_annotations(file_path):
    """
    Load baseline LD annotations from a predefined file path.

    Returns:
    """
    df = pd.read_csv(file_path, sep="\t", compression="gzip")
    return df


def load_bim_file(file_path):
    """ """

    bim_cols = ["chrom", "SNP", "genetic_dist", "pos", "ref", "alt"]
    bim_df = pd.read_csv(file_path, sep="\t", header=None, names=bim_cols)

    return bim_df


def merge_ld_bim(ld_df, bim_df):
    """
    Merge LD and BIM dataframes on SNP column.

    Returns:
    """

    merged_df = pd.merge(ld_df, bim_df, on="SNP", how="inner")

    return merged_df


def load_traitgym_data(file_path, split):
    """
    Load TraitGym dataset splits

    Returns:
    """
    df = pd.read_parquet(file_path)
    return df


def merge_varient_features(traitgym_df, annotation_df):
    """ """
    for df in (traitgym_df, annotation_df):
        df["chrom"] = df["chrom"].astype(str)
        df["pos"] = df["pos"].astype(int)
        df["ref"] = df["ref"].astype(str)
        df["alt"] = df["alt"].astype(str)

    before_traitgym_rows = len(traitgym_df)
    before_annotation_rows = len(annotation_df)
    merged_df = pd.merge(
        traitgym_df,
        annotation_df,
        on=["chrom", "pos", "ref", "alt"],
        how="inner",
    )

    rows_after = len(merged_df)
    traitgym_lost = before_traitgym_rows - rows_after
    annotation_lost = before_annotation_rows - rows_after
    logging.info(
        "Merged variant features; lost %s / %s rows from traitgym_df and %s / %s rows from annotation_df",
        traitgym_lost,
        before_traitgym_rows,
        annotation_lost,
        before_annotation_rows,
    )
    return merged_df


def load_data(chromosome, include_graph=False, per_snp_df=None):
    """
    Load data for a specific chromosome.

    Parameters
    ----------
    chromosome : int or str
        Chromosome identifier.
    include_graph : bool, optional
        Whether to include graph annotations. Defaults to ``False``.
    per_snp_df : pandas.DataFrame or None, optional
        Optional dataframe containing per-SNP binaries for this chromosome.

    Returns
    -------
    pandas.DataFrame
        Dataframe with baseline annotations (and optionally graph and
        per-SNP features) for the requested chromosome.
    """
    ld_annotations = load_baselineLD_annotations(
        os.path.join(BASELINELD_PATH, f"baselineLD.{chromosome}.annot.gz")
    )
    bim_file = load_bim_file(
        os.path.join(PLINK_PATH, f"1000G.EUR.hg38.{chromosome}.bim")
    )
    merged_data = merge_ld_bim(ld_annotations, bim_file)

    if per_snp_df is not None:
        before_rows = len(merged_data)
        merged_data = pd.merge(
            merged_data,
            per_snp_df,
            on=["chrom", "pos"],
            how="inner",
        )
    after_rows = len(merged_data)
    logging.info(
        "Merged per_snp data for chromosome %s; lost %s / %s rows",
        chromosome,
        before_rows - after_rows,
        before_rows,
    )
    
    if include_graph:
        graph_annotations = load_graph_annotations(chromosome)
        merged_data = merge_graph_annotations(graph_annotations, merged_data)

    return merged_data


def load_all_chromosomes(
    chromosomes=None,
    include_graph=False,
    include_per_snp=False,
):
    """Load and combine data for multiple chromosomes.

    Parameters
    ----------
    chromosomes : list of int or str, optional
        Chromosome identifiers to load. Defaults to ``range(1, 23)``.
    include_graph : bool, optional
        Whether to include graph annotations in the loaded data. Defaults to ``False``.

    Returns
    -------
    pandas.DataFrame
        Concatenated dataframe containing data from all specified chromosomes.
    """
    if chromosomes is None:
        chromosomes = list(range(1, 23))

    per_snp_binaries = None

    if include_per_snp:
        per_snp_binaries = load_trait_directory(SNP_BINARY_PATH)
        logging.info(
            f"Loaded SNP binary data with shape: {per_snp_binaries.shape}"
        )

        annotation_dfs = []
    for chrom in chromosomes:
        logging.info(f"Loading chromosome {chrom}")
        chrom_snps = None
        if per_snp_binaries is not None:
            chrom_snps = per_snp_binaries[
                per_snp_binaries["chrom"].astype(str) == str(chrom)
            ]
        annotation_dfs.append(
            load_data(
                chromosome=chrom,
                include_graph=include_graph,
                per_snp_df=chrom_snps,
            )
        )

    combined_annotations = pd.concat(annotation_dfs, ignore_index=True)
    logging.info(
        f"Combined annotation dataframe shape: {combined_annotations.shape}"
    )

    traitgym_data = load_traitgym_data(
        os.path.join(TRAITGYM_PATH, "complex_traits_all", "test.parquet"),
        split="test",
    )
    combined_df = merge_varient_features(traitgym_data, combined_annotations)

    convert_columns = {
        "chrom": int,
        "ref": "category",
        "alt": "category",
        "OMIM": "category",
        "consequence": "category",
        "SNP": "category",
        "trait": "category",
    }
    for col, dtype in convert_columns.items():
        if col in combined_df.columns:
            combined_df[col] = combined_df[col].astype(dtype)
        else:
            logging.warning(f"Column {col} not found in merged_traitgym_data")
    return combined_df
