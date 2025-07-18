import logging
import os

import pandas as pd

from settings import BASELINELD_PATH, GRAPH_ANNOTATIONS_PATH, PLINK_PATH, TRAITGYM_PATH


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


def load_data(chromosome, include_graph=False):
    """
    Load data for a specific chromosome.

    Returns:
    """
    ld_annotations = load_baselineLD_annotations(
        os.path.join(BASELINELD_PATH, f"baselineLD.{chromosome}.annot.gz")
    )
    bim_file = load_bim_file(
        os.path.join(PLINK_PATH, f"1000G.EUR.hg38.{chromosome}.bim")
    )
    merged_data = merge_ld_bim(ld_annotations, bim_file)

    traitgym_data = load_traitgym_data(
        os.path.join(TRAITGYM_PATH, "complex_traits_all", f"test.parquet"),
        split="test",
    )
    merged_traitgym_data = merge_varient_features(traitgym_data, merged_data)
    if include_graph:
        graph_annotations = load_graph_annotations(chromosome)
        merged_traitgym_data = merge_graph_annotations(
            graph_annotations, merged_traitgym_data
        )
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
        if col in merged_traitgym_data.columns:
            merged_traitgym_data[col] = merged_traitgym_data[col].astype(dtype)

    if len(merged_traitgym_data.select_dtypes(include="object").columns.tolist()) > 0:
        logging.warning(
            f"Object columns found in merged_traitgym_data: {merged_traitgym_data.select_dtypes(include='object').columns.tolist()}"
        )

    return merged_traitgym_data


def load_all_chromosomes(chromosomes=None, include_graph=False):
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

    all_dfs = []
    for chrom in chromosomes:
        logging.info(f"Loading chromosome {chrom}")
        all_dfs.append(load_data(chromosome=chrom, include_graph=include_graph))

    combined_df = pd.concat(all_dfs, ignore_index=True)
    logging.info(f"Combined dataframe shape: {combined_df.shape}")

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
