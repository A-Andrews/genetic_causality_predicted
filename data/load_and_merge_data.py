import logging

import pandas as pd
from load_baseline_annotations import load_baseline_annotations
from load_graph_annotations import load_graph_annotations
from load_traitgym import load_traitgym_data


def merge_graph_annotations(graph_df, df):
    graph_df = graph_df.rename(columns={"SNP_ID": "SNP"})
    merged_df = pd.merge(df, graph_df, on="SNP", how="inner")
    return merged_df


def merge_traitgym_per_snp(traitgym_df, per_snp_df):
    """Merge TraitGym data with per-SNP binaries on `chrom` and `pos`."""

    for df in (traitgym_df, per_snp_df):
        df["chrom"] = df["chrom"].astype(str)
        df["pos"] = df["pos"].astype(int)

    before_traitgym_rows = len(traitgym_df)
    before_per_snp_rows = len(per_snp_df)
    merged_df = pd.merge(traitgym_df, per_snp_df, on=["chrom", "pos"], how="inner")

    rows_after = len(merged_df)
    traitgym_lost = before_traitgym_rows - rows_after
    per_snp_lost = before_per_snp_rows - rows_after
    logging.info(
        "Merged per_snp features; lost %s / %s rows from traitgym_df and %s / %s rows from per_snp_df",
        traitgym_lost,
        before_traitgym_rows,
        per_snp_lost,
        before_per_snp_rows,
    )
    return merged_df


def load_data(chromosome, per_snp_df, include_graph=False):
    """
    Load data for a specific chromosome.

    Parameters
    ----------
    chromosome : int or str
        Chromosome identifier.
    include_graph : bool, optional
        Whether to include graph annotations. Defaults to `False`.
    per_snp_df : pandas.DataFrame or None, optional
        Optional dataframe containing per-SNP binaries for this chromosome.


    Returns
    -------
    pandas.DataFrame
        Dataframe with baseline annotations (and optionally graph and
        per-SNP features) for the requested chromosome.
    """

    merged_data = per_snp_df.copy()

    if include_graph:
        graph_annotations = load_graph_annotations(chromosome)
        merged_data = merge_graph_annotations(graph_annotations, merged_data)

    return merged_data


def load_all_chromosomes(
    chromosomes=None,
    include_graph=False,
):
    """Load and combine data for multiple chromosomes.

    Parameters
    ----------
    chromosomes : list of int or str, optional
        Chromosome identifiers to load. Defaults to ``range(1, 23)``.
    include_graph : bool, optional
        Whether to include graph annotations in the loaded data. Defaults to ``False``.
    include_per_snp : bool, optional
        If ``True`` load per-SNP binaries instead of LD annotations.

    Returns
    -------
    pandas.DataFrame
        Concatenated dataframe containing data from all specified chromosomes.
    """
    if chromosomes is None:
        chromosomes = list(range(1, 23))

    per_snp_binaries = load_baseline_annotations()
    logging.info(f"Loaded SNP binary data with shape: {per_snp_binaries.shape}")

    annotation_dfs = []
    for chrom in chromosomes:
        logging.info(f"Loading chromosome {chrom}")
        chrom_snps = None

        chrom_snps = per_snp_binaries[
            per_snp_binaries["chrom"].astype(str) == str(chrom)
        ]
        annotation_dfs.append(
            load_data(
                chromosome=chrom,
                per_snp_df=chrom_snps,
                include_graph=include_graph,
            )
        )

    combined_annotations = pd.concat(annotation_dfs, ignore_index=True)
    logging.info(f"Combined annotation dataframe shape: {combined_annotations.shape}")

    traitgym_data = load_traitgym_data()

    combined_df = merge_traitgym_per_snp(traitgym_data, combined_annotations)

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
