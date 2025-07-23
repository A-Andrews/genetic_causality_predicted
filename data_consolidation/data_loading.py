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


def load_trait_directory(dir_path):
    """
    Build a SNP × trait presence/absence matrix from every *.binary.gz file
    found directly inside `dir_path`.

    Returns
    -------
    pandas.DataFrame
        Columns:
            chrom  – chromosome without the 'chr' prefix (int where possible)
            pos    – base-pair position (int)
            <trait1>, <trait2>, … – 1 if SNP is present in that trait file
    """
    dir_path = Path(dir_path)
    file_paths = sorted(fp for fp in dir_path.glob("*.binary.gz") if fp.is_file())

    if not file_paths:
        raise FileNotFoundError(f"No .binary.gz files found in {dir_path}")

    trait_to_snps = {}
    trait_order = []

    # 1) Read each file and collect its SNPs
    for fp in file_paths:
        trait = fp.stem.replace(".binary", "")  # drop .binary if present
        trait_order.append(trait)
        snp_set = set()

        with gzip.open(fp, "rt") as f:
            for line in f:
                if not line.strip():
                    continue
                _, chrom_raw, pos_str, *_ = line.rstrip("\n").split("\t")

                chrom = chrom_raw.lower().removeprefix("chr")
                pos = int(pos_str)
                snp_set.add((chrom, pos))

        trait_to_snps[trait] = snp_set

    # 2) Master list of all unique (chrom, pos) pairs
    def sort_key(item):
        chrom, pos = item
        return (0, int(chrom)) if chrom.isdigit() else (1, chrom), pos

    all_snps = sorted(
        {snp for snps in trait_to_snps.values() for snp in snps},
        key=sort_key,
    )

    # 3) Assemble the DataFrame
    data = {
        "chrom": [c for c, _ in all_snps],
        "pos": [p for _, p in all_snps],
    }
    for trait in trait_order:
        present = trait_to_snps[trait]
        data[trait] = [1 if snp in present else 0 for snp in all_snps]

    df = pd.DataFrame(data)
    df["chrom"] = pd.to_numeric(df["chrom"], errors="ignore")  # ints where possible
    return df


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

    per_snp_binaries = load_trait_directory(SNP_BINARY_PATH)
    logging.info(f"Loaded SNP binary data with shape: {per_snp_binaries.shape}")
    logging.info("Columns in per_snp_binaries: %s", per_snp_binaries.columns.tolist())
    logging.info("Example data from per_snp_binaries:\n%s", per_snp_binaries.head())

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
