import os

BASELINELD_PATH = os.path.join(
    "/well",
    "palamara",
    "projects",
    "S-LDSC_reference_files",
    "GRCh38",
    "baselineLD_v2.2",
)
PLINK_PATH = os.path.join(
    "/well", "palamara", "projects", "S-LDSC_reference_files", "GRCh38", "plink_files"
)
TRAITGYM_PATH = os.path.join(
    "/well",
    "palamara",
    "users",
    "nrw600",
    "contribution_prediction",
    "TraitGym",
    "results",
    "dataset",
    "complex_traits",
    "test.parquet",
)
GRAPH_ANNOTATIONS_PATH = os.path.join(
    "/well", "palamara", "projects", "learn_h2", "arg_sumstats_annotated"
)
SNP_BINARY_PATH = os.path.join(
    "/well",
    "palamara",
    "projects",
    "learn_h2",
    "ldsc_baseline_bed_files_hg38",
    "per_snp_binary",
)
ZARR_PATH = os.path.join(
    "/well",
    "palamara",
    "users",
    "nrw600",
    "contribution_prediction",
    "gpn-msa_data",
    "89.zarr",
)

BAR_COLOURS = ["#FE615A", "#B9D6F2", "#A0AF84"]
EDGE_COLOUR = "#D9D8D6"
