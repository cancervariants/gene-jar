"""Download and load Gene JAR datasets."""

from __future__ import annotations

import os
import tempfile
import zlib
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlsplit
from urllib.request import urlretrieve
from zipfile import BadZipFile, ZipFile

import pandas as pd

if TYPE_CHECKING:
    from . import GeneJar


DATA_FILES = (
    ("primary_df", "summary_df.h5"),
    ("ortholog_df", "ortholog_match_subset_genes_df.h5"),
    ("flj_clone_df", "flj_clone_symbol_match_subset_genes_df.h5"),
    ("phenotype_df", "phenotype_match_subset_genes_df.h5"),
    ("hgnc_gene_group_df", "gene_group_prefix_match_subset_genes_df.h5"),
    ("gene_id_df", "gene_id_match_subset_genes_df.h5"),
    ("mgi_withdrawn_df", "mgi_withdrawn_symbol_match_subset_genes_df.h5"),
    ("ncbi_related_gene_df", "ncbi_related_gene_match_subset_genes_df.h5"),
    (
        "ncbi_gene_interaction_df",
        "ncbi_gene_interaction_symbol_match_subset_genes_df.h5",
    ),
    ("ncbi_gene_neighbor_df", "ncbi_gene_neighbor_match_subset_genes_df.h5"),
    ("placeholder_df", "placeholder_match_subset_genes_df.h5"),
    ("previous_df", "previous_symbol_match_subset_genes_df.h5"),
    ("protein_mass_df", "protein_mass_match_subset_genes_df.h5"),
    ("alternate_abbreviation_df", "alt_abbrev_symbol_match_subset_genes_df.h5"),
)

GITHUB_REPO = "cancervariants/gene-jar"


def _ensure_zip_cached(url: str, cache_path: Path) -> Path:
    """Return a valid cached ZIP, downloading it if needed."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    def is_valid_zip(path: Path) -> bool:
        try:
            with ZipFile(path) as archive:
                return archive.testzip() is None
        except (BadZipFile, EOFError, OSError, zlib.error):
            return False

    if is_valid_zip(cache_path):
        return cache_path

    fd, temp_name = tempfile.mkstemp(dir=cache_path.parent, suffix=".zip")
    os.close(fd)
    temp_path = Path(temp_name)

    try:
        urlretrieve(url, temp_path)  # noqa: S310
        if not is_valid_zip(temp_path):
            msg = "Downloaded ZIP archive is invalid or incomplete"
            raise BadZipFile(msg)
        temp_path.replace(cache_path)
    finally:
        temp_path.unlink(missing_ok=True)

    return cache_path


def load_datasets(version: str, gene_jar_cls: type[GeneJar]) -> GeneJar:
    """Download and load datasets produced by gene-jar-analysis."""
    asset_name = f"gene-jar-data-{version}.zip"
    cache_dir = Path.home() / ".cache" / "gene-jar" / version
    data_dir = cache_dir / f"gene-jar-data-{version}"
    zip_path = cache_dir.parent / asset_name

    if not all((data_dir / filename).is_file() for _, filename in DATA_FILES):
        download_url = (
            f"https://github.com/{GITHUB_REPO}/releases/download/{version}/{asset_name}"
        )
        parsed_url = urlsplit(download_url)
        if parsed_url.scheme != "https" or parsed_url.netloc != "github.com":
            msg = "Release download URL must use https://github.com"
            raise ValueError(msg)

        zip_path = _ensure_zip_cached(download_url, zip_path)
        with ZipFile(zip_path) as archive:
            archive.extractall(cache_dir)

    missing_files = [
        filename for _, filename in DATA_FILES if not (data_dir / filename).is_file()
    ]
    if missing_files:
        msg = f"Missing files in downloaded release: {missing_files}"
        raise FileNotFoundError(msg)

    dataframes = {
        key: pd.read_hdf(data_dir / filename, key="df") for key, filename in DATA_FILES
    }
    return gene_jar_cls(**dataframes)
