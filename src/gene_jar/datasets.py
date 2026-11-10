"""Download and load Gene JAR datasets."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlsplit
from zipfile import ZipFile

import pandas as pd
import requests

if TYPE_CHECKING:
    from . import GeneJar

ZENODO_RECORD_ID = "23965261"
MD5_HEX_LENGTH = 32
NESTED_ZIP_NAME = "output/relationship_annotations.zip"
NESTED_DATA_DIR = "relationship_annotations"

DATA_FILES = (
    ("primary_df", "summary_df.h5"),
    ("ortholog_df", "ortholog_match_subset_genes_df.h5"),
    ("flj_clone_df", "flj_clone_symbol_match_subset_genes_df.h5"),
    ("phenotype_df", "phenotype_match_subset_genes_df.h5"),
    ("hgnc_gene_group_df", "hgnc_gene_group_match_subset_genes_df.h5"),
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


def _verify_checksum(zip_path: Path, expected_checksum: str) -> bool:
    """Check a file against its Zenodo MD5 checksum."""
    algorithm, separator, expected_hash = expected_checksum.partition(":")
    if (
        separator != ":"
        or algorithm.lower() != "md5"
        or len(expected_hash) != MD5_HEX_LENGTH
        or any(char not in "0123456789abcdefABCDEF" for char in expected_hash)
    ):
        msg = f"Unsupported or malformed Zenodo checksum: {expected_checksum}"
        raise ValueError(msg)

    if not zip_path.is_file():
        return False

    digest = hashlib.md5(usedforsecurity=False)
    with zip_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().lower() == expected_hash.lower()


def _ensure_zip_cached(url: str, cache_path: Path, expected_checksum: str) -> Path:
    """Reuse a verified ZIP or download and verify a replacement atomically."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if _verify_checksum(cache_path, expected_checksum):
        return cache_path

    fd, name = tempfile.mkstemp(dir=cache_path.parent, suffix=".zip")
    os.close(fd)
    temp_path = Path(name)
    try:
        with requests.get(url, stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            with temp_path.open("wb") as stream:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        stream.write(chunk)

        if not _verify_checksum(temp_path, expected_checksum):
            msg = "Downloaded Zenodo ZIP does not match the published checksum"
            raise ValueError(msg)

        # Validate the required nested archive, rather than macOS metadata
        # members that are irrelevant to Gene JAR.
        with (
            ZipFile(temp_path) as outer,
            outer.open(NESTED_ZIP_NAME) as nested_stream,
            ZipFile(nested_stream) as inner,
        ):
            for _, filename in DATA_FILES:
                member_name = f"{NESTED_DATA_DIR}/{filename}"
                with inner.open(member_name) as source:
                    while source.read(1024 * 1024):
                        pass

        temp_path.replace(cache_path)
    finally:
        temp_path.unlink(missing_ok=True)
    return cache_path


def _extract_output(zip_path: Path, cache_dir: Path, expected_checksum: str) -> None:
    """Stage required HDF5 files from the nested ZIP and publish atomically."""
    data_dir = cache_dir / "output"
    marker_path = cache_dir / ".extracted_checksum"
    required = {filename for _, filename in DATA_FILES}

    if (
        marker_path.is_file()
        and marker_path.read_text(encoding="utf-8") == expected_checksum
        and all((data_dir / filename).is_file() for filename in required)
    ):
        return

    marker_path.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory(dir=cache_dir, prefix="extract-") as temp_name:
        staged_data = Path(temp_name) / "output"
        staged_data.mkdir()

        with (
            ZipFile(zip_path) as outer,
            outer.open(NESTED_ZIP_NAME) as nested_stream,
            ZipFile(nested_stream) as inner,
        ):
            for filename in sorted(required):
                member_name = f"{NESTED_DATA_DIR}/{filename}"
                destination = staged_data / filename
                with inner.open(member_name) as source, destination.open("wb") as dest:
                    shutil.copyfileobj(source, dest)

        missing = sorted(
            name for name in required if not (staged_data / name).is_file()
        )
        if missing:
            msg = f"Missing required files in Zenodo ZIP: {missing}"
            raise FileNotFoundError(msg)

        if data_dir.exists():
            shutil.rmtree(data_dir)
        staged_data.replace(data_dir)
        marker_path.write_text(expected_checksum, encoding="utf-8")


def load_datasets(
    gene_jar_cls: type[GeneJar],
    version: str | None = None,
) -> GeneJar:
    """Download and load the latest or selected Gene JAR dataset from Zenodo."""
    base_url = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"
    base_response = requests.get(base_url, timeout=30)
    base_response.raise_for_status()
    base_record = base_response.json()
    expected_concept_id = str(base_record["conceptrecid"])

    if version is None:
        record_url = f"{base_url}/versions/latest"
    else:
        # Follow all pages so older versions are not silently excluded.
        versions_url = f"{base_url}/versions"
        matching_record = None
        while versions_url:
            response = requests.get(versions_url, timeout=30)
            response.raise_for_status()
            listing = response.json()
            matching_record = next(
                (
                    item
                    for item in listing.get("hits", {}).get("hits", [])
                    if item.get("metadata", {}).get("version") == version
                    and str(item.get("conceptrecid")) == expected_concept_id
                ),
                None,
            )
            if matching_record is not None:
                break
            versions_url = listing.get("links", {}).get("next")
            if versions_url:
                parsed_next = urlsplit(versions_url)
                if parsed_next.scheme != "https" or parsed_next.netloc != "zenodo.org":
                    msg = "Invalid Zenodo versions pagination URL"
                    raise ValueError(msg)

        if matching_record is None:
            msg = f"Zenodo dataset version not found: {version}"
            raise ValueError(msg)
        record_url = f"https://zenodo.org/api/records/{matching_record['id']}"

    response = requests.get(record_url, timeout=30)
    response.raise_for_status()
    record = response.json()
    if str(record["conceptrecid"]) != expected_concept_id:
        msg = "Zenodo record belongs to a different dataset"
        raise ValueError(msg)
    if version is not None and record.get("metadata", {}).get("version") != version:
        msg = f"Zenodo record does not match requested version: {version}"
        raise ValueError(msg)

    record_id = str(record["id"])
    zip_file = next(
        (item for item in record.get("files", []) if item.get("key") == "output.zip"),
        None,
    )
    if zip_file is None:
        msg = f"output.zip not found in Zenodo record {record_id}"
        raise FileNotFoundError(msg)

    download_url = zip_file["links"]["self"]
    parsed_url = urlsplit(download_url)
    if parsed_url.scheme != "https" or parsed_url.netloc != "zenodo.org":
        msg = "Download URL must use https://zenodo.org"
        raise ValueError(msg)

    expected_checksum = zip_file["checksum"]
    cache_dir = Path.home() / ".cache" / "gene-jar" / record_id
    zip_path = cache_dir / "output.zip"
    zip_was_invalid = zip_path.is_file() and not _verify_checksum(
        zip_path, expected_checksum
    )
    _ensure_zip_cached(download_url, zip_path, expected_checksum)
    if zip_was_invalid:
        (cache_dir / ".extracted_checksum").unlink(missing_ok=True)

    _extract_output(zip_path, cache_dir, expected_checksum)
    data_dir = cache_dir / "output"
    dataframes = {
        key: pd.read_hdf(data_dir / filename, key="df") for key, filename in DATA_FILES
    }
    return gene_jar_cls(**dataframes)
