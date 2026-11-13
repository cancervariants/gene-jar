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
from tqdm.auto import tqdm

if TYPE_CHECKING:
    from . import GeneJar

ZENODO_RECORD_ID = "23965261"
MD5_HEX_LENGTH = 32
NESTED_ZIP_NAME = "output/relationship_annotations.zip"
NESTED_DATA_DIR = "relationship_annotations"
ZENODO_API_BASE_URL = "https://zenodo.org/api/records/"

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
    """Check a file against its Zenodo MD5 checksum.

    :param zip_path: Path to the ZIP file to verify.
    :param expected_checksum: Expected Zenodo MD5 checksum.
    :return: True if the file exists and its checksum matches, otherwise False.
    :raises ValueError: If the checksum format is invalid or unsupported.
    :raises OSError: If the file cannot be read.
    """
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
    """Reuse a verified ZIP or atomically download a replacement.

    :param url: URL from which to download the ZIP file.
    :param cache_path: Local path for the cached ZIP file.
    :param expected_checksum: Expected Zenodo MD5 checksum.
    :return: Path to the verified cached ZIP file.
    :raises ValueError: If the checksum is malformed or verification fails.
    :raises requests.RequestException: If the download fails.
    :raises OSError: If a filesystem operation fails.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Reuse only a checksum-verified archive; invalidate extracted data otherwise.
    if _verify_checksum(cache_path, expected_checksum):
        return cache_path
    (cache_path.parent / ".extracted_checksum").unlink(missing_ok=True)

    # Download to a temporary file so interrupted transfers never replace the cache.
    fd, name = tempfile.mkstemp(dir=cache_path.parent, suffix=".zip")
    os.close(fd)
    temp_path = Path(name)
    try:
        with requests.get(url, stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            total_size = int(getattr(response, "headers", {}).get("content-length", 0))
            with (
                temp_path.open("wb") as stream,
                tqdm(
                    total=total_size or None,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc="Downloading Gene JAR datasets",
                ) as progress,
            ):
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        stream.write(chunk)
                        progress.update(len(chunk))

        # Publish the downloaded archive only after its checksum matches.
        if not _verify_checksum(temp_path, expected_checksum):
            msg = "Downloaded Zenodo ZIP does not match the published checksum"
            raise ValueError(msg)
        temp_path.replace(cache_path)
    finally:
        temp_path.unlink(missing_ok=True)
    return cache_path


def _extract_output(zip_path: Path, cache_dir: Path, expected_checksum: str) -> None:
    """Stage only required HDF5 files from the nested ZIP and publish atomically.

    :param zip_path: Path to the verified ZIP archive.
    :param cache_dir: Directory where the extracted files are cached.
    :param expected_checksum: Checksum identifying the extracted dataset version.
    :return: None.
    :raises FileNotFoundError: If required extracted files are missing.
    :raises BadZipFile: If the ZIP or nested archive is invalid.
    :raises KeyError: If a required archive member is missing.
    :raises OSError: If extraction or filesystem operations fail.
    """
    data_dir = cache_dir / "output"
    marker_path = cache_dir / ".extracted_checksum"
    required = {filename for _, filename in DATA_FILES}

    if (
        marker_path.is_file()
        and marker_path.read_text(encoding="utf-8") == expected_checksum
        and all((data_dir / filename).is_file() for filename in required)
    ):
        return

    # Extract required members into a temporary directory, validating as we read.
    marker_path.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory(dir=cache_dir, prefix="extract-") as temp_name:
        staged_data = Path(temp_name) / "output"
        staged_data.mkdir()

        # Extract only the required HDF5 files, ignoring unrelated archive contents.
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

        # Publish only after all required members have been extracted successfully.
        if data_dir.exists():
            shutil.rmtree(data_dir)
        staged_data.replace(data_dir)
        marker_path.write_text(expected_checksum, encoding="utf-8")


def load_datasets(
    gene_jar_cls: type[GeneJar],
    version: str | None = None,
) -> GeneJar:
    """Load the latest dataset or a specific Zenodo record ID (version).

    :param gene_jar_cls: GeneJar class used to construct the loaded dataset.
    :param version: Numeric Zenodo record ID, or None for the latest version.
    :return: GeneJar instance populated with the downloaded datasets.
    :raises ValueError: If the version, record, URL, or checksum is invalid.
    :raises FileNotFoundError: If required dataset files are missing.
    :raises requests.RequestException: If a Zenodo API request or download fails.
    :raises BadZipFile: If the ZIP or nested archive is invalid.
    :raises KeyError: If required metadata or archive members are missing.
    :raises OSError: If a filesystem operation fails.
    """
    # Resolve and validate the requested Zenodo release.
    load_url = f"{ZENODO_API_BASE_URL}{ZENODO_RECORD_ID}"
    base_response = requests.get(load_url, timeout=30)
    base_response.raise_for_status()
    base_record = base_response.json()
    expected_concept_id = str(base_record["conceptrecid"])

    if version is None:
        # Default to the latest published version of this Zenodo dataset.
        record_url = f"{load_url}/versions/latest"
    else:
        # Zenodo does not require metadata.version; use an immutable record ID.
        if not version.isascii() or not version.isdecimal():
            msg = f"Zenodo version must be a numeric record ID: {version}"
            raise ValueError(msg)
        record_url = f"{ZENODO_API_BASE_URL}{version}"

    response = requests.get(record_url, timeout=30)
    response.raise_for_status()
    record = response.json()
    if str(record["conceptrecid"]) != expected_concept_id:
        msg = "Zenodo record belongs to a different dataset"
        raise ValueError(msg)
    if version is not None and str(record["id"]) != version:
        msg = f"Zenodo record does not match requested ID: {version}"
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
    # Verify or download the archive, then reuse or rebuild the extracted files.
    _ensure_zip_cached(download_url, zip_path, expected_checksum)
    _extract_output(zip_path, cache_dir, expected_checksum)
    # Load the extracted HDF5 tables into a GeneJar instance.
    data_dir = cache_dir / "output"
    dataframes = {
        key: pd.read_hdf(data_dir / filename, key="df") for key, filename in DATA_FILES
    }
    return gene_jar_cls(**dataframes)
