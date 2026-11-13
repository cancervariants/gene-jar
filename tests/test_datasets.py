"""Tests for downloading the latest or an explicit Zenodo record ID."""

from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd
import pytest

from gene_jar import GeneJar, SymbolCategory, datasets

# Expected archive layout is intentionally independent of production DATA_FILES.
EXPECTED_DATA_FILES = (
    (SymbolCategory.PRIMARY, "primary_df", "summary_df.h5"),
    (SymbolCategory.ORTHOLOG, "ortholog_df", "ortholog_match_subset_genes_df.h5"),
    (
        SymbolCategory.CLONE_NAME,
        "flj_clone_df",
        "flj_clone_symbol_match_subset_genes_df.h5",
    ),
    (SymbolCategory.PHENOTYPE, "phenotype_df", "phenotype_match_subset_genes_df.h5"),
    (
        SymbolCategory.GENE_GROUP,
        "hgnc_gene_group_df",
        "hgnc_gene_group_match_subset_genes_df.h5",
    ),
    (SymbolCategory.GENE_IDENTIFIER, "gene_id_df", "gene_id_match_subset_genes_df.h5"),
    (
        SymbolCategory.WITHDRAWN_ORTHOLOG,
        "mgi_withdrawn_df",
        "mgi_withdrawn_symbol_match_subset_genes_df.h5",
    ),
    (
        SymbolCategory.RELATED_GENE,
        "ncbi_related_gene_df",
        "ncbi_related_gene_match_subset_genes_df.h5",
    ),
    (
        SymbolCategory.GENE_INTERACTION,
        "ncbi_gene_interaction_df",
        "ncbi_gene_interaction_symbol_match_subset_genes_df.h5",
    ),
    (
        SymbolCategory.GENE_NEIGHBOR,
        "ncbi_gene_neighbor_df",
        "ncbi_gene_neighbor_match_subset_genes_df.h5",
    ),
    (
        SymbolCategory.PLACEHOLDER,
        "placeholder_df",
        "placeholder_match_subset_genes_df.h5",
    ),
    (
        SymbolCategory.PREVIOUS,
        "previous_df",
        "previous_symbol_match_subset_genes_df.h5",
    ),
    (
        SymbolCategory.PROTEIN_MASS,
        "protein_mass_df",
        "protein_mass_match_subset_genes_df.h5",
    ),
    (
        SymbolCategory.ALTERNATE_ABBREVIATION,
        "alternate_abbreviation_df",
        "alt_abbrev_symbol_match_subset_genes_df.h5",
    ),
)


def _assert_loaded_categories(result: GeneJar) -> None:
    """Check that each category loads its distinct expected archive contents.

    :param result: GeneJar instance loaded from the fixture archive.
    :return: None.
    :raises AssertionError: If a category is missing or contains incorrect data.
    """
    assert set(result.dfs) == {category for category, _, _ in EXPECTED_DATA_FILES}
    for category, key, _ in EXPECTED_DATA_FILES:
        assert result.dfs[category]["fixture_key"].tolist() == [key]


class FakeResponse:
    """Minimal requests response supporting JSON and streamed downloads."""

    def __init__(self, *, payload: bytes = b"", metadata: dict | None = None):
        self.payload = payload
        self.metadata = metadata
        self.headers = (
            {"content-length": str(len(payload))} if payload is not None else {}
        )

    def raise_for_status(self) -> None:
        """Simulate a successful HTTP response."""

    def json(self) -> dict:
        assert self.metadata is not None
        return self.metadata

    def iter_content(self, chunk_size: int):
        for start in range(0, len(self.payload), chunk_size):
            yield self.payload[start : start + chunk_size]

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def _release_zip(tmp_path: Path, *, omit_file: str | None = None) -> bytes:
    """Build a Zenodo-style ZIP containing nested annotation datasets."""
    data_dir = tmp_path / "fixture" / "relationship_annotations"
    data_dir.mkdir(parents=True, exist_ok=True)

    inner_buffer = BytesIO()
    with ZipFile(inner_buffer, "w", compression=ZIP_DEFLATED) as inner:
        for _, key, filename in EXPECTED_DATA_FILES:
            if filename == omit_file:
                continue

            file_path = data_dir / filename
            pd.DataFrame({"fixture_key": [key]}).to_hdf(file_path, key="df")
            inner.write(
                file_path,
                arcname=f"relationship_annotations/{filename}",
            )

    outer_buffer = BytesIO()
    with ZipFile(outer_buffer, "w", compression=ZIP_DEFLATED) as outer:
        outer.writestr(
            "output/relationship_annotations.zip",
            inner_buffer.getvalue(),
        )

        # Include unrelated analysis outputs from the published archive.
        outer.writestr("output/summary_df.csv", b"unrelated")
        outer.writestr("output/ortholog_analysis_df.csv", b"unrelated")
        outer.writestr("output/subset_genes_df.h5", b"unrelated")

    return outer_buffer.getvalue()


def _mock_zenodo(
    monkeypatch,
    payload: bytes,
    *,
    latest_id: str = "23965261",
    concept_id: str = "123456",
    latest_concept_id: str | None = None,
    selected_id: str | None = None,
    selected_concept_id: str | None = None,
    returned_selected_id: str | None = None,
):
    """Mock Zenodo latest and record-ID selection without metadata.version."""
    load_url = f"{datasets.ZENODO_API_BASE_URL}{datasets.ZENODO_RECORD_ID}"
    checksum = hashlib.md5(payload, usedforsecurity=False).hexdigest()
    calls = {"metadata": [], "downloads": []}

    def record(record_id: str, record_concept_id: str) -> dict:
        return {
            "id": record_id,
            "conceptrecid": record_concept_id,
            "status": "published",
            "metadata": {"title": "Gene JAR Resources Data"},
            "files": [
                {
                    "key": "output.zip",
                    "checksum": f"md5:{checksum}",
                    "links": {
                        "self": (
                            f"{datasets.ZENODO_API_BASE_URL}{record_id}/files/output.zip/content"
                        )
                    },
                }
            ],
        }

    def fake_get(url, **kwargs):
        if url == load_url:
            calls["metadata"].append(url)
            metadata = (
                record(selected_id, selected_concept_id or concept_id)
                if selected_id == datasets.ZENODO_RECORD_ID
                else {"id": datasets.ZENODO_RECORD_ID, "conceptrecid": concept_id}
            )
            return FakeResponse(metadata=metadata)
        if url == f"{load_url}/versions/latest":
            calls["metadata"].append(url)
            return FakeResponse(
                metadata=record(
                    latest_id,
                    latest_concept_id if latest_concept_id is not None else concept_id,
                )
            )
        if selected_id is not None and url == (
            f"{datasets.ZENODO_API_BASE_URL}{selected_id}"
        ):
            calls["metadata"].append(url)
            return FakeResponse(
                metadata=record(
                    returned_selected_id or selected_id,
                    selected_concept_id or concept_id,
                )
            )
        if url.endswith("/files/output.zip/content"):
            assert kwargs.get("stream") is True
            calls["downloads"].append(url)
            return FakeResponse(payload=payload)
        msg = f"Unexpected request: {url}"
        raise AssertionError(msg)

    monkeypatch.setattr(datasets.requests, "get", fake_get)
    return calls


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    """Keep the dataset cache out of the real home directory."""
    monkeypatch.setattr(datasets.Path, "home", staticmethod(lambda: tmp_path))
    return tmp_path


def test_download_datasets_loads_latest_release(tmp_path, monkeypatch, isolated_home):
    payload = _release_zip(tmp_path)
    calls = _mock_zenodo(monkeypatch, payload)

    result = GeneJar.download_datasets()

    assert isinstance(result, GeneJar)
    _assert_loaded_categories(result)
    assert len(calls["downloads"]) == 1
    extracted = isolated_home / ".cache" / "gene-jar" / "23965261" / "output"
    assert (extracted / "summary_df.h5").is_file()
    assert not (extracted / "subset_genes_df.h5").exists()
    assert not (extracted / "__pycache__").exists()


@pytest.mark.usefixtures("isolated_home")
def test_download_datasets_reuses_cached_datasets(tmp_path, monkeypatch):
    payload = _release_zip(tmp_path)
    calls = _mock_zenodo(monkeypatch, payload)

    first = GeneJar.download_datasets()
    second = GeneJar.download_datasets()

    assert isinstance(first, GeneJar)
    assert isinstance(second, GeneJar)
    assert len(calls["downloads"]) == 1
    _assert_loaded_categories(first)
    _assert_loaded_categories(second)


def test_download_datasets_recovers_from_corrupt_cache(
    tmp_path, monkeypatch, isolated_home
):
    payload = _release_zip(tmp_path)
    calls = _mock_zenodo(monkeypatch, payload)
    cache_dir = isolated_home / ".cache" / "gene-jar" / "23965261"
    cache_dir.mkdir(parents=True)
    (cache_dir / "output.zip").write_bytes(b"corrupted cached zip")

    result = GeneJar.download_datasets()

    assert isinstance(result, GeneJar)
    assert len(calls["downloads"]) == 1
    assert (cache_dir / "output.zip").read_bytes() == payload
    _assert_loaded_categories(result)


@pytest.mark.usefixtures("isolated_home")
def test_download_datasets_rejects_wrong_download_checksum(tmp_path, monkeypatch):
    payload = _release_zip(tmp_path)
    calls = _mock_zenodo(monkeypatch, payload)
    original_get = datasets.requests.get

    def fake_get(url, **kwargs):
        if url.endswith("/files/output.zip/content"):
            calls["downloads"].append(url)
            return FakeResponse(payload=b"not the expected ZIP")
        return original_get(url, **kwargs)

    monkeypatch.setattr(datasets.requests, "get", fake_get)

    with pytest.raises(ValueError, match="checksum"):
        GeneJar.download_datasets()

    cache_zip = Path.home() / ".cache" / "gene-jar" / "23965261" / "output.zip"
    assert not cache_zip.exists()


@pytest.mark.usefixtures("isolated_home")
def test_download_datasets_rejects_unrelated_latest_record(tmp_path, monkeypatch):
    payload = _release_zip(tmp_path)
    calls = _mock_zenodo(monkeypatch, payload, latest_concept_id="different")

    with pytest.raises(ValueError, match="different dataset"):
        GeneJar.download_datasets()

    assert calls["downloads"] == []


def test_download_datasets_uses_new_record_cache(tmp_path, monkeypatch, isolated_home):
    payload = _release_zip(tmp_path)
    first_calls = _mock_zenodo(monkeypatch, payload, latest_id="23965261")
    first = GeneJar.download_datasets()

    second_calls = _mock_zenodo(monkeypatch, payload, latest_id="23965262")
    second = GeneJar.download_datasets()

    assert isinstance(first, GeneJar)
    assert isinstance(second, GeneJar)
    _assert_loaded_categories(first)
    _assert_loaded_categories(second)
    assert len(first_calls["downloads"]) == 1
    assert len(second_calls["downloads"]) == 1
    assert (isolated_home / ".cache" / "gene-jar" / "23965261" / "output.zip").exists()
    assert (isolated_home / ".cache" / "gene-jar" / "23965262" / "output.zip").exists()


def test_download_datasets_loads_selected_version(tmp_path, monkeypatch, isolated_home):
    """Select a fixed Zenodo record ID even without metadata.version."""
    payload = _release_zip(tmp_path)
    selected_id = "23965000"
    calls = _mock_zenodo(monkeypatch, payload, selected_id=selected_id)

    result = GeneJar.download_datasets(version=selected_id)

    assert isinstance(result, GeneJar)
    _assert_loaded_categories(result)
    assert len(calls["downloads"]) == 1
    assert f"/{selected_id}/" in calls["downloads"][0]
    assert (isolated_home / ".cache" / "gene-jar" / selected_id / "output.zip").exists()
    assert f"{datasets.ZENODO_API_BASE_URL}{selected_id}" in calls["metadata"]
    assert not any(url.endswith("/versions/latest") for url in calls["metadata"])


@pytest.mark.usefixtures("isolated_home")
def test_download_datasets_rejects_non_numeric_version(tmp_path, monkeypatch):
    """Reject semantic version strings when Zenodo record IDs are required."""
    payload = _release_zip(tmp_path)
    calls = _mock_zenodo(monkeypatch, payload)

    with pytest.raises(ValueError, match="numeric record ID"):
        GeneJar.download_datasets(version="1.0.0")

    assert calls["downloads"] == []


@pytest.mark.usefixtures("isolated_home")
def test_download_datasets_rejects_unrelated_selected_record(tmp_path, monkeypatch):
    """Reject record IDs from another Zenodo concept."""
    payload = _release_zip(tmp_path)
    calls = _mock_zenodo(
        monkeypatch, payload, selected_id="23965000", selected_concept_id="other"
    )

    with pytest.raises(ValueError, match="different dataset"):
        GeneJar.download_datasets(version="23965000")

    assert calls["downloads"] == []


@pytest.mark.usefixtures("isolated_home")
def test_download_datasets_rejects_mismatched_selected_id(tmp_path, monkeypatch):
    """Ensure a requested record ID matches the returned metadata."""
    payload = _release_zip(tmp_path)
    calls = _mock_zenodo(
        monkeypatch, payload, selected_id="23965000", returned_selected_id="23965001"
    )

    with pytest.raises(ValueError, match="does not match requested ID"):
        GeneJar.download_datasets(version="23965000")

    assert calls["downloads"] == []


@pytest.mark.usefixtures("isolated_home")
def test_download_datasets_rejects_missing_nested_hdf5(tmp_path, monkeypatch):
    payload = _release_zip(tmp_path, omit_file="summary_df.h5")
    _mock_zenodo(monkeypatch, payload)

    with pytest.raises(KeyError, match=r"summary_df\.h5"):
        GeneJar.download_datasets()
