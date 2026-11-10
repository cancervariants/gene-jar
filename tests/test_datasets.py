"""Tests for downloading and loading the latest Zenodo Gene JAR datasets."""

from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd
import pytest

from gene_jar import GeneJar, SymbolCategory, datasets


class FakeResponse:
    """Minimal requests response supporting JSON and streamed downloads."""

    def __init__(self, *, payload: bytes = b"", metadata: dict | None = None):
        self.payload = payload
        self.metadata = metadata

    def raise_for_status(self) -> None:
        pass

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


def _release_zip(tmp_path: Path) -> bytes:
    """Build output.zip containing every required HDF5 file."""
    output_dir = tmp_path / "fixture" / "output"
    output_dir.mkdir(parents=True)
    for key, filename in datasets.DATA_FILES:
        pd.DataFrame({"fixture_key": [key]}).to_hdf(output_dir / filename, key="df")

    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        for _, filename in datasets.DATA_FILES:
            archive.write(output_dir / filename, arcname=f"output/{filename}")
    return buffer.getvalue()


def _mock_zenodo(
    monkeypatch,
    payload: bytes,
    *,
    latest_id: str = "23965261",
    concept_id: str = "123456",
    latest_concept_id: str | None = None,
):
    """Mock Zenodo metadata and streamed ZIP downloads."""
    base_url = f"https://zenodo.org/api/records/{datasets.ZENODO_RECORD_ID}"
    download_url = (
        f"https://zenodo.org/api/records/{latest_id}/files/output.zip/content"
    )
    checksum = hashlib.md5(payload, usedforsecurity=False).hexdigest()
    calls = {"metadata": [], "downloads": []}

    def fake_get(url, **kwargs):
        if url == base_url:
            calls["metadata"].append(url)
            return FakeResponse(
                metadata={"id": datasets.ZENODO_RECORD_ID, "conceptrecid": concept_id}
            )
        if url == f"{base_url}/versions/latest":
            calls["metadata"].append(url)
            return FakeResponse(
                metadata={
                    "id": latest_id,
                    "conceptrecid": latest_concept_id
                    if latest_concept_id is not None
                    else concept_id,
                    "status": "published",
                    "files": [
                        {
                            "key": "output.zip",
                            "checksum": f"md5:{checksum}",
                            "links": {"self": download_url},
                        }
                    ],
                }
            )
        if url == download_url:
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
    assert result.dfs[SymbolCategory.PRIMARY].loc[0, "fixture_key"] == "primary_df"
    assert len(calls["downloads"]) == 1
    assert (
        isolated_home / ".cache" / "gene-jar" / "23965261" / "output" / "summary_df.h5"
    ).is_file()


@pytest.mark.usefixtures("isolated_home")
def test_download_datasets_reuses_cached_datasets(tmp_path, monkeypatch):
    payload = _release_zip(tmp_path)
    calls = _mock_zenodo(monkeypatch, payload)

    first = GeneJar.download_datasets()
    second = GeneJar.download_datasets()

    assert isinstance(first, GeneJar)
    assert isinstance(second, GeneJar)
    assert len(calls["downloads"]) == 1
    assert len(calls["metadata"]) == 4  # Latest record is checked each time.
    assert second.dfs[SymbolCategory.PRIMARY].loc[0, "fixture_key"] == "primary_df"


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
    assert result.dfs[SymbolCategory.PRIMARY].loc[0, "fixture_key"] == "primary_df"


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
    assert len(first_calls["downloads"]) == 1
    assert len(second_calls["downloads"]) == 1
    assert (isolated_home / ".cache" / "gene-jar" / "23965261" / "output.zip").exists()
    assert (isolated_home / ".cache" / "gene-jar" / "23965262" / "output.zip").exists()
