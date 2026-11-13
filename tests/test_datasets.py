"""Tests for downloading the latest or an explicit Zenodo record ID."""

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
    """Build a real-world nested output.zip, including irrelevant macOS metadata."""
    data_dir = tmp_path / "fixture" / "relationship_annotations"
    data_dir.mkdir(parents=True, exist_ok=True)

    inner_buffer = BytesIO()
    with ZipFile(inner_buffer, "w", compression=ZIP_DEFLATED) as inner:
        for key, filename in datasets.DATA_FILES:
            if filename == omit_file:
                continue
            file_path = data_dir / filename
            pd.DataFrame({"fixture_key": [key]}).to_hdf(file_path, key="df")
            inner.write(file_path, arcname=f"relationship_annotations/{filename}")
        inner.writestr("__MACOSX/relationship_annotations/._summary_df.h5", b"metadata")

    outer_buffer = BytesIO()
    with ZipFile(outer_buffer, "w", compression=ZIP_DEFLATED) as outer:
        outer.writestr("__MACOSX/._output", b"metadata")
        outer.writestr("output/relationship_annotations.zip", inner_buffer.getvalue())
        outer.writestr("output/unrelated_analysis.csv", b"unrelated")
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
    base_url = f"https://zenodo.org/api/records/{datasets.ZENODO_RECORD_ID}"
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
                            f"https://zenodo.org/api/records/{record_id}"
                            "/files/output.zip/content"
                        )
                    },
                }
            ],
        }

    def fake_get(url, **kwargs):
        if url == base_url:
            calls["metadata"].append(url)
            metadata = (
                record(selected_id, selected_concept_id or concept_id)
                if selected_id == datasets.ZENODO_RECORD_ID
                else {"id": datasets.ZENODO_RECORD_ID, "conceptrecid": concept_id}
            )
            return FakeResponse(metadata=metadata)
        if url == f"{base_url}/versions/latest":
            calls["metadata"].append(url)
            return FakeResponse(
                metadata=record(
                    latest_id,
                    latest_concept_id if latest_concept_id is not None else concept_id,
                )
            )
        if selected_id is not None and url == (
            f"https://zenodo.org/api/records/{selected_id}"
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
    assert len(calls["metadata"]) == 4
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


def test_download_datasets_loads_selected_version(tmp_path, monkeypatch, isolated_home):
    """Select a fixed Zenodo record ID even without metadata.version."""
    payload = _release_zip(tmp_path)
    selected_id = "23965000"
    calls = _mock_zenodo(monkeypatch, payload, selected_id=selected_id)

    result = GeneJar.download_datasets(version=selected_id)

    assert isinstance(result, GeneJar)
    assert len(calls["downloads"]) == 1
    assert f"/{selected_id}/" in calls["downloads"][0]
    assert (isolated_home / ".cache" / "gene-jar" / selected_id / "output.zip").exists()
    assert f"https://zenodo.org/api/records/{selected_id}" in calls["metadata"]
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
