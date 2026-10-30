"""Tests for downloading and loading Gene JAR dataset releases."""

from __future__ import annotations

import struct
import zlib
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd
import pytest

from gene_jar import GeneJar, SymbolCategory, datasets


def _release_zip(tmp_path: Path, version: str) -> bytes:
    """Build a small local release ZIP with all expected HDF files."""
    release_dir = tmp_path / f"gene-jar-data-{version}"
    release_dir.mkdir()

    for key, filename in datasets.DATA_FILES:
        pd.DataFrame({"fixture_key": [key]}).to_hdf(
            release_dir / filename,
            key="df",
        )

    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        for _, filename in datasets.DATA_FILES:
            archive.write(
                release_dir / filename,
                arcname=f"{release_dir.name}/{filename}",
            )
    return buffer.getvalue()


def _corrupt_first_deflate_member(payload: bytes, version: str) -> bytes:
    """Corrupt a member's DEFLATE stream while preserving the ZIP directory."""
    data = bytearray(payload)
    member_name = f"gene-jar-data-{version}/{datasets.DATA_FILES[0][1]}"
    with ZipFile(BytesIO(payload)) as archive:
        info = archive.getinfo(member_name)

    name_len, extra_len = struct.unpack_from("<HH", data, info.header_offset + 26)
    data_offset = info.header_offset + 30 + name_len + extra_len
    data[data_offset] = 0x06  # Reserved DEFLATE block type.
    return bytes(data)


def _use_test_home(monkeypatch, home: Path) -> None:
    monkeypatch.setattr(datasets.Path, "home", staticmethod(lambda: home))


def test_download_datasets_loads_release(tmp_path, monkeypatch):
    version = "0.1.0"
    payload = _release_zip(tmp_path, version)
    _use_test_home(monkeypatch, tmp_path)
    calls = []

    def fake_urlretrieve(url, filename):
        calls.append(url)
        Path(filename).write_bytes(payload)
        return str(filename), None

    monkeypatch.setattr(datasets, "urlretrieve", fake_urlretrieve)

    result = GeneJar.download_datasets(version=version)

    assert isinstance(result, GeneJar)
    assert result.dfs[SymbolCategory.PRIMARY].loc[0, "fixture_key"] == "primary_df"
    assert calls == [
        "https://github.com/cancervariants/gene-jar/releases/download/"
        "0.1.0/gene-jar-data-0.1.0.zip"
    ]


def test_download_datasets_reuses_cached_datasets(tmp_path, monkeypatch):
    version = "0.1.0"
    payload = _release_zip(tmp_path, version)
    _use_test_home(monkeypatch, tmp_path)
    calls = []

    def fake_urlretrieve(_url, filename):
        calls.append(filename)
        Path(filename).write_bytes(payload)
        return str(filename), None

    monkeypatch.setattr(datasets, "urlretrieve", fake_urlretrieve)

    first = GeneJar.download_datasets(version=version)
    second = GeneJar.download_datasets(version=version)

    assert isinstance(first, GeneJar)
    assert isinstance(second, GeneJar)
    assert len(calls) == 1
    assert second.dfs[SymbolCategory.PRIMARY].loc[0, "fixture_key"] == "primary_df"


def test_download_datasets_recovers_from_corrupt_deflate_cache(tmp_path, monkeypatch):
    version = "0.1.0"
    payload = _release_zip(tmp_path, version)
    corrupt_payload = _corrupt_first_deflate_member(payload, version)
    _use_test_home(monkeypatch, tmp_path)

    # Confirm the directory is readable but validating the member raises zlib.error.
    with ZipFile(BytesIO(corrupt_payload)) as archive:
        assert archive.namelist()
        with pytest.raises(zlib.error):
            archive.testzip()

    cache_dir = tmp_path / ".cache" / "gene-jar" / version
    cached_zip = cache_dir.parent / f"gene-jar-data-{version}.zip"
    cached_zip.parent.mkdir(parents=True, exist_ok=True)
    cached_zip.write_bytes(corrupt_payload)
    calls = []

    def fake_urlretrieve(_url, filename):
        calls.append(filename)
        Path(filename).write_bytes(payload)
        return str(filename), None

    monkeypatch.setattr(datasets, "urlretrieve", fake_urlretrieve)

    result = GeneJar.download_datasets(version=version)

    assert isinstance(result, GeneJar)
    assert len(calls) == 1
    assert result.dfs[SymbolCategory.PRIMARY].loc[0, "fixture_key"] == "primary_df"
