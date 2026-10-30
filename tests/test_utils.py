"""Tests for reusable ZIP download and caching utilities."""

import struct
import zlib
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from gene_jar import utils


def _zip_bytes() -> bytes:
    """Build a small valid ZIP without external test data."""
    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr("example.txt", "sample data")
    return buffer.getvalue()


def test_ensure_zip_cached_downloads_missing_archive(tmp_path, monkeypatch):
    cache_path = tmp_path / "nested" / "dataset.zip"
    payload = _zip_bytes()
    calls = []

    def fake_urlretrieve(url, filename):
        calls.append(url)
        Path(filename).write_bytes(payload)
        return str(filename), None

    monkeypatch.setattr(utils, "urlretrieve", fake_urlretrieve)

    result = utils.ensure_zip_cached("https://example.org/dataset.zip", cache_path)

    assert result == cache_path
    assert cache_path.read_bytes() == payload
    assert calls == ["https://example.org/dataset.zip"]
    assert list(cache_path.parent.iterdir()) == [cache_path]


def test_ensure_zip_cached_reuses_valid_archive(tmp_path, monkeypatch):
    cache_path = tmp_path / "dataset.zip"
    payload = _zip_bytes()
    cache_path.write_bytes(payload)

    def unexpected_download(*_args, **_kwargs):
        pytest.fail("A valid cached ZIP must not be downloaded again")

    monkeypatch.setattr(utils, "urlretrieve", unexpected_download)

    assert (
        utils.ensure_zip_cached("https://example.org/dataset.zip", cache_path)
        == cache_path
    )
    assert cache_path.read_bytes() == payload


def test_ensure_zip_cached_replaces_corrupt_archive(tmp_path, monkeypatch):
    cache_path = tmp_path / "dataset.zip"
    cache_path.write_bytes(b"not a ZIP archive")
    payload = _zip_bytes()

    def fake_urlretrieve(_url, filename):
        Path(filename).write_bytes(payload)
        return str(filename), None

    monkeypatch.setattr(utils, "urlretrieve", fake_urlretrieve)

    assert (
        utils.ensure_zip_cached("https://example.org/dataset.zip", cache_path)
        == cache_path
    )
    assert cache_path.read_bytes() == payload
    with ZipFile(cache_path) as archive:
        assert archive.read("example.txt") == b"sample data"
    assert list(tmp_path.iterdir()) == [cache_path]


def test_ensure_zip_cached_cleans_up_after_failed_download(tmp_path, monkeypatch):
    cache_path = tmp_path / "dataset.zip"
    original = b"existing corrupt cache"
    cache_path.write_bytes(original)

    download_error = OSError("connection interrupted")

    def interrupted_download(_url, filename):
        Path(filename).write_bytes(b"partial ZIP content")
        raise download_error

    monkeypatch.setattr(utils, "urlretrieve", interrupted_download)

    with pytest.raises(OSError, match="connection interrupted"):
        utils.ensure_zip_cached("https://example.org/dataset.zip", cache_path)

    assert cache_path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [cache_path]


def test_ensure_zip_cached_replaces_archive_with_corrupt_deflate_data(
    tmp_path, monkeypatch
):
    cache_path = tmp_path / "dataset.zip"

    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("example.txt", "sample data " * 100)
    payload = buffer.getvalue()
    cache_path.write_bytes(payload)

    # Corrupt the DEFLATE stream while leaving the ZIP directory readable.
    data = bytearray(payload)
    with ZipFile(BytesIO(payload)) as archive:
        info = archive.getinfo("example.txt")
    name_len, extra_len = struct.unpack_from("<HH", data, info.header_offset + 26)
    data_offset = info.header_offset + 30 + name_len + extra_len
    data[data_offset] = 0x06  # Reserved DEFLATE block type.
    cache_path.write_bytes(data)

    with ZipFile(cache_path) as archive:
        assert archive.namelist() == ["example.txt"]
        with pytest.raises(zlib.error):
            archive.testzip()

    def fake_urlretrieve(_url, filename):
        Path(filename).write_bytes(payload)
        return str(filename), None

    monkeypatch.setattr(utils, "urlretrieve", fake_urlretrieve)

    assert (
        utils.ensure_zip_cached("https://example.org/dataset.zip", cache_path)
        == cache_path
    )
    with ZipFile(cache_path) as archive:
        assert archive.read("example.txt") == b"sample data " * 100
