"""Tests for reusable ZIP download and caching utilities."""

from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

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
