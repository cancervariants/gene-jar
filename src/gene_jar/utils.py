"""Configure useful utilities."""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlretrieve
from zipfile import BadZipFile, ZipFile


def initialize_logs(log_level: int = logging.INFO) -> None:
    """Configure logging.

    :param log_level: app log level to set
    """
    root = logging.getLogger()
    if root.handlers:
        return

    root.setLevel(log_level)
    formatter = logging.Formatter(
        "[%(asctime)s] - %(name)s - %(levelname)s : %(message)s"
    )
    fh = RotatingFileHandler(f"{__package__}.log", maxBytes=5_000_000, backupCount=3)
    fh.setFormatter(formatter)
    root.addHandler(fh)


class InvalidDownloadedZipError(BadZipFile):
    """Raised when a downloaded ZIP archive fails validation."""

    def __init__(self) -> None:
        """Initialize the exception with a descriptive error message."""
        super().__init__("Downloaded ZIP archive is invalid or incomplete")


def ensure_zip_cached(url: str, cache_path: Path) -> Path:
    """Download and cache a validated ZIP archive.

    Reuse valid cached archives. Download to a temporary file
    and atomically replace invalid or missing cached archives.
    """
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    def is_valid_zip(path: Path) -> bool:
        try:
            with ZipFile(path) as archive:
                return archive.testzip() is None
        except (BadZipFile, OSError):
            return False

    if is_valid_zip(cache_path):
        return cache_path

    with NamedTemporaryFile(
        dir=cache_path.parent,
        suffix=".zip",
        delete=False,
    ) as temp:
        temp_path = Path(temp.name)

    try:
        urlretrieve(url, temp_path)  # noqa: S310

        if not is_valid_zip(temp_path):
            raise InvalidDownloadedZipError

        temp_path.replace(cache_path)
    finally:
        temp_path.unlink(missing_ok=True)

    return cache_path
