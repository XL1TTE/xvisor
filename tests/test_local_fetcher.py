from pathlib import Path

from data.sources import LocalFile, LocalDirectory
from data.fetching import (
    LocalFetcher,
    LocalFetchSuccess,
    LocalFetchError,
)


def test_local_fetcher_existing_file(tmp_path: Path) -> None:
    test_file = tmp_path / "sample.mp4"
    test_file.touch()

    source = LocalFile(path=test_file)
    result = LocalFetcher.fetch(source)

    assert isinstance(result, LocalFetchSuccess)
    assert result.resolved == source


def test_local_fetcher_missing_file(tmp_path: Path) -> None:
    missing_file = tmp_path / "missing.mp4"

    source = LocalFile(path=missing_file)
    result = LocalFetcher.fetch(source)

    assert isinstance(result, LocalFetchError)
    assert "Path does not exist" in result.error_message


def test_local_fetcher_existing_directory(tmp_path: Path) -> None:
    source = LocalDirectory(path=tmp_path)
    result = LocalFetcher.fetch(source)

    assert isinstance(result, LocalFetchSuccess)
    assert result.resolved == source
