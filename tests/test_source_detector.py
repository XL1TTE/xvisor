from pathlib import Path
import pytest

from data.sources import (
    SourceDetector,
    LocalFile,
    LocalDirectory,
)


def test_detect_local_directory(tmp_path: Path) -> None:
    detector = SourceDetector.default()
    result = detector.detect(str(tmp_path))

    assert isinstance(result, LocalDirectory)
    assert result.path == tmp_path.resolve()


def test_detect_local_file(tmp_path: Path) -> None:
    test_file = tmp_path / "video.mp4"
    test_file.touch()

    detector = SourceDetector.default()
    result = detector.detect(str(test_file))

    assert isinstance(result, LocalFile)
    assert result.path == test_file.resolve()


def test_detect_unknown_raises_value_error() -> None:
    detector = SourceDetector.default()

    with pytest.raises(ValueError, match="Unable to determine input source"):
        detector.detect("non_existent_folder_or_file_12345")
