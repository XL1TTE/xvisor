import zipfile
from pathlib import Path
from PIL import Image

import pytest

from data.sources import LocalFile, LocalDirectory
from data.media import (
    classify_media,
    resolve_frame_provider,
    ImageDirectoryMedia,
    ArchiveMedia,
    FrameProvider,
)


def _create_dummy_image(path: Path) -> None:
    img = Image.new("RGB", (32, 32), color="red")
    img.save(path)


def test_classify_and_extract_directory(tmp_path: Path) -> None:
    # Create two dummy images
    img1 = tmp_path / "frame_01.png"
    img2 = tmp_path / "frame_02.png"
    _create_dummy_image(img1)
    _create_dummy_image(img2)

    media = classify_media(LocalDirectory(path=tmp_path))
    assert isinstance(media, ImageDirectoryMedia)
    assert len(media.image_paths) == 2

    provider = resolve_frame_provider(media)
    assert isinstance(provider, FrameProvider)
    assert provider.total_frames == 2

    frame0 = provider.get_frame(0)
    assert isinstance(frame0, Image.Image)
    assert frame0.size == (32, 32)


def test_classify_empty_directory_raises_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="contains no supported image frames"):
        classify_media(LocalDirectory(path=tmp_path))


def test_classify_and_extract_archive(tmp_path: Path) -> None:
    img_path = tmp_path / "sample.jpg"
    _create_dummy_image(img_path)

    zip_path = tmp_path / "frames.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(img_path, arcname="frame_001.jpg")

    media = classify_media(LocalFile(path=zip_path))
    assert isinstance(media, ArchiveMedia)

    provider = resolve_frame_provider(media)
    assert provider.total_frames == 1

    frame = provider.get_frame(0)
    assert isinstance(frame, Image.Image)
    assert frame.size == (32, 32)


def test_frame_provider_out_of_bounds_raises_error(tmp_path: Path) -> None:
    img_path = tmp_path / "single.png"
    _create_dummy_image(img_path)

    zip_path = tmp_path / "single.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(img_path, arcname="single.png")

    media = classify_media(LocalFile(path=zip_path))
    provider = resolve_frame_provider(media)

    with pytest.raises(IndexError, match="out of range"):
        provider.get_frame(99)
