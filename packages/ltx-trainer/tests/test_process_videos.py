import json
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError

import process_videos as process_videos_script


def test_media_dataset_treats_webp_inputs_as_images(tmp_path: Path, monkeypatch) -> None:
    image_path = tmp_path / "images" / "sample.webp"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"placeholder")

    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps([{"media_path": "images/sample.webp"}]), encoding="utf-8")

    def fail_if_video_checked(path: Path) -> int:
        raise AssertionError(f"image path was treated as video: {path}")

    def fail_if_video_preprocessed(self, path: Path) -> tuple[torch.Tensor, float]:
        raise AssertionError(f"image path was sent to video preprocessing: {path}")

    monkeypatch.setattr(
        process_videos_script,
        "get_video_frame_count",
        fail_if_video_checked,
    )
    monkeypatch.setattr(
        process_videos_script.MediaDataset,
        "_preprocess_image",
        lambda self, path: torch.zeros(3, 1, 32, 32),
    )
    monkeypatch.setattr(
        process_videos_script.MediaDataset,
        "_preprocess_video",
        fail_if_video_preprocessed,
    )

    dataset = process_videos_script.MediaDataset(
        dataset_file=dataset_path,
        main_media_column="media_path",
        video_column="media_path",
        resolution_buckets=[(1, 32, 32)],
    )

    item = dataset[0]

    assert len(dataset) == 1
    assert item["relative_path"] == "images/sample.webp"
    assert item["video"].shape == (3, 1, 32, 32)
    assert item["video_metadata"]["fps"] == 1.0


def test_media_dataset_skips_unidentified_images_during_validation(tmp_path: Path, monkeypatch) -> None:
    bad_image_path = tmp_path / "images" / "bad.png"
    good_image_path = tmp_path / "images" / "good.png"
    bad_image_path.parent.mkdir(parents=True, exist_ok=True)
    bad_image_path.write_bytes(b"not-an-image")
    good_image_path.write_bytes(b"placeholder")

    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        json.dumps(
            [
                {"media_path": "images/bad.png"},
                {"media_path": "images/good.png"},
            ]
        ),
        encoding="utf-8",
    )

    def fake_open_image_as_srgb(path: Path) -> Image.Image:
        if path == bad_image_path:
            raise UnidentifiedImageError(f"cannot identify image file {path}")
        return Image.new("RGB", (32, 32), color="white")

    monkeypatch.setattr(process_videos_script, "open_image_as_srgb", fake_open_image_as_srgb)

    dataset = process_videos_script.MediaDataset(
        dataset_file=dataset_path,
        main_media_column="media_path",
        video_column="media_path",
        resolution_buckets=[(1, 32, 32)],
    )

    item = dataset[0]

    assert len(dataset) == 1
    assert item["relative_path"] == "images/good.png"
    assert item["video"].shape == (3, 1, 32, 32)
    assert item["video_metadata"]["fps"] == 1.0
