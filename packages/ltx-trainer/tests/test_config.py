from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from ltx_trainer.config import LtxTrainerConfig


def build_base_config(tmp_path: Path) -> dict:
    model_path = tmp_path / "model.safetensors"
    model_path.touch()

    train_data_root = tmp_path / "train-data"
    train_data_root.mkdir()

    text_encoder_path = tmp_path / "text-encoder"
    text_encoder_path.mkdir()

    return {
        "model": {
            "model_path": str(model_path),
            "text_encoder_path": str(text_encoder_path),
            "training_mode": "lora",
        },
        "lora": {
            "rank": 8,
            "alpha": 8,
            "dropout": 0.0,
            "target_modules": ["to_k"],
        },
        "training_strategy": {
            "name": "text_to_video",
        },
        "data": {
            "preprocessed_data_root": str(train_data_root),
        },
        "validation": {
            "interval": None,
            "preprocessed_data_root": None,
            "loss_interval": None,
        },
        "tensorboard": {
            "enabled": False,
            "log_dir": None,
        },
        "output_dir": str(tmp_path / "outputs"),
    }


@pytest.mark.parametrize(
    ("data_root", "loss_interval"),
    [
        ("set", None),
        (None, 25),
    ],
)
def test_validation_loss_config_requires_both_fields(
    tmp_path: Path, data_root: str | None, loss_interval: int | None
):
    config_data = build_base_config(tmp_path)
    validation_root = tmp_path / "val-data"
    validation_root.mkdir()

    config_data["validation"]["preprocessed_data_root"] = (
        str(validation_root) if data_root else None
    )
    config_data["validation"]["loss_interval"] = loss_interval

    with pytest.raises(
        ValueError,
        match="validation.preprocessed_data_root and validation.loss_interval",
    ):
        LtxTrainerConfig(**config_data)


def test_validation_paths_expand_and_tensorboard_defaults_from_output_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))

    model_path = fake_home / "model.safetensors"
    model_path.touch()
    text_encoder_path = fake_home / "text-encoder"
    text_encoder_path.mkdir()
    train_data_root = fake_home / "train-data"
    train_data_root.mkdir()
    validation_root = fake_home / "validation-data"
    validation_root.mkdir()

    config_data = {
        "model": {
            "model_path": str(model_path),
            "text_encoder_path": str(text_encoder_path),
            "training_mode": "lora",
        },
        "lora": {
            "rank": 8,
            "alpha": 8,
            "dropout": 0.0,
            "target_modules": ["to_k"],
        },
        "data": {
            "preprocessed_data_root": str(train_data_root),
        },
        "validation": {
            "interval": None,
            "preprocessed_data_root": "~/validation-data",
            "loss_interval": 40,
        },
        "tensorboard": {
            "enabled": True,
            "log_dir": None,
        },
        "output_dir": "~/runs/demo",
    }

    config = LtxTrainerConfig(**config_data)

    expected_output_dir = str((fake_home / "runs" / "demo").resolve())
    assert config.output_dir == expected_output_dir
    assert config.validation.preprocessed_data_root == str(validation_root.resolve())
    assert config.tensorboard.log_dir == str(
        (Path(expected_output_dir) / "tensorboard").resolve()
    )


def test_example_configs_still_parse(tmp_path: Path):
    example_paths = [
        Path("packages/ltx-trainer/configs/ltx2_av_lora.yaml"),
        Path("packages/ltx-trainer/configs/ltx2_av_lora_low_vram.yaml"),
        Path("packages/ltx-trainer/configs/ltx2_v2v_ic_lora.yaml"),
    ]

    for example_path in example_paths:
        config_data = yaml.safe_load(example_path.read_text())
        config_data = deepcopy(config_data)

        model_path = tmp_path / f"{example_path.stem}.safetensors"
        model_path.touch()
        text_encoder_path = tmp_path / f"{example_path.stem}-text-encoder"
        text_encoder_path.mkdir()
        train_data_root = tmp_path / f"{example_path.stem}-train-data"
        train_data_root.mkdir()

        config_data["model"]["model_path"] = str(model_path)
        config_data["model"]["text_encoder_path"] = str(text_encoder_path)
        config_data["data"]["preprocessed_data_root"] = str(train_data_root)
        config_data["output_dir"] = str(tmp_path / f"{example_path.stem}-outputs")
        config_data["validation"]["preprocessed_data_root"] = None
        config_data["validation"]["loss_interval"] = None
        config_data["validation"]["images"] = None
        config_data["validation"]["reference_videos"] = None

        if config_data["training_strategy"]["name"] == "video_to_video":
            config_data["validation"]["interval"] = None

        parsed = LtxTrainerConfig(**config_data)
        assert parsed.output_dir.endswith(f"{example_path.stem}-outputs")
