from pathlib import Path

import pytest

from ltx_trainer.config import LtxTrainerConfig


def _base_config(tmp_path: Path) -> dict:
    model_path = tmp_path / "model.safetensors"
    model_path.write_bytes(b"checkpoint")
    text_encoder_path = tmp_path / "gemma"
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
        "data": {
            "preprocessed_data_root": str(tmp_path / "data"),
            "num_dataloader_workers": 0,
        },
    }


def test_nsync_config_accepts_full_training_mode(tmp_path: Path) -> None:
    config_data = _base_config(tmp_path)
    config_data["model"]["training_mode"] = "full"
    config_data.pop("lora")
    config_data["nsync"] = {"enabled": True}

    config = LtxTrainerConfig(**config_data)

    assert config.model.training_mode == "full"
    assert config.nsync.enabled is True


def test_nsync_requires_negative_audio_dir_when_audio_training(tmp_path: Path) -> None:
    config_data = _base_config(tmp_path)
    config_data["training_strategy"] = {
        "name": "text_to_video",
        "with_audio": True,
    }
    config_data["nsync"] = {
        "enabled": True,
        "negative_audio_latents_dir": "",
    }

    with pytest.raises(ValueError, match="negative_audio_latents_dir"):
        LtxTrainerConfig(**config_data)


def test_nsync_accepts_audio_training_when_negative_audio_dir_is_present(tmp_path: Path) -> None:
    config_data = _base_config(tmp_path)
    config_data["training_strategy"] = {
        "name": "text_to_video",
        "with_audio": True,
    }
    config_data["nsync"] = {
        "enabled": True,
        "negative_audio_latents_dir": "negative_audio_latents",
    }

    config = LtxTrainerConfig(**config_data)

    assert config.nsync.enabled is True
    assert config.nsync.negative_audio_latents_dir == "negative_audio_latents"
