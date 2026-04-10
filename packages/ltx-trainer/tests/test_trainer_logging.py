from pathlib import Path
from types import SimpleNamespace

import pytest

import ltx_trainer.trainer as trainer_module
from ltx_trainer.trainer import LtxvTrainer


class FakeWandbRun:
    def __init__(self) -> None:
        self.logged: list[tuple[dict, int | None]] = []

    def log(self, payload: dict, step: int | None = None) -> None:
        self.logged.append((payload, step))


class FakeTensorboardWriter:
    def __init__(self) -> None:
        self.scalars: list[tuple[str, float, int]] = []

    def add_scalar(self, name: str, value: float, global_step: int) -> None:
        self.scalars.append((name, value, global_step))


def make_logging_trainer() -> LtxvTrainer:
    trainer = object.__new__(LtxvTrainer)
    trainer._wandb_run = FakeWandbRun()
    trainer._tensorboard_writer = FakeTensorboardWriter()
    trainer._config = SimpleNamespace(wandb=SimpleNamespace(log_validation_videos=True))
    trainer._global_step = 12
    return trainer


def test_log_metrics_writes_to_wandb_and_tensorboard_with_same_step():
    trainer = make_logging_trainer()

    trainer._log_metrics(
        {
            "train/loss": 1.25,
            "train/learning_rate": 2e-4,
        },
        step=12,
    )

    assert trainer._wandb_run.logged == [
        (
            {
                "train/loss": 1.25,
                "train/learning_rate": 2e-4,
            },
            12,
        )
    ]
    assert trainer._tensorboard_writer.scalars == [
        ("train/loss", 1.25, 12),
        ("train/learning_rate", 2e-4, 12),
    ]


def test_validation_samples_are_logged_to_wandb_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    trainer = make_logging_trainer()
    sample_path = tmp_path / "sample.mp4"
    sample_path.touch()

    monkeypatch.setattr(
        trainer_module.wandb,
        "Video",
        lambda path, caption, format: (path, caption, format),
    )

    trainer._log_validation_samples([sample_path], ["demo prompt"])

    assert trainer._wandb_run.logged == [
        (
            {
                "validation_samples": [(str(sample_path), "demo prompt", "mp4")],
            },
            12,
        )
    ]
    assert trainer._tensorboard_writer.scalars == []
