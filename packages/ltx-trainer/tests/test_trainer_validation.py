import random
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import ltx_trainer.trainer as trainer_module
from ltx_trainer.trainer import LtxvTrainer


class FakeAccelerator:
    def __init__(
        self,
        reduced_stats: torch.Tensor | None = None,
        num_processes: int = 1,
        process_index: int = 0,
    ) -> None:
        self.device = torch.device("cpu")
        self.reduced_stats = reduced_stats
        self.reduction_calls: list[tuple[torch.Tensor, str]] = []
        self.num_processes = num_processes
        self.process_index = process_index

    def prepare(self, *objects):
        if len(objects) == 1:
            return objects[0]
        return objects

    def reduce(self, stats: torch.Tensor, reduction: str = "sum") -> torch.Tensor:
        self.reduction_calls.append((stats.clone(), reduction))
        if self.reduced_stats is not None:
            return self.reduced_stats.to(device=stats.device, dtype=stats.dtype)
        return stats


class FakeTransform:
    def __init__(self) -> None:
        self.training = True

    def eval(self) -> None:
        self.training = False

    def train(self) -> None:
        self.training = True


class FakeDataset:
    def __init__(self, root: str, data_sources: dict[str, str]) -> None:
        self.root = root
        self.data_sources = data_sources

    def __len__(self) -> int:
        return 3


class FakeDataLoader:
    def __init__(self, dataset, **kwargs) -> None:
        self.dataset = dataset
        self.kwargs = kwargs


def make_validation_trainer() -> LtxvTrainer:
    trainer = object.__new__(LtxvTrainer)
    trainer._config = SimpleNamespace(
        validation=SimpleNamespace(
            skip_initial_validation=False,
            interval=10,
            loss_interval=5,
            preprocessed_data_root="/tmp/validation",
            prompts=["prompt"],
        ),
        wandb=SimpleNamespace(log_validation_videos=True),
        data=SimpleNamespace(num_dataloader_workers=3),
        optimization=SimpleNamespace(batch_size=4),
    )
    trainer._training_strategy = SimpleNamespace(
        get_data_sources=lambda: {"latents": "latents", "conditions": "conditions"}
    )
    trainer._accelerator = FakeAccelerator()
    trainer._validation_dataset = None
    trainer._validation_dataloader = None
    trainer._transformer = FakeTransform()
    trainer._global_step = 0
    trainer._log_metrics = lambda metrics, step=None: None
    return trainer


def make_training_trainer() -> LtxvTrainer:
    trainer = object.__new__(LtxvTrainer)
    trainer._config = SimpleNamespace(
        data=SimpleNamespace(
            num_dataloader_workers=3,
            preprocessed_data_root="/tmp/training",
        ),
        optimization=SimpleNamespace(batch_size=4),
    )
    trainer._training_strategy = SimpleNamespace(
        get_data_sources=lambda: {"latents": "latents", "conditions": "conditions"}
    )
    trainer._accelerator = FakeAccelerator()
    trainer._dataset = None
    trainer._dataloader = None
    return trainer


def test_init_validation_dataloader_uses_non_shuffling_loader(
    monkeypatch: pytest.MonkeyPatch,
):
    trainer = make_validation_trainer()

    monkeypatch.setattr(trainer_module, "PrecomputedDataset", FakeDataset)
    monkeypatch.setattr(trainer_module, "DataLoader", FakeDataLoader)

    trainer._init_validation_dataloader()

    assert isinstance(trainer._validation_dataset, FakeDataset)
    assert trainer._validation_dataset.root == "/tmp/validation"
    assert trainer._validation_dataset.data_sources == {
        "latents": "latents",
        "conditions": "conditions",
    }
    assert trainer._validation_dataloader.kwargs["shuffle"] is False
    assert trainer._validation_dataloader.kwargs["drop_last"] is False
    assert trainer._validation_dataloader.kwargs["batch_size"] == 4
    assert trainer._validation_dataloader.kwargs["num_workers"] == 3
    assert trainer._validation_dataloader.kwargs.get("generator") is None


def test_init_training_dataloader_uses_fresh_shuffle_generator(
    monkeypatch: pytest.MonkeyPatch,
):
    trainer = make_training_trainer()

    monkeypatch.setattr(trainer_module, "PrecomputedDataset", FakeDataset)
    monkeypatch.setattr(trainer_module, "DataLoader", FakeDataLoader)

    seeds = iter([101, 202])
    monkeypatch.setattr(
        trainer_module.os,
        "urandom",
        lambda size: next(seeds).to_bytes(size, byteorder="big"),
    )

    trainer._init_dataloader()
    first_seed = trainer._dataloader.kwargs["generator"].initial_seed()

    trainer._init_dataloader()
    second_seed = trainer._dataloader.kwargs["generator"].initial_seed()

    assert trainer._dataset.root == "/tmp/training"
    assert trainer._dataset.data_sources == {
        "latents": "latents",
        "conditions": "conditions",
    }
    assert trainer._dataloader.kwargs["shuffle"] is True
    assert trainer._dataloader.kwargs["drop_last"] is True
    assert first_seed == 101
    assert second_seed == 202


def test_training_dataloader_uses_broadcast_shuffle_seed_in_multi_process(
    monkeypatch: pytest.MonkeyPatch,
):
    trainer = make_training_trainer()
    trainer._accelerator = FakeAccelerator(num_processes=2, process_index=1)

    monkeypatch.setattr(
        trainer_module.os,
        "urandom",
        lambda size: (111).to_bytes(size, byteorder="big"),
    )
    monkeypatch.setattr(
        trainer_module.torch.distributed,
        "is_available",
        lambda: True,
    )
    monkeypatch.setattr(
        trainer_module.torch.distributed,
        "is_initialized",
        lambda: True,
    )

    def fake_broadcast_object_list(obj_list, src: int) -> None:
        assert src == 0
        obj_list[0] = 999

    monkeypatch.setattr(
        trainer_module.torch.distributed,
        "broadcast_object_list",
        fake_broadcast_object_list,
    )

    generator = trainer._make_training_dataloader_generator()

    assert generator.initial_seed() == 999


def test_validation_loss_schedule_uses_own_interval():
    trainer = make_validation_trainer()
    trainer._validation_dataloader = object()

    trainer._global_step = 10
    assert trainer._should_run_validation_loss(is_optimization_step=True) is True
    assert trainer._should_run_validation_sampling(is_optimization_step=True) is True

    trainer._global_step = 5
    assert trainer._should_run_validation_loss(is_optimization_step=True) is True
    assert trainer._should_run_validation_sampling(is_optimization_step=True) is False

    trainer._global_step = 4
    assert trainer._should_run_validation_loss(is_optimization_step=True) is False
    assert trainer._should_run_validation_loss(is_optimization_step=False) is False


def test_initial_validations_run_loss_before_sampling(monkeypatch: pytest.MonkeyPatch):
    trainer = make_validation_trainer()
    trainer._validation_dataloader = object()
    call_order: list[str] = []
    sample_path = Path("/tmp/sample.mp4")

    trainer._run_validation_loss = lambda: call_order.append("loss")
    trainer._sample_videos = lambda progress: call_order.append("sample") or [
        sample_path
    ]
    trainer._log_validation_samples = lambda paths, prompts: call_order.append("media")

    monkeypatch.setattr(trainer_module, "IS_MAIN_PROCESS", True)

    sampled = trainer._run_initial_validations(progress=object())

    assert sampled == [sample_path]
    assert call_order == ["loss", "sample", "media"]


def test_run_validation_loss_preserves_rng(monkeypatch: pytest.MonkeyPatch):
    trainer = make_validation_trainer()
    trainer._validation_dataloader = [{"idx": [0, 1]}]
    trainer._accelerator = FakeAccelerator()
    trainer._transformer = FakeTransform()
    trainer._global_step = 7
    trainer._log_metrics = lambda metrics, step=None: None

    def consume_rng(_batch):
        random.random()
        torch.rand(1)
        return torch.tensor(0.5)

    trainer._compute_batch_loss = consume_rng
    monkeypatch.setattr(trainer_module, "IS_MAIN_PROCESS", False)

    random.seed(1234)
    torch.manual_seed(5678)
    expected_python = random.random()
    expected_torch = torch.rand(1).item()

    random.seed(1234)
    torch.manual_seed(5678)
    trainer._run_validation_loss()
    actual_python = random.random()
    actual_torch = torch.rand(1).item()

    assert actual_python == expected_python
    assert actual_torch == pytest.approx(expected_torch)


def test_run_validation_loss_uses_reduced_totals_and_logs_only_on_main_process(
    monkeypatch: pytest.MonkeyPatch,
):
    trainer = make_validation_trainer()
    trainer._validation_dataloader = [{"idx": [0, 1]}, {"idx": [2]}]
    trainer._accelerator = FakeAccelerator(
        reduced_stats=torch.tensor([9.0, 3.0], dtype=torch.float64)
    )
    trainer._transformer = FakeTransform()
    trainer._global_step = 15
    logged: list[tuple[dict[str, float], int | None]] = []
    trainer._log_metrics = lambda metrics, step=None: logged.append((metrics, step))
    trainer._compute_batch_loss = lambda _batch: torch.tensor(1.0)

    monkeypatch.setattr(trainer_module, "IS_MAIN_PROCESS", False)
    validation_loss = trainer._run_validation_loss()
    assert validation_loss == pytest.approx(3.0)
    assert logged == []

    monkeypatch.setattr(trainer_module, "IS_MAIN_PROCESS", True)
    validation_loss = trainer._run_validation_loss()
    assert validation_loss == pytest.approx(3.0)
    assert logged == [({"validation/loss": 3.0}, 15)]
    assert trainer._accelerator.reduction_calls[-1][1] == "sum"
