from __future__ import annotations

# ruff: noqa: E402, I001

import sys
import json
import os
import socket
import threading
from pathlib import Path

import pytest
import torch
from typer.testing import CliRunner

REPO_ROOT = Path(__file__).resolve().parents[3]
CORE_SRC = REPO_ROOT / "packages" / "ltx-core" / "src"
TRAINER_SRC = REPO_ROOT / "packages" / "ltx-trainer" / "src"
TRAINER_SCRIPTS = REPO_ROOT / "packages" / "ltx-trainer" / "scripts"
sys.path.insert(0, str(CORE_SRC))
sys.path.insert(0, str(TRAINER_SRC))
sys.path.insert(0, str(TRAINER_SCRIPTS))

import process_dataset
import train as train_script
from ltx_trainer.model_pool import ModelCacheKey, WarmModelPool
from ltx_trainer.trainer import LtxvTrainer
from ltx_trainer.warm_client import SOCKET_ENV, submit_if_running


class _TrackingModule(torch.nn.Linear):
    def __init__(self) -> None:
        super().__init__(1, 1)
        self.moves: list[torch.device] = []

    def to(self, device: torch.device | str, *args: object, **kwargs: object) -> _TrackingModule:
        self.moves.append(torch.device(device))
        return super().to(device, *args, **kwargs)


def test_warm_model_pool_loads_once_and_reuses_instance(tmp_path: Path) -> None:
    source = tmp_path / "model.safetensors"
    source.write_bytes(b"weights")
    key = ModelCacheKey.create("video_vae_encoder", source, torch.bfloat16)
    loads = 0

    def loader(device: torch.device) -> _TrackingModule:
        nonlocal loads
        loads += 1
        return _TrackingModule().to(device)

    pool = WarmModelPool()
    first = pool.get_or_load(key, loader, "cpu")
    second = pool.get_or_load(key, loader, "cpu")

    assert first is second
    assert loads == 1
    assert pool.size == 1

    pool.offload_all()
    assert first.weight.device.type == "cpu"


def test_preprocess_dataset_propagates_warm_pool(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.json"
    dataset.write_text("[]", encoding="utf-8")
    model = tmp_path / "model.safetensors"
    model.write_bytes(b"model")
    text_encoder = tmp_path / "gemma.safetensors"
    text_encoder.write_bytes(b"gemma")
    calls: list[tuple[str, WarmModelPool | None]] = []

    monkeypatch.setattr(
        process_dataset,
        "compute_captions_embeddings",
        lambda **kwargs: calls.append(("captions", kwargs.get("model_pool"))),
    )
    monkeypatch.setattr(
        process_dataset,
        "compute_latents",
        lambda **kwargs: calls.append(("latents", kwargs.get("model_pool"))),
    )

    pool = WarmModelPool()
    process_dataset.preprocess_dataset(
        dataset_file=str(dataset),
        caption_column="caption",
        video_column="media_path",
        resolution_buckets=[(9, 64, 64)],
        batch_size=1,
        output_dir=str(tmp_path / "out"),
        lora_trigger=None,
        vae_tiling=False,
        decode=False,
        model_path=str(model),
        text_encoder_path=str(text_encoder),
        device="cpu",
        model_pool=pool,
    )

    assert calls == [("captions", pool), ("latents", pool)]


def test_release_warm_models_removes_adapters_without_merging(tmp_path: Path) -> None:
    source = tmp_path / "model.safetensors"
    source.write_bytes(b"weights")
    key = ModelCacheKey.create("training_transformer", source, torch.bfloat16, quantization="none")
    base = _TrackingModule()
    base.weight.requires_grad_(True)
    checkpointing: list[bool] = []
    base.set_gradient_checkpointing = checkpointing.append

    class FakeTuner:
        unload_calls = 0

        def unload(self) -> _TrackingModule:
            self.unload_calls += 1
            return base

    class FakePeftModel:
        base_model = FakeTuner()

    class FakeAccelerator:
        freed = False

        @staticmethod
        def unwrap_model(model: object, **_kwargs: object) -> object:
            return model

        def free_memory(self, *_objects: object) -> None:
            self.freed = True

    pool = WarmModelPool()
    pool.replace(key, base)
    trainer = object.__new__(LtxvTrainer)
    trainer._released = False
    trainer._model_pool = pool
    trainer._warm_transformer_key = key
    trainer._training_ended = True
    trainer._transformer = FakePeftModel()
    trainer._accelerator = FakeAccelerator()

    trainer.release_warm_models()

    cached = pool.get_or_load(key, lambda _device: None, "cpu", move_cached=False)
    assert cached is base
    assert trainer._transformer is base
    assert FakePeftModel.base_model.unload_calls == 1
    assert trainer._accelerator.freed
    assert not base.weight.requires_grad
    assert checkpointing == [False]


def test_warm_client_submits_to_running_server(monkeypatch: pytest.MonkeyPatch) -> None:
    socket_path = Path("/tmp") / f"ltx-warm-test-{os.getpid()}.sock"
    socket_path.unlink(missing_ok=True)
    monkeypatch.setenv(SOCKET_ENV, str(socket_path))
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(socket_path))
    listener.listen(1)
    received: list[dict] = []

    def serve_one() -> None:
        connection, _ = listener.accept()
        with connection, connection.makefile("r", encoding="utf-8") as request_file:
            received.append(json.loads(request_file.readline()))
            connection.sendall(b'{"status":"accepted"}\n{"status":"ok","result":{}}\n')
        listener.close()

    thread = threading.Thread(target=serve_one)
    thread.start()
    assert submit_if_running("preprocess", {"dataset_file": "dataset.json"})
    thread.join(timeout=2)
    socket_path.unlink(missing_ok=True)

    assert received[0]["command"] == "preprocess"
    assert received[0]["args"] == {"dataset_file": "dataset.json"}


def test_train_cli_uses_warm_server_without_changing_command(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("not parsed by client: true", encoding="utf-8")
    submitted: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        train_script,
        "submit_if_running",
        lambda command, args: submitted.append((command, args)) or True,
    )

    result = CliRunner().invoke(train_script.app, [str(config_path)])

    assert result.exit_code == 0
    assert submitted[0][0] == "train"
    assert submitted[0][1]["config_path"] == str(config_path.resolve())


def test_process_dataset_cli_uses_warm_server_without_changing_command(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset.json"
    dataset.write_text("[]", encoding="utf-8")
    model = tmp_path / "model.safetensors"
    model.write_bytes(b"model")
    text_encoder = tmp_path / "gemma.safetensors"
    text_encoder.write_bytes(b"gemma")
    submitted: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        process_dataset,
        "submit_if_running",
        lambda command, args: submitted.append((command, args)) or True,
    )

    result = CliRunner().invoke(
        process_dataset.app,
        [
            str(dataset),
            "--resolution-buckets",
            "64x64x9",
            "--model-path",
            str(model),
            "--text-encoder-path",
            str(text_encoder),
            "--device",
            "cpu",
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )

    assert result.exit_code == 0
    assert submitted[0][0] == "preprocess"
    assert submitted[0][1]["resolution_buckets"] == [(9, 64, 64)]
