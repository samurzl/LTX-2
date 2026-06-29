from __future__ import annotations

# ruff: noqa: E402, I001

import sys
import json
import os
import socket
import threading
from io import StringIO
from pathlib import Path

import pytest
import torch
from rich.console import Console
from typer.testing import CliRunner

REPO_ROOT = Path(__file__).resolve().parents[3]
CORE_SRC = REPO_ROOT / "packages" / "ltx-core" / "src"
TRAINER_SRC = REPO_ROOT / "packages" / "ltx-trainer" / "src"
TRAINER_SCRIPTS = REPO_ROOT / "packages" / "ltx-trainer" / "scripts"
sys.path.insert(0, str(CORE_SRC))
sys.path.insert(0, str(TRAINER_SRC))
sys.path.insert(0, str(TRAINER_SCRIPTS))

import process_dataset
import process_captions
import process_videos
import train as train_script
import ltx_trainer.trainer as trainer_module
from ltx_core.text_encoders.gemma.encoders.base_encoder import GemmaTextEncoder
from ltx_core.text_encoders.gemma.tokenizer import LTXVGemmaTokenizer
from ltx_trainer.config import LtxTrainerConfig
import warm_server
from ltx_trainer.model_pool import ModelCacheKey, ModelStatus, WarmModelPool
from ltx_trainer.trainer import LtxvTrainer
from ltx_trainer.warm_client import SOCKET_ENV, submit_if_running
from ltx_trainer.warm_console import WarmConsoleDashboard, WarmServerState


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


def test_warm_model_pool_reports_model_lifecycle(tmp_path: Path) -> None:
    source = tmp_path / "model.safetensors"
    source.write_bytes(b"weights")
    key = ModelCacheKey.create("text_encoder", source, torch.bfloat16, quantization="8bit")
    statuses: list[str] = []
    pool = WarmModelPool(status_listener=lambda status: statuses.append(status.status))

    pool.get_or_load(key, lambda _device: _TrackingModule(), "cpu", offloadable=False)
    pool.offload(key)

    assert statuses == ["loading", "loaded", "unloaded"]
    snapshot = pool.statuses[0].as_dict()
    assert snapshot["component"] == "text_encoder"
    assert snapshot["name"] == "model.safetensors"
    assert snapshot["options"] == {"quantization": "8bit"}
    assert snapshot["status"] == "unloaded"


def test_warm_model_pool_moves_real_tensors_without_materializing_meta_tensors(tmp_path: Path) -> None:
    source = tmp_path / "partial-model.safetensors"
    source.write_bytes(b"weights")
    key = ModelCacheKey.create("video_vae_encoder", source, torch.bfloat16)
    model = torch.nn.Module()
    model.encoder = torch.nn.Linear(2, 2)
    model.unused_decoder = torch.nn.Linear(2, 2, device="meta")
    pool = WarmModelPool()
    pool.replace(key, model)

    pool.offload(key)

    assert model.encoder.weight.device.type == "cpu"
    assert model.unused_decoder.weight.device.type == "meta"
    assert pool.statuses[0].device == "cpu + meta"


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


def test_warm_client_submits_to_running_server(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
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
            connection.sendall(
                b'{"status":"accepted"}\n'
                b'{"status":"output","stream":"stdout","text":"job output\\n"}\n'
                b'{"status":"output","stream":"stderr","text":"job warning\\n"}\n'
                b'{"status":"ok","result":{}}\n'
            )
        listener.close()

    thread = threading.Thread(target=serve_one)
    thread.start()
    assert submit_if_running("preprocess", {"dataset_file": "dataset.json"})
    thread.join(timeout=2)
    socket_path.unlink(missing_ok=True)

    assert received[0]["command"] == "preprocess"
    assert received[0]["args"] == {"dataset_file": "dataset.json"}
    captured = capsys.readouterr()
    assert "job output" in captured.out
    assert "job warning" in captured.err


def test_warm_server_forwards_job_streams_to_client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    server_socket, client_socket = socket.socketpair()
    server = warm_server.WarmModelServer(tmp_path / "warm.sock")

    def fake_job(_command: str, _args: dict) -> dict:
        sys.stdout.write("forwarded stdout\n")
        sys.stderr.write("forwarded stderr\n")
        return {"done": True}

    monkeypatch.setattr(server, "_run_job", fake_job)
    request = {
        "protocol": 1,
        "command": "preprocess",
        "args": {},
        "cwd": str(tmp_path),
        "terminal": {"stdout_isatty": False, "stderr_isatty": False, "columns": 100},
    }
    client_socket.sendall((json.dumps(request) + "\n").encode())
    server._handle_connection(server_socket)
    server_socket.close()
    responses = [json.loads(line) for line in client_socket.makefile("r", encoding="utf-8")]
    client_socket.close()

    output = {
        (response.get("stream"), response.get("text")) for response in responses if response["status"] == "output"
    }
    assert ("stdout", "forwarded stdout\n") in output
    assert ("stderr", "forwarded stderr\n") in output
    assert responses[-1] == {"status": "ok", "result": {"done": True}}
    assert server._state.snapshot()["completed_jobs"] == 1


def test_warm_console_dashboard_renders_model_status(tmp_path: Path) -> None:
    source = tmp_path / "model.safetensors"
    source.write_bytes(b"weights")
    key = ModelCacheKey.create("training_transformer", source, torch.bfloat16)
    state = WarmServerState()
    output = StringIO()
    render_console = Console(file=output, width=160, force_terminal=False)
    dashboard = WarmConsoleDashboard(
        state,
        render_console,
    )
    pool = WarmModelPool(status_listener=dashboard.update_model)
    pool.get_or_load(key, lambda _device: _TrackingModule(), "cpu")
    loading_key = ModelCacheKey.create("video_vae_encoder", source, torch.bfloat16)
    dashboard.update_model(ModelStatus(loading_key, "loading", device="cpu", detail="Moving to cuda:0"))
    render_console.print(dashboard.render())
    rendered = output.getvalue()

    assert state.snapshot()["counts"] == {"loaded": 1, "loading": 1, "unloaded": 0}
    assert "LTX WARM MODEL SERVER" in rendered
    assert "TRAINING TRANSFORMER" in rendered
    assert "VIDEO VAE ENCODER" in rendered
    assert "1 loaded" in rendered
    assert "1 loading" in rendered


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


def test_media_frame_count_metadata_is_reused(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"video")
    dataset_file = tmp_path / "dataset.json"
    dataset_file.write_text('[{"media_path":"clip.mp4"}]', encoding="utf-8")
    cache_path = tmp_path / ".precomputed" / ".media_metadata.json"
    probes: list[Path] = []
    monkeypatch.setattr(
        process_videos,
        "get_video_frame_count",
        lambda path: probes.append(path) or 25,
    )

    first = process_videos.MediaDataset(
        dataset_file=dataset_file,
        main_media_column="media_path",
        video_column="media_path",
        resolution_buckets=[(9, 64, 64)],
        frame_count_cache_path=cache_path,
    )
    second = process_videos.MediaDataset(
        dataset_file=dataset_file,
        main_media_column="media_path",
        video_column="media_path",
        resolution_buckets=[(9, 64, 64)],
        frame_count_cache_path=cache_path,
    )

    assert len(first) == len(second) == 1
    assert probes == [media]


def test_completed_latents_skip_video_metadata_probe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (tmp_path / "clip.mp4").write_bytes(b"video")
    dataset_file = tmp_path / "dataset.json"
    dataset_file.write_text('[{"media_path":"clip.mp4"}]', encoding="utf-8")
    output_dir = tmp_path / ".precomputed" / "latents"
    output_dir.mkdir(parents=True)
    (output_dir / "clip.pt").write_bytes(b"complete")
    monkeypatch.setattr(
        process_videos,
        "get_video_frame_count",
        lambda _path: (_ for _ in ()).throw(AssertionError("completed videos must not be probed")),
    )

    process_videos.compute_latents(
        dataset_file=dataset_file,
        video_column="media_path",
        resolution_buckets=[(9, 64, 64)],
        output_dir=str(output_dir),
        model_path=str(tmp_path / "unused.safetensors"),
        device="cpu",
    )


def test_preflight_dataset_index_is_reused_from_warm_pool(tmp_path: Path) -> None:
    model = tmp_path / "model.safetensors"
    model.write_bytes(b"model")
    data_root = tmp_path / "data" / ".precomputed"
    for source in ("latents", "conditions"):
        (data_root / source).mkdir(parents=True)
        (data_root / source / "sample.pt").write_bytes(b"sample")
    config = LtxTrainerConfig(
        model={"model_path": str(model), "training_mode": "lora"},
        lora={"rank": 8, "alpha": 8, "target_modules": ["to_q"]},
        data={"preprocessed_data_root": str(data_root.parent), "num_dataloader_workers": 0},
        validation={"prompts": [], "interval": None},
        output_dir=str(tmp_path / "output"),
    )
    pool = WarmModelPool()

    first = LtxvTrainer.preflight_config(config, model_pool=pool)
    second = LtxvTrainer.preflight_config(config, model_pool=pool)

    assert first.training_dataset is second.training_dataset


def test_gemma_text_encoder_batches_prompts_in_one_forward() -> None:
    class FakeTokenizer:
        @staticmethod
        def tokenize_batch(texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
            batch = len(texts)
            return torch.arange(batch * 3).reshape(batch, 3), torch.ones(batch, 3, dtype=torch.long)

    class FakeBackbone(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.language_model = torch.nn.Linear(1, 1)
            self.calls = 0

        def forward(
            self,
            *,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            output_hidden_states: bool,
        ) -> object:
            self.calls += 1
            assert output_hidden_states
            assert input_ids.shape == attention_mask.shape == (2, 3)
            return type("Output", (), {"hidden_states": (input_ids.unsqueeze(-1).float(),)})()

    class FakeConditionalModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = FakeBackbone()

    model = FakeConditionalModel()
    encoder = GemmaTextEncoder(model=model, tokenizer=FakeTokenizer())

    hidden_states, attention_mask = encoder.encode_batch(["first", "second"])

    assert model.model.calls == 1
    assert hidden_states[0].shape == (2, 3, 1)
    assert attention_mask.shape == (2, 3)


def test_caption_preprocessing_saves_each_item_from_one_batched_forward(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset_file = tmp_path / "dataset.json"
    dataset_file.write_text(
        '[{"caption":"first","media_path":"first.mp4"},{"caption":"second","media_path":"second.mp4"}]',
        encoding="utf-8",
    )
    batch_calls = 0
    saved: dict[str, dict[str, torch.Tensor]] = {}

    class FakeTextEncoder:
        def encode_batch(
            self,
            texts: list[str],
            *,
            padding_side: str,
        ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
            nonlocal batch_calls
            batch_calls += 1
            assert texts == ["first", "second"]
            assert padding_side == "left"
            hidden = torch.tensor([[[1.0]], [[2.0]]])
            return (hidden,), torch.ones(2, 1, dtype=torch.long)

    class FakeFeatureExtractor(torch.nn.Module):
        @staticmethod
        def forward(
            hidden_states: tuple[torch.Tensor, ...],
            _mask: torch.Tensor,
            _padding_side: str,
        ) -> tuple[torch.Tensor, None]:
            return hidden_states[0], None

    class FakeProcessor:
        feature_extractor = FakeFeatureExtractor()

    class FakeDataLoader(list):
        def __init__(self, items: list[dict[str, list[str]]]) -> None:
            super().__init__(items)
            self.dataset = [0, 1]

    monkeypatch.setattr(
        process_captions,
        "_build_sharded_dataloader",
        lambda *_args, **_kwargs: FakeDataLoader(
            [{"prompt": ["first", "second"], "output_path": ["first.pt", "second.pt"]}]
        ),
    )
    monkeypatch.setattr(process_captions, "load_text_encoder", lambda *_args, **_kwargs: FakeTextEncoder())
    monkeypatch.setattr(process_captions, "load_embeddings_processor", lambda *_args, **_kwargs: FakeProcessor())
    monkeypatch.setattr(
        process_captions,
        "_atomic_save",
        lambda payload, path: saved.__setitem__(path.name, payload),
    )

    process_captions.compute_captions_embeddings(
        dataset_file=dataset_file,
        output_dir=str(tmp_path / "conditions"),
        model_path=str(tmp_path / "model.safetensors"),
        text_encoder_path=str(tmp_path / "gemma.safetensors"),
        batch_size=2,
        device="cpu",
    )

    assert batch_calls == 1
    assert saved["first.pt"]["video_prompt_embeds"].item() == 1.0
    assert saved["second.pt"]["video_prompt_embeds"].item() == 2.0


def test_gemma_tokenizer_scalar_api_uses_batch_implementation() -> None:
    class FakeHuggingFaceTokenizer:
        def __call__(self, texts: list[str], **_kwargs: object) -> object:
            assert texts == ["trim me"]
            return type(
                "Encoded",
                (),
                {
                    "input_ids": torch.tensor([[1, 2, 0]]),
                    "attention_mask": torch.tensor([[1, 1, 0]]),
                },
            )()

    tokenizer = object.__new__(LTXVGemmaTokenizer)
    tokenizer.tokenizer = FakeHuggingFaceTokenizer()
    tokenizer.max_length = 3

    tokens = tokenizer.tokenize_with_weights("  trim me  ")["gemma"]

    assert [(int(token), int(weight)) for token, weight in tokens] == [(1, 1), (2, 1), (0, 0)]


def test_validation_embeddings_and_negative_prompt_are_cached(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model.safetensors"
    text_path = tmp_path / "gemma.safetensors"
    model_path.write_bytes(b"model")
    text_path.write_bytes(b"text")
    config = LtxTrainerConfig(
        model={
            "model_path": str(model_path),
            "text_encoder_path": str(text_path),
            "training_mode": "lora",
        },
        lora={"rank": 8, "alpha": 8, "target_modules": ["to_q"]},
        data={"preprocessed_data_root": str(tmp_path / "unused")},
        validation={
            "prompts": ["first", "second"],
            "negative_prompt": "negative",
            "interval": 10,
            "generate_audio": False,
        },
        output_dir=str(tmp_path / "output"),
    )
    encoded: list[str] = []
    text_encoder_loads = 0

    class FakeTextEncoder(torch.nn.Module):
        def encode(self, text: str) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
            encoded.append(text)
            value = torch.tensor([[[float(len(text))]]])
            return (value,), torch.ones(1, 1, dtype=torch.long)

    class FakeEmbeddingsProcessor(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.feature_extractor = torch.nn.Identity()

        @staticmethod
        def process_hidden_states(
            hidden_states: tuple[torch.Tensor, ...],
            _mask: torch.Tensor,
        ) -> object:
            return type(
                "Embeddings",
                (),
                {"video_encoding": hidden_states[0], "audio_encoding": None},
            )()

    def load_text_encoder(*_args: object, **_kwargs: object) -> FakeTextEncoder:
        nonlocal text_encoder_loads
        text_encoder_loads += 1
        return FakeTextEncoder()

    monkeypatch.setattr(trainer_module, "load_text_encoder", load_text_encoder)
    monkeypatch.setattr(
        trainer_module,
        "load_embeddings_processor",
        lambda *_args, **_kwargs: FakeEmbeddingsProcessor(),
    )
    pool = WarmModelPool()

    def build_trainer() -> LtxvTrainer:
        trainer = object.__new__(LtxvTrainer)
        trainer._config = config
        trainer._model_pool = pool
        trainer._warm_embeddings_key = None
        return trainer

    first = build_trainer()._load_text_encoder_and_cache_embeddings()
    second = build_trainer()._load_text_encoder_and_cache_embeddings()

    assert first is second
    assert text_encoder_loads == 1
    assert encoded == ["negative", "first", "second"]
