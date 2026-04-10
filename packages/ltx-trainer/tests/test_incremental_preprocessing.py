from __future__ import annotations

import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader as TorchDataLoader

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))
sys.path.insert(0, str(PACKAGE_ROOT / "scripts"))

import process_captions as process_captions_module
import process_dataset as process_dataset_module
import process_videos as process_videos_module
from ltx_trainer.datasets import PrecomputedDataset
from ltx_trainer.preprocessing_manifest import PreprocessingManifest


def write_dataset(path: Path, rows: list[dict[str, str]]) -> None:
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def save_latent_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "latents": torch.ones(1, 1, 1, 1),
            "num_frames": 1,
            "height": 1,
            "width": 1,
            "fps": 24,
        },
        path,
    )


def save_condition_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "video_prompt_embeds": torch.ones(1, 1),
            "audio_prompt_embeds": torch.ones(1, 1),
            "prompt_attention_mask": torch.ones(1, dtype=torch.bool),
        },
        path,
    )


def single_worker_dataloader(*args, **kwargs):
    kwargs["num_workers"] = 0
    return TorchDataLoader(*args, **kwargs)


def install_fake_captioning(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    encode_calls: list[str] = []

    class FakeTextEncoder:
        def encode(self, prompt: str, padding_side: str = "left"):
            del padding_side
            encode_calls.append(prompt)
            hidden_states = torch.full((1, 1, 1), float(len(prompt)), dtype=torch.float32)
            attention_mask = torch.ones(1, 1, dtype=torch.bool)
            return hidden_states, attention_mask

    class FakeEmbeddingsProcessor:
        def feature_extractor(self, hidden_states: torch.Tensor, prompt_attention_mask: torch.Tensor, padding_side: str):
            del padding_side
            return hidden_states + 1, hidden_states + 2

    monkeypatch.setattr(process_captions_module, "DataLoader", single_worker_dataloader)
    monkeypatch.setattr(process_captions_module, "load_text_encoder", lambda *args, **kwargs: FakeTextEncoder())
    monkeypatch.setattr(
        process_captions_module,
        "load_embeddings_processor",
        lambda *args, **kwargs: FakeEmbeddingsProcessor(),
    )
    return encode_calls


def install_fake_video_processing(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[tuple[int, ...]], list[int], list[str]]:
    encode_video_calls: list[tuple[int, ...]] = []
    encode_audio_calls: list[int] = []
    read_video_calls: list[str] = []

    def fake_read_video(path: Path, max_frames: int | None = None):
        read_video_calls.append(path.name)
        num_frames = max_frames or 9
        fill_value = 1.0 if path.stem.startswith("a") else 2.0
        video = torch.full((num_frames, 3, 32, 32), fill_value, dtype=torch.float32)
        return video, 24.0

    def fake_encode_video(vae, video: torch.Tensor, use_tiling: bool = False, **kwargs):
        del vae, use_tiling, kwargs
        encode_video_calls.append(tuple(video.shape))
        batch_size = video.shape[0]
        values = video.reshape(batch_size, -1).mean(dim=1).view(batch_size, 1, 1, 1, 1)
        return {
            "latents": values,
            "num_frames": 1,
            "height": 1,
            "width": 1,
        }

    def fake_encode_audio(audio_vae_encoder, audio_processor, audio):
        del audio_vae_encoder, audio_processor
        encode_audio_calls.append(audio.sampling_rate)
        return {
            "latents": torch.ones(1, 1, 1),
            "num_time_steps": 1,
            "frequency_bins": 1,
            "duration": audio.waveform.shape[-1] / audio.sampling_rate,
        }

    class FakeAudioProcessor:
        def __init__(self, **kwargs) -> None:
            del kwargs

        def to(self, device):
            del device
            return self

    fake_audio_vae = SimpleNamespace(
        sample_rate=16000,
        mel_bins=64,
        mel_hop_length=160,
        n_fft=400,
    )

    monkeypatch.setattr(process_videos_module, "DataLoader", single_worker_dataloader)
    monkeypatch.setattr(process_videos_module, "get_video_frame_count", lambda path: 9)
    monkeypatch.setattr(process_videos_module, "read_video", fake_read_video)
    monkeypatch.setattr(process_videos_module, "load_video_vae_encoder", lambda *args, **kwargs: object())
    monkeypatch.setattr(process_videos_module, "load_audio_vae_encoder", lambda *args, **kwargs: fake_audio_vae)
    monkeypatch.setattr(process_videos_module, "AudioProcessor", FakeAudioProcessor)
    monkeypatch.setattr(process_videos_module, "encode_video", fake_encode_video)
    monkeypatch.setattr(process_videos_module, "encode_audio", fake_encode_audio)
    monkeypatch.setattr(
        process_videos_module.torchaudio,
        "load",
        lambda path: (torch.ones(1, 16_000, dtype=torch.float32), 16_000),
    )

    return encode_video_calls, encode_audio_calls, read_video_calls


def test_preprocessing_manifest_round_trip_and_legacy_fallback(tmp_path: Path):
    precomputed_root = tmp_path / ".precomputed"
    latents_dir = precomputed_root / "latents"
    conditions_dir = precomputed_root / "conditions"

    save_latent_file(latents_dir / "sample.pt")
    save_condition_file(conditions_dir / "sample.pt")

    legacy_dataset = PrecomputedDataset(
        str(tmp_path),
        {"latents": "latents", "conditions": "conditions"},
    )
    assert len(legacy_dataset) == 1

    manifest = PreprocessingManifest.load(precomputed_root)
    manifest.update_source(
        source_name="latents",
        source_dir=latents_dir,
        config_signature="video-config",
        active_samples=["sample.pt"],
        sample_fingerprints={"sample.pt": "video-sample"},
    )
    manifest.update_source(
        source_name="conditions",
        source_dir=conditions_dir,
        config_signature="text-config",
        active_samples=["sample.pt"],
        sample_fingerprints={"sample.pt": "text-sample"},
    )
    manifest.save()

    reloaded_manifest = PreprocessingManifest.load(precomputed_root)
    assert reloaded_manifest.get_source("latents")["active_samples"] == ["sample.pt"]
    assert reloaded_manifest.get_source("conditions")["config_signature"] == "text-config"


def test_compute_captions_embeddings_is_incremental(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    encode_calls = install_fake_captioning(monkeypatch)

    dataset_path = tmp_path / "dataset.json"
    write_dataset(
        dataset_path,
        [
            {"caption": "first caption", "media_path": "videos/a.mp4"},
            {"caption": "second caption", "media_path": "videos/b.mp4"},
        ],
    )

    output_dir = tmp_path / ".precomputed" / "conditions"

    processed = process_captions_module.compute_captions_embeddings(
        dataset_file=dataset_path,
        output_dir=str(output_dir),
        model_path="model.safetensors",
        text_encoder_path="gemma",
        batch_size=2,
        device="cpu",
    )
    assert len(processed) == 2
    assert len(encode_calls) == 2

    encode_calls.clear()
    processed = process_captions_module.compute_captions_embeddings(
        dataset_file=dataset_path,
        output_dir=str(output_dir),
        model_path="model.safetensors",
        text_encoder_path="gemma",
        batch_size=2,
        device="cpu",
    )
    assert processed == []
    assert encode_calls == []

    write_dataset(
        dataset_path,
        [
            {"caption": "first caption", "media_path": "videos/a.mp4"},
            {"caption": "changed caption", "media_path": "videos/b.mp4"},
        ],
    )
    processed = process_captions_module.compute_captions_embeddings(
        dataset_file=dataset_path,
        output_dir=str(output_dir),
        model_path="model.safetensors",
        text_encoder_path="gemma",
        batch_size=2,
        device="cpu",
    )
    assert [path.name for path in processed] == ["b.pt"]

    encode_calls.clear()
    processed = process_captions_module.compute_captions_embeddings(
        dataset_file=dataset_path,
        output_dir=str(output_dir),
        model_path="model.safetensors",
        text_encoder_path="gemma",
        batch_size=2,
        device="cpu",
        lora_trigger="TOKEN",
    )
    assert len(processed) == 2
    assert len(encode_calls) == 2


def test_compute_latents_is_incremental_with_audio_and_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    encode_video_calls, encode_audio_calls, _ = install_fake_video_processing(monkeypatch)

    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    video_a = videos_dir / "a.mp4"
    video_b = videos_dir / "b.mp4"
    video_a.write_bytes(b"a")
    video_b.write_bytes(b"b")

    dataset_path = tmp_path / "dataset.json"
    write_dataset(
        dataset_path,
        [
            {"caption": "first", "media_path": "videos/a.mp4"},
            {"caption": "second", "media_path": "videos/b.mp4"},
        ],
    )

    output_dir = tmp_path / ".precomputed" / "latents"
    audio_output_dir = tmp_path / ".precomputed" / "audio_latents"

    result = process_videos_module.compute_latents(
        dataset_file=dataset_path,
        video_column="media_path",
        resolution_buckets=[(9, 32, 32)],
        output_dir=str(output_dir),
        model_path="model.safetensors",
        batch_size=4,
        device="cpu",
        with_audio=True,
        audio_output_dir=str(audio_output_dir),
    )
    assert len(result.processed_video_files) == 2
    assert len(result.processed_audio_files) == 2
    assert len(encode_video_calls) == 2
    assert len(encode_audio_calls) == 2

    encode_video_calls.clear()
    encode_audio_calls.clear()
    result = process_videos_module.compute_latents(
        dataset_file=dataset_path,
        video_column="media_path",
        resolution_buckets=[(9, 32, 32)],
        output_dir=str(output_dir),
        model_path="model.safetensors",
        batch_size=4,
        device="cpu",
        with_audio=True,
        audio_output_dir=str(audio_output_dir),
    )
    assert result.processed_video_files == []
    assert result.processed_audio_files == []
    assert encode_video_calls == []
    assert encode_audio_calls == []

    video_b.write_bytes(b"updated-media")
    result = process_videos_module.compute_latents(
        dataset_file=dataset_path,
        video_column="media_path",
        resolution_buckets=[(9, 32, 32)],
        output_dir=str(output_dir),
        model_path="model.safetensors",
        batch_size=4,
        device="cpu",
        with_audio=True,
        audio_output_dir=str(audio_output_dir),
    )
    assert [path.name for path in result.processed_video_files] == ["b.pt"]
    assert [path.name for path in result.processed_audio_files] == ["b.pt"]

    encode_video_calls.clear()
    encode_audio_calls.clear()
    result = process_videos_module.compute_latents(
        dataset_file=dataset_path,
        video_column="media_path",
        resolution_buckets=[(9, 64, 32)],
        output_dir=str(output_dir),
        model_path="model.safetensors",
        batch_size=4,
        device="cpu",
        with_audio=True,
        audio_output_dir=str(audio_output_dir),
    )
    assert len(result.processed_video_files) == 2
    assert len(result.processed_audio_files) == 2
    assert len(encode_video_calls) == 2
    assert len(encode_audio_calls) == 2

    encode_video_calls.clear()
    encode_audio_calls.clear()
    result = process_videos_module.compute_latents(
        dataset_file=dataset_path,
        video_column="media_path",
        resolution_buckets=[(9, 64, 32)],
        output_dir=str(output_dir),
        model_path="model.safetensors",
        batch_size=4,
        device="cpu",
        with_audio=True,
        audio_output_dir=str(audio_output_dir),
        override=True,
    )
    assert len(result.processed_video_files) == 2
    assert len(result.processed_audio_files) == 2
    assert len(encode_video_calls) == 2
    assert len(encode_audio_calls) == 2


def test_process_dataset_uses_manifest_for_removed_samples_and_optional_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    install_fake_captioning(monkeypatch)
    install_fake_video_processing(monkeypatch)
    monkeypatch.setattr(process_dataset_module, "free_gpu_memory_context", nullcontext)

    decoder_instances: list[object] = []

    class FakeDecoder:
        def __init__(self, **kwargs) -> None:
            del kwargs
            self.video_calls: list[list[str]] = []
            self.audio_calls: list[list[str]] = []
            decoder_instances.append(self)

        def decode_files(self, latent_files: list[Path], latents_dir: Path, output_dir: Path, seed: int | None = None):
            del latents_dir, output_dir, seed
            self.video_calls.append([path.name for path in latent_files])

        def decode_audio_files(self, latent_files: list[Path], latents_dir: Path, output_dir: Path):
            del latents_dir, output_dir
            self.audio_calls.append([path.name for path in latent_files])

    monkeypatch.setattr(process_dataset_module, "LatentsDecoder", FakeDecoder)

    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    video_a = videos_dir / "a.mp4"
    video_b = videos_dir / "b.mp4"
    video_a.write_bytes(b"a")
    video_b.write_bytes(b"b")

    dataset_path = tmp_path / "dataset.json"
    write_dataset(
        dataset_path,
        [
            {"caption": "first", "media_path": "videos/a.mp4"},
            {"caption": "second", "media_path": "videos/b.mp4"},
        ],
    )

    process_dataset_module.preprocess_dataset(
        dataset_file=str(dataset_path),
        caption_column="caption",
        video_column="media_path",
        resolution_buckets=[(9, 32, 32)],
        batch_size=2,
        output_dir=None,
        lora_trigger=None,
        vae_tiling=False,
        decode=True,
        model_path="model.safetensors",
        text_encoder_path="gemma",
        device="cpu",
        with_audio=True,
    )

    assert decoder_instances[0].video_calls == [["a.pt", "b.pt"]]
    assert decoder_instances[0].audio_calls == [["a.pt", "b.pt"]]

    write_dataset(
        dataset_path,
        [
            {"caption": "first", "media_path": "videos/a.mp4"},
        ],
    )
    video_a.write_bytes(b"updated-a")

    process_dataset_module.preprocess_dataset(
        dataset_file=str(dataset_path),
        caption_column="caption",
        video_column="media_path",
        resolution_buckets=[(9, 32, 32)],
        batch_size=2,
        output_dir=None,
        lora_trigger=None,
        vae_tiling=False,
        decode=True,
        model_path="model.safetensors",
        text_encoder_path="gemma",
        device="cpu",
        with_audio=False,
    )

    assert decoder_instances[1].video_calls == [["a.pt"]]
    assert decoder_instances[1].audio_calls == []

    dataset = PrecomputedDataset(
        str(tmp_path),
        {"latents": "latents", "conditions": "conditions"},
    )
    assert len(dataset) == 1

    with pytest.raises(FileNotFoundError):
        PrecomputedDataset(
            str(tmp_path),
            {
                "latents": "latents",
                "conditions": "conditions",
                "audio_latents": "audio_latents",
            },
        )

    with pytest.raises(FileNotFoundError):
        PrecomputedDataset(
            str(tmp_path),
            {
                "latents": "latents",
                "conditions": "conditions",
                "reference_latents": "reference_latents",
            },
        )

    stale_latent = tmp_path / ".precomputed" / "latents" / "videos" / "b.pt"
    assert stale_latent.exists()
