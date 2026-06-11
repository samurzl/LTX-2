# ruff: noqa: E402, I001

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
TRAINER_SRC = REPO_ROOT / "packages" / "ltx-trainer" / "src"
TRAINER_SCRIPTS = REPO_ROOT / "packages" / "ltx-trainer" / "scripts"
sys.path.insert(0, str(TRAINER_SRC))
sys.path.insert(0, str(TRAINER_SCRIPTS))

from ltx_trainer.config import ModelConfig, LoraConfig
from ltx_trainer.datasets import (
    OptionalSourceGroupedBatchSampler,
    PrecomputedDataset,
    collate_precomputed_samples,
)
from ltx_trainer.timestep_samplers import RangeScaledTimestepSampler, UniformTimestepSampler
from ltx_trainer.trainer import LtxvTrainer
from process_videos import _audio_has_activity


def test_noise_expert_config_rejects_overlaps() -> None:
    with pytest.raises(ValueError, match="must not overlap"):
        LoraConfig(noise_experts={"low": (0.0, 0.6), "high": (0.5, 1.0)})


def test_component_paths_can_replace_base_model_path() -> None:
    config = ModelConfig(
        model_path=None,
        text_encoder_path=None,
        component_paths={
            "transformer": "/comfy/diffusion_model.safetensors",
            "embeddings_processor": "/comfy/text_embedding_projection.safetensors",
            "video_vae_decoder": "/comfy/video_vae.safetensors",
            "text_encoder": "/comfy/gemma",
        },
    )

    assert config.model_path is None

    with pytest.raises(ValueError, match="component paths must be provided"):
        ModelConfig(model_path=None, component_paths={"transformer": "/comfy/diffusion_model.safetensors"})


def test_range_scaled_timestep_sampler_maps_into_expert_range() -> None:
    torch.manual_seed(0)
    sampler = RangeScaledTimestepSampler(UniformTimestepSampler(), 0.25, 0.5)
    samples = sampler.sample(batch_size=128, device=torch.device("cpu"))

    assert torch.all(samples >= 0.25)
    assert torch.all(samples <= 0.5)


def test_optional_audio_dataset_keeps_video_only_samples(tmp_path: Path) -> None:
    root = tmp_path / ".precomputed"
    for directory in ("latents", "conditions", "audio_latents"):
        (root / directory).mkdir(parents=True)

    latent = {"latents": torch.zeros(1, 1, 1, 1), "num_frames": 1, "height": 1, "width": 1}
    conditions = {"prompt_embeds": torch.zeros(2, 4), "prompt_attention_mask": torch.ones(2, dtype=torch.bool)}

    for name in ("a.pt", "b.pt"):
        torch.save(latent, root / "latents" / name)
        torch.save(conditions, root / "conditions" / name)
    torch.save({"latents": torch.zeros(8, 2, 16)}, root / "audio_latents" / "a.pt")

    dataset = PrecomputedDataset(
        str(tmp_path),
        data_sources={"latents": "latents", "conditions": "conditions", "audio_latents": "audio_latents"},
        optional_data_sources={"audio_latents"},
    )

    assert len(dataset) == 2
    assert dataset.has_source("audio_latents", 0)
    assert not dataset.has_source("audio_latents", 1)
    assert dataset[1]["audio_latents"] is None

    batches = list(OptionalSourceGroupedBatchSampler(dataset, "audio_latents", batch_size=1, shuffle=False))
    assert batches == [[0], [1]]

    collated = collate_precomputed_samples([dataset[1]])
    assert collated["audio_latents"] is None


def test_audio_activity_threshold_rejects_silence_and_accepts_signal() -> None:
    silent = torch.zeros(2, 16_000)
    signal = torch.zeros(2, 16_000)
    signal[:, :8_000] = 0.1
    phase_cancelled_signal = signal.clone()
    phase_cancelled_signal[1] *= -1

    assert not _audio_has_activity(silent, 16_000, -60.0, 0.01, 100.0)
    assert _audio_has_activity(signal, 16_000, -60.0, 0.01, 100.0)
    assert _audio_has_activity(phase_cancelled_signal, 16_000, -60.0, 0.01, 100.0)


def test_nsync_projection_is_layer_wise() -> None:
    p1 = torch.nn.Parameter(torch.zeros(2))
    p2 = torch.nn.Parameter(torch.zeros(2))
    p3 = torch.nn.Parameter(torch.zeros(2))
    p1.grad = torch.tensor([1.0, 0.0])
    p2.grad = torch.tensor([1.0, 1.0])
    p3.grad = torch.tensor([0.0, 1.0])
    negative_grads = {
        p1: torch.tensor([1.0, 0.0]),
        p2: torch.tensor([0.0, 1.0]),
        p3: torch.tensor([0.0, 1.0]),
    }

    trainer = object.__new__(LtxvTrainer)
    trainer._config = SimpleNamespace(lora=SimpleNamespace(nsync=SimpleNamespace(eps=1e-12)))
    trainer._gradient_projection_groups = {
        "transformer_blocks.0": [p1, p2],
        "transformer_blocks.1": [p3],
    }

    trainer._apply_nsync_projection(negative_grads)

    layer0_dot = torch.dot(p1.grad, negative_grads[p1]) + torch.dot(p2.grad, negative_grads[p2])
    layer1_dot = torch.dot(p3.grad, negative_grads[p3])
    assert layer0_dot.abs() < 1e-6
    assert layer1_dot.abs() < 1e-6
