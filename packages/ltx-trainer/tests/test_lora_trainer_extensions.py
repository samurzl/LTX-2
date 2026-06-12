# ruff: noqa: E402, I001

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
CORE_SRC = REPO_ROOT / "packages" / "ltx-core" / "src"
TRAINER_SRC = REPO_ROOT / "packages" / "ltx-trainer" / "src"
TRAINER_SCRIPTS = REPO_ROOT / "packages" / "ltx-trainer" / "scripts"
sys.path.insert(0, str(CORE_SRC))
sys.path.insert(0, str(TRAINER_SRC))
sys.path.insert(0, str(TRAINER_SCRIPTS))

from ltx_core.text_encoders.gemma import (
    DEFAULT_GEMMA_ASSET_SOURCE,
    gemma_weight_paths_from_source,
    module_ops_from_gemma_source,
)
from ltx_trainer.comfy_negative_backend import (
    SAVE_LATENT_NODE_ID,
    build_ltxv_negative_workflow,
    comfy_latent_bytes_to_trainer_payload,
)
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


def test_gemma_single_file_source_resolves_weights_and_tokenizer_ops(tmp_path: Path) -> None:
    gemma_file = tmp_path / "gemma.safetensors"
    gemma_file.write_bytes(b"")

    assert gemma_weight_paths_from_source(gemma_file) == (str(gemma_file),)

    ops = module_ops_from_gemma_source(gemma_file, include_processor=False)
    assert [op.name for op in ops] == ["TokenizerLoad"]


def test_gemma_directory_source_keeps_local_assets(tmp_path: Path) -> None:
    model_dir = tmp_path / "gemma"
    model_dir.mkdir()
    (model_dir / "model-00001-of-00001.safetensors").write_bytes(b"")
    (model_dir / "tokenizer.model").write_bytes(b"")

    assert gemma_weight_paths_from_source(model_dir) == (str(model_dir / "model-00001-of-00001.safetensors"),)

    ops = module_ops_from_gemma_source(model_dir, include_processor=False, allow_default_assets=False)
    assert [op.name for op in ops] == ["TokenizerLoad"]


def test_gemma_single_file_source_rejects_processor_when_default_disabled(tmp_path: Path) -> None:
    gemma_file = tmp_path / "gemma.safetensors"
    gemma_file.write_bytes(b"")

    with pytest.raises(FileNotFoundError, match=DEFAULT_GEMMA_ASSET_SOURCE):
        module_ops_from_gemma_source(gemma_file, allow_default_assets=False)


def test_comfy_negative_workflow_uses_ltxv_nodes() -> None:
    workflow = build_ltxv_negative_workflow(
        checkpoint_name="ltx.safetensors",
        text_encoder_name="gemma_fp4.safetensors",
        prompt="a test prompt",
        negative_prompt="low quality",
        width=768,
        height=512,
        frames=97,
        frame_rate=24.0,
        seed=123,
        guidance_scale=1.0,
        distilled_lora_name="distilled.safetensors",
        distilled_lora_strength=0.5,
        sigmas=[1.0, 0.5, 0.0],
    )

    assert workflow["2"]["class_type"] == "LTXAVTextEncoderLoader"
    assert workflow["2"]["inputs"]["text_encoder"] == "gemma_fp4.safetensors"
    assert workflow["3"]["inputs"]["lora_name"] == "distilled.safetensors"
    assert workflow["4"]["inputs"]["text"] == "a test prompt"
    assert workflow["7"]["inputs"] == {"width": 768, "height": 512, "length": 97, "batch_size": 1}
    assert workflow["13"]["class_type"] == "ManualSigmas"
    assert workflow["13"]["inputs"]["sigmas"] == "1, 0.5, 0"
    assert workflow[SAVE_LATENT_NODE_ID]["class_type"] == "SaveLatent"


def test_comfy_latent_bytes_convert_to_trainer_payload(tmp_path: Path) -> None:
    from safetensors.torch import save_file

    latent_file = tmp_path / "sample.latent"
    save_file(
        {
            "latent_tensor": torch.zeros(1, 128, 3, 4, 5),
            "latent_format_version_0": torch.tensor([]),
        },
        latent_file,
    )

    payload = comfy_latent_bytes_to_trainer_payload(latent_file.read_bytes(), frame_rate=24.0)

    assert payload["latents"].shape == (128, 3, 4, 5)
    assert payload["num_frames"] == 3
    assert payload["height"] == 4
    assert payload["width"] == 5
    assert payload["fps"] == 24.0


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
