# ruff: noqa: E402, I001

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Protocol

import pytest
import torch
import yaml
from accelerate import DistributedType
from pydantic import ValidationError
from safetensors.torch import save_file
from typer.testing import CliRunner

REPO_ROOT = Path(__file__).resolve().parents[3]
CORE_SRC = REPO_ROOT / "packages" / "ltx-core" / "src"
TRAINER_SRC = REPO_ROOT / "packages" / "ltx-trainer" / "src"
TRAINER_SCRIPTS = REPO_ROOT / "packages" / "ltx-trainer" / "scripts"
sys.path.insert(0, str(CORE_SRC))
sys.path.insert(0, str(TRAINER_SRC))
sys.path.insert(0, str(TRAINER_SCRIPTS))

from ltx_core.text_encoders.gemma import (
    DEFAULT_GEMMA_ASSET_SOURCE,
    GEMMA_LLM_KEY_OPS,
    gemma_weight_paths_from_source,
    module_ops_from_gemma_source,
)
from ltx_core.loader.sft_loader import SafetensorsStateDictLoader
from ltx_core.model.transformer.model_configurator import LTXV_MODEL_COMFY_RENAMING_MAP
from ltx_trainer.comfy_negative_backend import (
    SAVE_LATENT_NODE_ID,
    build_ltxv_negative_workflow,
    comfy_latent_bytes_to_trainer_payload,
)
import train as train_script
from ltx_trainer.config import AccelerationConfig, LoraConfig, LtxTrainerConfig, ModelConfig
from ltx_trainer.datasets import (
    AnchorSampleDataset,
    OptionalSourceGroupedBatchSampler,
    PrecomputedDataset,
    collate_precomputed_samples,
)
from ltx_trainer.model_loader import _transformer_sd_ops_for_checkpoint
from ltx_trainer.timestep_samplers import RangeScaledTimestepSampler, UniformTimestepSampler
from ltx_trainer.trainer import LtxvTrainer, NoiseExpert, TrainingStepOutput
from process_videos import _audio_has_activity


class _PeftLoraConfigLike(Protocol):
    r: int


def _write_lora_checkpoint(
    path: Path,
    rank: int,
    prefix: str = "diffusion_model.transformer_blocks.0.attn1.to_q",
) -> Path:
    save_file(
        {
            f"{prefix}.lora_A.weight": torch.zeros(rank, 3),
            f"{prefix}.lora_B.weight": torch.zeros(5, rank),
        },
        path,
    )
    return path


def _write_preflight_dataset(root: Path, *, sample_count: int = 1, include_negative: bool = False) -> Path:
    precomputed_root = root / ".precomputed"
    for directory in ("latents", "conditions"):
        (precomputed_root / directory).mkdir(parents=True)
    if include_negative:
        (precomputed_root / "negative_latents").mkdir(parents=True)

    for idx in range(sample_count):
        name = f"sample_{idx}.pt"
        (precomputed_root / "latents" / name).write_bytes(b"")
        (precomputed_root / "conditions" / name).write_bytes(b"")
        if include_negative:
            (precomputed_root / "negative_latents" / name).write_bytes(b"")

    return root


def _write_preflight_file(path: Path) -> str:
    path.write_bytes(b"")
    return str(path)


def _base_preflight_config(tmp_path: Path, *, sample_count: int = 1) -> LtxTrainerConfig:
    model_path = _write_preflight_file(tmp_path / "model.safetensors")
    text_encoder_path = _write_preflight_file(tmp_path / "gemma.safetensors")
    data_root = _write_preflight_dataset(tmp_path / "data", sample_count=sample_count)

    return LtxTrainerConfig(
        model={"model_path": model_path, "text_encoder_path": text_encoder_path, "training_mode": "lora"},
        lora={"rank": 8, "alpha": 8, "target_modules": ["to_q"]},
        data={"preprocessed_data_root": str(data_root), "num_dataloader_workers": 0},
        validation={"prompts": [], "interval": None, "generate_audio": True},
        output_dir=str(tmp_path / "outputs"),
    )


def _lora_rank_config(
    checkpoint_path: Path,
    rank: int,
    noise_experts: dict[str, tuple[float, float]] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        model=SimpleNamespace(training_mode="lora", load_checkpoint=checkpoint_path),
        lora=SimpleNamespace(
            rank=rank,
            alpha=rank,
            dropout=0.0,
            target_modules=["to_q"],
            noise_experts=noise_experts,
        ),
    )


def test_noise_expert_config_rejects_overlaps() -> None:
    with pytest.raises(ValueError, match="must not overlap"):
        LoraConfig(noise_experts={"low": (0.0, 0.6), "high": (0.5, 1.0)})


def test_nsync_config_uses_anchor_projection_by_default() -> None:
    assert LoraConfig(nsync={"enabled": True}).nsync.anchor_strength == 1.0


def test_validation_decode_block_offload_config_is_nonnegative() -> None:
    assert AccelerationConfig().offload_transformer_blocks_during_validation == 0
    config = AccelerationConfig(offload_transformer_blocks_during_validation=8)
    assert config.offload_transformer_blocks_during_validation == 8

    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        AccelerationConfig(offload_transformer_blocks_during_validation=-1)


def test_validation_decode_offload_moves_only_requested_trailing_blocks_and_restores_on_error() -> None:
    class TrackingBlock(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.moves: list[torch.device] = []

        def to(self, device: torch.device | str, *args, **kwargs) -> "TrackingBlock":
            del args, kwargs
            self.moves.append(torch.device(device))
            return self

    class FakeAccelerator:
        device = torch.device("cuda:0")
        distributed_type = DistributedType.NO

        @staticmethod
        def unwrap_model(model: torch.nn.Module, keep_torch_compile: bool = False) -> torch.nn.Module:
            del keep_torch_compile
            return model

    transformer = torch.nn.Module()
    transformer.transformer_blocks = torch.nn.ModuleList([TrackingBlock() for _ in range(4)])
    trainer = object.__new__(LtxvTrainer)
    trainer._config = SimpleNamespace(acceleration=SimpleNamespace(offload_transformer_blocks_during_validation=2))
    trainer._accelerator = FakeAccelerator()
    trainer._transformer = transformer

    with (
        pytest.raises(RuntimeError, match="decode failed"),
        trainer._offloaded_transformer_blocks_for_validation_decode(),
    ):
        raise RuntimeError("decode failed")

    blocks = list(transformer.transformer_blocks)
    assert blocks[0].moves == []
    assert blocks[1].moves == []
    assert blocks[2].moves == [torch.device("cpu"), torch.device("cuda:0")]
    assert blocks[3].moves == [torch.device("cpu"), torch.device("cuda:0")]


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


def test_trainer_preflight_accepts_valid_monolithic_config(tmp_path: Path) -> None:
    config = _base_preflight_config(tmp_path)

    result = LtxvTrainer.preflight_config(config)

    assert result.checkpoint_path is None
    assert result.resume_state is None


def test_validation_loss_requires_held_out_dataset(tmp_path: Path) -> None:
    config_data = _base_preflight_config(tmp_path).model_dump()
    config_data["validation"]["loss_interval"] = 10

    with pytest.raises(ValidationError, match="validation_data_root is required"):
        LtxTrainerConfig(**config_data)


def test_trainer_preflight_accepts_held_out_dataset(tmp_path: Path) -> None:
    config = _base_preflight_config(tmp_path)
    validation_root = _write_preflight_dataset(tmp_path / "validation_data", sample_count=2)
    config.data.validation_data_root = str(validation_root)
    config.validation.loss_interval = 10

    result = LtxvTrainer.preflight_config(config)

    assert result.checkpoint_path is None


def test_held_out_validation_loss_aggregates_batches_and_restores_mode() -> None:
    class FakeAccelerator:
        device = torch.device("cpu")
        process_index = 0
        is_main_process = True

        @staticmethod
        def gather_for_metrics(value: torch.Tensor) -> torch.Tensor:
            return value

        @staticmethod
        def wait_for_everyone() -> None:
            return None

    trainer = object.__new__(LtxvTrainer)
    trainer._config = SimpleNamespace(validation=SimpleNamespace(seed=42, max_loss_batches=2))
    trainer._accelerator = FakeAccelerator()
    trainer._transformer = torch.nn.Linear(1, 1)
    trainer._transformer.train()
    trainer._validation_dataloader = [
        {"loss": torch.tensor([1.0, 3.0])},
        {"loss": torch.tensor([5.0])},
        {"loss": torch.tensor([100.0])},
    ]
    expert = NoiseExpert("default", 0.0, 1.0, UniformTimestepSampler())
    trainer._noise_experts = [expert]
    trainer._active_noise_expert = expert
    trainer._global_step = 7
    trainer._set_active_lora_adapter = lambda _name: None
    trainer._training_step = lambda batch, _sampler: TrainingStepOutput(
        loss=batch["loss"],
        sigma=torch.zeros_like(batch["loss"]),
    )
    logged: list[tuple[dict[str, float], int | None]] = []
    trainer._log_metrics = lambda metrics, step=None: logged.append((metrics, step))

    rng_state = torch.random.get_rng_state()
    metrics = trainer._run_validation_loss()

    assert metrics == {"validation/loss": pytest.approx(3.0)}
    assert logged == [(metrics, 7)]
    assert trainer._transformer.training
    assert torch.equal(torch.random.get_rng_state(), rng_state)


def test_metric_logging_writes_tensorboard_scalars_at_global_step() -> None:
    class FakeWriter:
        def __init__(self) -> None:
            self.scalars: list[tuple[str, float, int]] = []

        def add_scalar(self, name: str, value: float, step: int) -> None:
            self.scalars.append((name, value, step))

    trainer = object.__new__(LtxvTrainer)
    writer = FakeWriter()
    trainer._wandb_run = None
    trainer._tensorboard_writer = writer
    trainer._global_step = 12

    trainer._log_metrics({"train/loss": 1.25, "validation/loss": 2.5})

    assert writer.scalars == [
        ("train/loss", 1.25, 12),
        ("validation/loss", 2.5, 12),
    ]


def test_validation_metric_logging_does_not_advance_wandb_step() -> None:
    class FakeWandbRun:
        def __init__(self) -> None:
            self.calls: list[tuple[dict[str, float], int, bool]] = []

        def log(self, metrics: dict[str, float], *, step: int, commit: bool) -> None:
            self.calls.append((metrics, step, commit))

    trainer = object.__new__(LtxvTrainer)
    run = FakeWandbRun()
    trainer._wandb_run = run
    trainer._tensorboard_writer = None
    trainer._global_step = 12

    trainer._log_metrics({"validation/loss": 2.5}, step=12)

    assert run.calls == [({"validation/loss": 2.5}, 12, False)]


def test_trainer_preflight_accepts_split_components_without_unused_validation_models(tmp_path: Path) -> None:
    data_root = _write_preflight_dataset(tmp_path / "data")
    transformer = _write_preflight_file(tmp_path / "transformer.safetensors")
    embeddings_processor = _write_preflight_file(tmp_path / "embeddings_processor.safetensors")

    config = LtxTrainerConfig(
        model={
            "model_path": None,
            "training_mode": "lora",
            "component_paths": {
                "transformer": transformer,
                "embeddings_processor": embeddings_processor,
            },
        },
        lora={"rank": 8, "alpha": 8, "target_modules": ["to_q"]},
        data={"preprocessed_data_root": str(data_root), "num_dataloader_workers": 0},
        validation={"prompts": [], "interval": None, "generate_audio": False},
        output_dir=str(tmp_path / "outputs"),
    )

    result = LtxvTrainer.preflight_config(config)

    assert result.checkpoint_path is None


def test_trainer_preflight_requires_used_split_audio_components(tmp_path: Path) -> None:
    data_root = _write_preflight_dataset(tmp_path / "data")
    transformer = _write_preflight_file(tmp_path / "transformer.safetensors")
    embeddings_processor = _write_preflight_file(tmp_path / "embeddings_processor.safetensors")
    video_vae = _write_preflight_file(tmp_path / "video_vae.safetensors")
    audio_vae = _write_preflight_file(tmp_path / "audio_vae.safetensors")
    text_encoder = _write_preflight_file(tmp_path / "text_encoder.safetensors")

    config = LtxTrainerConfig(
        model={
            "model_path": None,
            "training_mode": "lora",
            "component_paths": {
                "transformer": transformer,
                "embeddings_processor": embeddings_processor,
                "video_vae_decoder": video_vae,
                "audio_vae_decoder": audio_vae,
                "text_encoder": text_encoder,
            },
        },
        lora={"rank": 8, "alpha": 8, "target_modules": ["to_q"]},
        data={"preprocessed_data_root": str(data_root), "num_dataloader_workers": 0},
        validation={"prompts": ["validation prompt"], "interval": 10, "generate_audio": True},
        output_dir=str(tmp_path / "outputs"),
    )

    with pytest.raises(ValueError, match="vocoder"):
        LtxvTrainer.preflight_config(config)


def test_trainer_preflight_rejects_missing_dataset_source(tmp_path: Path) -> None:
    config = _base_preflight_config(tmp_path)
    conditions_dir = Path(config.data.preprocessed_data_root) / ".precomputed" / "conditions"
    for sample in conditions_dir.glob("*.pt"):
        sample.unlink()
    conditions_dir.rmdir()

    with pytest.raises(FileNotFoundError, match="conditions"):
        LtxvTrainer.preflight_config(config)


def test_trainer_preflight_rejects_empty_dataset(tmp_path: Path) -> None:
    config = _base_preflight_config(tmp_path)
    for sample in (Path(config.data.preprocessed_data_root) / ".precomputed" / "latents").glob("*.pt"):
        sample.unlink()

    with pytest.raises(ValueError, match="No data files"):
        LtxvTrainer.preflight_config(config)


def test_trainer_preflight_rejects_single_sample_anchor_training(tmp_path: Path) -> None:
    model_path = _write_preflight_file(tmp_path / "model.safetensors")
    text_encoder_path = _write_preflight_file(tmp_path / "gemma.safetensors")
    data_root = _write_preflight_dataset(tmp_path / "data", sample_count=1, include_negative=True)

    config = LtxTrainerConfig(
        model={"model_path": model_path, "text_encoder_path": text_encoder_path, "training_mode": "lora"},
        lora={
            "rank": 8,
            "alpha": 8,
            "target_modules": ["to_q"],
            "nsync": {"enabled": True, "negative_strength": 0.0, "anchor_strength": 1.0},
        },
        data={"preprocessed_data_root": str(data_root), "num_dataloader_workers": 0},
        validation={"prompts": [], "interval": None, "generate_audio": False},
        output_dir=str(tmp_path / "outputs"),
    )

    with pytest.raises(ValueError, match="at least two positive samples"):
        LtxvTrainer.preflight_config(config)


def test_trainer_preflight_rejects_bad_output_dir(tmp_path: Path) -> None:
    config = _base_preflight_config(tmp_path)
    output_file = tmp_path / "not_a_dir"
    output_file.write_text("occupied", encoding="utf-8")
    config.output_dir = str(output_file)

    with pytest.raises(FileExistsError):
        LtxvTrainer.preflight_config(config)


def test_optimization_config_rejects_non_positive_values(tmp_path: Path) -> None:
    config = _base_preflight_config(tmp_path)
    data = config.model_dump()
    data["optimization"]["learning_rate"] = 0.0

    with pytest.raises(ValidationError, match="greater than 0"):
        LtxTrainerConfig(**data)


def test_trainer_preflight_rejects_invalid_scheduler_params(tmp_path: Path) -> None:
    config = _base_preflight_config(tmp_path)
    config.optimization.scheduler_params = {"not_a_scheduler_arg": 1}

    with pytest.raises(ValueError, match="Unknown scheduler_params"):
        LtxvTrainer.preflight_config(config)


def test_train_cli_preflight_runs_before_trainer_construction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _base_preflight_config(tmp_path)
    config.data.preprocessed_data_root = str(tmp_path / "missing_data")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config.model_dump()), encoding="utf-8")

    def fail_if_constructed(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("trainer constructor should not run after preflight failure")

    monkeypatch.setattr(train_script.LtxvTrainer, "__init__", fail_if_constructed)

    result = CliRunner().invoke(train_script.app, [str(config_path)])

    assert result.exit_code == 1
    assert "Preflight validation failed" in result.output


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


def test_gemma_key_ops_accept_hf_and_comfy_single_file_layouts() -> None:
    assert (
        GEMMA_LLM_KEY_OPS.apply_to_key("model.language_model.layers.0.self_attn.q_proj.weight")
        == "model.model.language_model.layers.0.self_attn.q_proj.weight"
    )
    assert (
        GEMMA_LLM_KEY_OPS.apply_to_key("language_model.model.layers.0.self_attn.q_proj.weight")
        == "model.model.language_model.layers.0.self_attn.q_proj.weight"
    )
    assert (
        GEMMA_LLM_KEY_OPS.apply_to_key("model.layers.47.self_attn.q_norm.weight")
        == "model.model.language_model.layers.47.self_attn.q_norm.weight"
    )
    assert (
        GEMMA_LLM_KEY_OPS.apply_to_key("model.embed_tokens.weight") == "model.model.language_model.embed_tokens.weight"
    )
    assert GEMMA_LLM_KEY_OPS.apply_to_key("_quantization_metadata") is None

    tensor = torch.zeros(1)
    mapped = GEMMA_LLM_KEY_OPS.apply_to_key_value("model.model.language_model.embed_tokens.weight", tensor)
    assert [key for key, _ in mapped] == [
        "model.model.language_model.embed_tokens.weight",
        "model.lm_head.weight",
    ]


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


def test_transformer_loader_folds_scaled_fp8_weights(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ltx_fp8.safetensors"
    save_file(
        {
            "model.diffusion_model.transformer_blocks.0.attn1.to_q.weight": torch.full(
                (2, 2),
                2.0,
                dtype=torch.float8_e4m3fn,
            ),
            "model.diffusion_model.transformer_blocks.0.attn1.to_q.weight_scale": torch.tensor(
                0.5,
                dtype=torch.float32,
            ),
        },
        checkpoint,
    )

    sd_ops = _transformer_sd_ops_for_checkpoint(checkpoint, LTXV_MODEL_COMFY_RENAMING_MAP)
    state = SafetensorsStateDictLoader().load(str(checkpoint), sd_ops=sd_ops, device=torch.device("cpu")).sd

    weight = state["transformer_blocks.0.attn1.to_q.weight"]
    assert weight.dtype == torch.bfloat16
    assert torch.allclose(weight.float(), torch.ones_like(weight.float()))
    assert "transformer_blocks.0.attn1.to_q.weight_scale" not in state


def test_transformer_loader_rejects_native_fp4_files(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ltx_nvfp4.safetensors"
    checkpoint.write_bytes(b"not a real checkpoint")

    with pytest.raises(ValueError, match="FP4/NVFP4"):
        _transformer_sd_ops_for_checkpoint(checkpoint, LTXV_MODEL_COMFY_RENAMING_MAP)


def test_lora_rank_detection_accepts_comfy_and_internal_keys() -> None:
    state_dict = {
        "diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight": torch.zeros(384, 3),
        "diffusion_model.transformer_blocks.0.attn1.to_q.lora_B.weight": torch.zeros(5, 384),
        "transformer_blocks.0.attn1.to_k.lora_A.weight": torch.zeros(384, 7),
        "transformer_blocks.0.attn1.to_k.lora_B.weight": torch.zeros(11, 384),
    }

    assert LtxvTrainer._detect_lora_rank_from_state_dict(state_dict, "checkpoint.safetensors") == 384


def test_lora_rank_detection_rejects_inconsistent_pair() -> None:
    state_dict = {
        "transformer_blocks.0.attn1.to_q.lora_A.weight": torch.zeros(8, 3),
        "transformer_blocks.0.attn1.to_q.lora_B.weight": torch.zeros(5, 4),
    }

    with pytest.raises(ValueError, match="inconsistent rank"):
        LtxvTrainer._detect_lora_rank_from_state_dict(state_dict, "checkpoint.safetensors")


def test_lora_checkpoint_rank_overrides_config_before_setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checkpoint = _write_lora_checkpoint(tmp_path / "lora_weights_step_00001.safetensors", rank=384)
    captured: dict[str, object] = {}

    class FakeTransformer:
        def set_adapter(self, adapter_name: str) -> None:
            self.adapter_name = adapter_name

        def named_parameters(self) -> list[tuple[str, torch.nn.Parameter]]:
            return []

    def fake_get_peft_model(
        transformer: FakeTransformer,
        lora_config: _PeftLoraConfigLike,
        adapter_name: str,
    ) -> FakeTransformer:
        captured["rank"] = lora_config.r
        captured["adapter"] = adapter_name
        return transformer

    monkeypatch.setattr("ltx_trainer.trainer.get_peft_model", fake_get_peft_model)
    trainer = object.__new__(LtxvTrainer)
    trainer._config = _lora_rank_config(checkpoint, rank=64)
    trainer._transformer = FakeTransformer()

    trainer._resolve_lora_rank_from_checkpoint()
    trainer._setup_lora()

    assert trainer._config.lora.rank == 384
    assert captured == {"rank": 384, "adapter": "default"}


def test_lora_checkpoint_matching_rank_keeps_config_rank(tmp_path: Path) -> None:
    checkpoint = _write_lora_checkpoint(tmp_path / "lora_weights_step_00001.safetensors", rank=64)
    trainer = object.__new__(LtxvTrainer)
    trainer._config = _lora_rank_config(checkpoint, rank=64)

    trainer._resolve_lora_rank_from_checkpoint()

    assert trainer._config.lora.rank == 64


def test_lora_multi_expert_rank_detection_requires_matching_ranks(tmp_path: Path) -> None:
    low = _write_lora_checkpoint(tmp_path / "lora_weights_low_step_00010.safetensors", rank=32)
    _write_lora_checkpoint(tmp_path / "lora_weights_high_step_00010.safetensors", rank=32)
    trainer = object.__new__(LtxvTrainer)
    trainer._config = _lora_rank_config(low, rank=64, noise_experts={"low": (0.0, 0.5), "high": (0.5, 1.0)})

    trainer._resolve_lora_rank_from_checkpoint()

    assert trainer._config.lora.rank == 32

    _write_lora_checkpoint(tmp_path / "lora_weights_high_step_00010.safetensors", rank=16)
    trainer._config = _lora_rank_config(low, rank=64, noise_experts={"low": (0.0, 0.5), "high": (0.5, 1.0)})

    with pytest.raises(ValueError, match="must all use the same rank"):
        trainer._resolve_lora_rank_from_checkpoint()


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


def test_anchor_dataset_adds_different_positive_sample(tmp_path: Path) -> None:
    root = tmp_path / ".precomputed"
    for directory in ("latents", "conditions"):
        (root / directory).mkdir(parents=True)

    for idx, name in enumerate(("a.pt", "b.pt")):
        latent = {
            "latents": torch.full((1, 1, 1, 1), float(idx)),
            "num_frames": 1,
            "height": 1,
            "width": 1,
        }
        conditions = {
            "prompt_embeds": torch.full((2, 4), float(idx)),
            "prompt_attention_mask": torch.ones(2, dtype=torch.bool),
        }
        torch.save(latent, root / "latents" / name)
        torch.save(conditions, root / "conditions" / name)

    dataset = AnchorSampleDataset(PrecomputedDataset(str(tmp_path), data_sources=["latents", "conditions"]))
    sample = dataset[0]

    assert sample["idx"] == 0
    assert sample["anchor_idx"] == 1
    assert sample["latents"]["latents"].item() == 0.0
    assert sample["anchor_latents"]["latents"].item() == 1.0
    assert sample["anchor_conditions"]["prompt_embeds"][0, 0].item() == 1.0


def test_anchor_dataset_requires_two_positive_samples(tmp_path: Path) -> None:
    root = tmp_path / ".precomputed"
    for directory in ("latents", "conditions"):
        (root / directory).mkdir(parents=True)

    latent = {
        "latents": torch.zeros(1, 1, 1, 1),
        "num_frames": 1,
        "height": 1,
        "width": 1,
    }
    conditions = {
        "prompt_embeds": torch.zeros(2, 4),
        "prompt_attention_mask": torch.ones(2, dtype=torch.bool),
    }
    torch.save(latent, root / "latents" / "a.pt")
    torch.save(conditions, root / "conditions" / "a.pt")

    dataset = PrecomputedDataset(str(tmp_path), data_sources=["latents", "conditions"])
    with pytest.raises(ValueError, match="at least two positive samples"):
        AnchorSampleDataset(dataset)


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
    trainer._config = SimpleNamespace(lora=SimpleNamespace(nsync=SimpleNamespace(eps=1e-12, negative_strength=1.0)))
    trainer._gradient_projection_groups = {
        "transformer_blocks.0": [p1, p2],
        "transformer_blocks.1": [p3],
    }

    trainer._apply_nsync_projection(negative_grads)

    layer0_dot = torch.dot(p1.grad, negative_grads[p1]) + torch.dot(p2.grad, negative_grads[p2])
    layer1_dot = torch.dot(p3.grad, negative_grads[p3])
    assert layer0_dot.abs() < 1e-6
    assert layer1_dot.abs() < 1e-6


def test_nsync_negative_strength_scales_projection() -> None:
    p = torch.nn.Parameter(torch.zeros(2))
    p.grad = torch.tensor([2.0, 1.0])
    negative_grads = {p: torch.tensor([2.0, 0.0])}

    trainer = object.__new__(LtxvTrainer)
    trainer._config = SimpleNamespace(lora=SimpleNamespace(nsync=SimpleNamespace(eps=1e-12, negative_strength=0.25)))
    trainer._gradient_projection_groups = {"all": [p]}

    trainer._apply_nsync_projection(negative_grads)

    assert torch.allclose(p.grad, torch.tensor([1.5, 1.0]))


def test_nsync_anchor_projection_is_added() -> None:
    p = torch.nn.Parameter(torch.zeros(2))
    p.grad = torch.tensor([1.0, 1.0])
    negative_grads = {p: torch.tensor([1.0, 0.0])}
    anchor_grads = {p: torch.tensor([0.0, 1.0])}

    trainer = object.__new__(LtxvTrainer)
    trainer._config = SimpleNamespace(
        lora=SimpleNamespace(nsync=SimpleNamespace(eps=1e-12, negative_strength=1.0, anchor_strength=1.0))
    )
    trainer._gradient_projection_groups = {"all": [p]}

    trainer._apply_nsync_projection(negative_grads, anchor_grads)

    assert torch.allclose(p.grad, torch.tensor([0.0, 2.0]))


def test_nsync_zero_negative_strength_skips_negative_batch_requirement() -> None:
    backward_losses: list[torch.Tensor] = []

    trainer = object.__new__(LtxvTrainer)
    trainer._accelerator = SimpleNamespace(backward=lambda loss: backward_losses.append(loss.detach()))
    trainer._config = SimpleNamespace(
        lora=SimpleNamespace(nsync=SimpleNamespace(eps=1e-12, negative_strength=0.0, anchor_strength=0.0))
    )
    trainer._training_step = lambda batch, timestep_sampler: SimpleNamespace(  # noqa: ARG005
        loss=torch.ones(1, requires_grad=True),
        sigma=torch.ones(1),
    )

    output = trainer._nsync_training_step({"latents": {"latents": torch.zeros(1)}}, timestep_sampler=object())

    assert output.loss.shape == (1,)
    assert len(backward_losses) == 1


def test_nsync_reuses_processed_conditions_without_second_backward_error() -> None:
    class FakeAccelerator:
        def backward(self, loss: torch.Tensor) -> None:
            loss.backward()

    class FakeEmbeddingsProcessor(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))

        def create_embeddings(
            self,
            video_features: torch.Tensor,
            _audio_features: torch.Tensor | None,
            _additive_attention_mask: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
            batch_size, seq_len = video_features.shape[:2]
            attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
            return video_features * self.weight, None, attention_mask

    class FakeTransformer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))

        def forward(
            self,
            *,
            video: SimpleNamespace,
            audio: None,  # noqa: ARG002
            perturbations: None,  # noqa: ARG002
        ) -> tuple[torch.Tensor, None]:
            return video.context * self.weight, None

    class FakeStrategy:
        def prepare_training_inputs(self, batch: dict, _timestep_sampler: object) -> SimpleNamespace:
            context = batch["conditions"]["video_prompt_embeds"]
            return SimpleNamespace(
                video=SimpleNamespace(enabled=True, sigma=torch.ones(context.shape[0]), context=context),
                audio=None,
            )

        def compute_loss(
            self,
            video_pred: torch.Tensor,
            _audio_pred: None,
            _model_inputs: SimpleNamespace,
        ) -> torch.Tensor:
            return video_pred.flatten(1).mean(dim=1)

    transformer = FakeTransformer()
    trainer = object.__new__(LtxvTrainer)
    trainer._accelerator = FakeAccelerator()
    trainer._config = SimpleNamespace(lora=SimpleNamespace(nsync=SimpleNamespace(eps=1e-12, negative_strength=1.0)))
    trainer._embeddings_processor = FakeEmbeddingsProcessor()
    trainer._training_strategy = FakeStrategy()
    trainer._transformer = transformer
    trainer._trainable_params = [transformer.weight]
    trainer._gradient_projection_groups = {"all": [transformer.weight]}

    conditions = {
        "video_prompt_embeds": torch.ones(1, 4, 2),
        "audio_prompt_embeds": None,
        "prompt_attention_mask": torch.ones(1, 4, dtype=torch.int64),
    }
    batch = {
        "conditions": conditions,
        "latents": {"latents": torch.zeros(1)},
        "negative_latents": {"latents": torch.zeros(1)},
    }

    output = trainer._nsync_training_step(batch, timestep_sampler=object())

    assert output.loss.shape == (1,)
    assert conditions["_embeddings_processed"] is True
    assert not conditions["video_prompt_embeds"].requires_grad
