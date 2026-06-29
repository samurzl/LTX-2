import contextlib
import importlib.util
import math
import os
import re
import time
import warnings
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import torch
import wandb
import yaml
from accelerate import Accelerator, DistributedDataParallelKwargs, DistributedType
from accelerate.utils import gather_object, set_seed
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils import ModulesToSaveWrapper
from pydantic import BaseModel
from safetensors.torch import load_file, save_file
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    LinearLR,
    LRScheduler,
    PolynomialLR,
    StepLR,
)
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import functional as F

from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.text_encoders.gemma import convert_to_additive_mask
from ltx_trainer import logger
from ltx_trainer.config import LtxTrainerConfig
from ltx_trainer.config_display import print_config
from ltx_trainer.datasets import (
    AnchorSampleDataset,
    OptionalSourceGroupedBatchSampler,
    PrecomputedDataset,
    collate_precomputed_samples,
)
from ltx_trainer.gpu_utils import free_gpu_memory, free_gpu_memory_context, get_gpu_memory_gb
from ltx_trainer.hf_hub_utils import push_to_hub
from ltx_trainer.model_loader import (
    load_audio_vae_decoder,
    load_embeddings_processor,
    load_text_encoder,
    load_transformer,
    load_video_vae_decoder,
    load_video_vae_encoder,
    load_vocoder,
)
from ltx_trainer.model_loader import load_model as load_ltx_model
from ltx_trainer.model_pool import ModelCacheKey, WarmModelPool
from ltx_trainer.progress import TrainingProgress
from ltx_trainer.quantization import quantize_model
from ltx_trainer.sigma_tracker import SigmaBucketTracker
from ltx_trainer.timestep_samplers import SAMPLERS, RangeScaledTimestepSampler, TimestepSampler
from ltx_trainer.training_state import ConfigFingerprint, RngStates, TrainingState
from ltx_trainer.training_strategies import get_training_strategy
from ltx_trainer.utils import open_image_as_srgb, save_image
from ltx_trainer.validation_sampler import CachedPromptEmbeddings, GenerationConfig, ValidationSampler
from ltx_trainer.video_utils import read_video, save_video

# Disable irrelevant warnings from transformers
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Silence bitsandbytes warnings about casting
warnings.filterwarnings(
    "ignore", message="MatMul8bitLt: inputs will be cast from torch.bfloat16 to float16 during quantization"
)

# Disable progress bars if not main process
IS_MAIN_PROCESS = os.environ.get("LOCAL_RANK", "0") == "0"
if not IS_MAIN_PROCESS:
    from transformers.utils.logging import disable_progress_bar

    disable_progress_bar()

StepCallback = Callable[[int, int, list[Path] | None], None]  # (step, total, sampled paths or None) -> None

MEMORY_CHECK_INTERVAL = 200


class TrainingStats(BaseModel):
    """Statistics collected during training"""

    total_time_seconds: float
    steps_per_second: float
    samples_per_second: float
    peak_gpu_memory_gb: float
    global_batch_size: int
    num_processes: int


@dataclass(frozen=True)
class TrainingStepOutput:
    """Output from a single training step."""

    loss: Tensor  # [B,] per-element loss (unreduced)
    sigma: Tensor  # [B,] sampled sigma, detached from computational graph


@dataclass(frozen=True)
class NoiseExpert:
    """A named LoRA adapter and the sigma interval it trains on."""

    name: str
    min_sigma: float
    max_sigma: float
    sampler: TimestepSampler


@dataclass(frozen=True)
class TrainerPreflightResult:
    """Cheap startup checks that can be reused during trainer initialization."""

    checkpoint_path: Path | None = None
    resume_state: tuple[int, TrainingState | None] | None = None


class LtxvTrainer:
    @classmethod
    def preflight_config(cls, trainer_config: LtxTrainerConfig) -> TrainerPreflightResult:
        """Run cheap validation before model/text/dataloader-heavy initialization."""
        cls._preflight_optimization_config(trainer_config)
        cls._preflight_model_paths(trainer_config)
        cls._preflight_output_dir(trainer_config)
        cls._preflight_dataset(trainer_config)

        checkpoint_path = cls._preflight_checkpoint(trainer_config)
        resume_state = None
        if checkpoint_path is not None:
            resume_state = cls._resolve_resume_state_from_checkpoint(trainer_config, checkpoint_path)

        return TrainerPreflightResult(checkpoint_path=checkpoint_path, resume_state=resume_state)

    @classmethod
    def _preflight_optimization_config(cls, trainer_config: LtxTrainerConfig) -> None:
        opt_cfg = trainer_config.optimization
        scheduler_type = opt_cfg.scheduler_type
        params = opt_cfg.scheduler_params or {}

        allowed_params = {
            "constant": set(),
            "linear": {"start_factor", "end_factor", "last_epoch"},
            "cosine": {"eta_min", "last_epoch"},
            "cosine_with_restarts": {"T_0", "T_mult", "eta_min", "last_epoch"},
            "polynomial": {"power", "last_epoch"},
            "step": {"step_size", "gamma", "last_epoch"},
        }[scheduler_type]

        unknown = sorted(set(params) - allowed_params)
        if unknown:
            unknown_list = ", ".join(unknown)
            raise ValueError(f"Unknown scheduler_params for scheduler_type={scheduler_type!r}: {unknown_list}")

        if scheduler_type == "linear":
            cls._validate_number_param(params, "start_factor", gt=0.0, le=1.0)
            cls._validate_number_param(params, "end_factor", ge=0.0)
        elif scheduler_type == "cosine":
            cls._validate_number_param(params, "eta_min", ge=0.0)
        elif scheduler_type == "cosine_with_restarts":
            cls._validate_int_param(params, "T_0", gt=0)
            cls._validate_int_param(params, "T_mult", ge=1)
            cls._validate_number_param(params, "eta_min", ge=0.0)
        elif scheduler_type == "polynomial":
            cls._validate_number_param(params, "power", gt=0.0)
        elif scheduler_type == "step":
            cls._validate_int_param(params, "step_size", gt=0)
            cls._validate_number_param(params, "gamma", ge=0.0)

        cls._validate_int_param(params, "last_epoch", ge=-1)

    @staticmethod
    def _validate_number_param(
        params: Mapping[str, Any],
        name: str,
        *,
        gt: float | None = None,
        ge: float | None = None,
        le: float | None = None,
    ) -> None:
        if name not in params:
            return
        value = params[name]
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"scheduler_params.{name} must be a number")
        if gt is not None and value <= gt:
            raise ValueError(f"scheduler_params.{name} must be > {gt}")
        if ge is not None and value < ge:
            raise ValueError(f"scheduler_params.{name} must be >= {ge}")
        if le is not None and value > le:
            raise ValueError(f"scheduler_params.{name} must be <= {le}")

    @staticmethod
    def _validate_int_param(
        params: Mapping[str, Any],
        name: str,
        *,
        gt: int | None = None,
        ge: int | None = None,
    ) -> None:
        if name not in params:
            return
        value = params[name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"scheduler_params.{name} must be an integer")
        if gt is not None and value <= gt:
            raise ValueError(f"scheduler_params.{name} must be > {gt}")
        if ge is not None and value < ge:
            raise ValueError(f"scheduler_params.{name} must be >= {ge}")

    @classmethod
    def _preflight_model_paths(cls, trainer_config: LtxTrainerConfig) -> None:
        model_cfg = trainer_config.model
        if model_cfg.model_path is not None:
            cls._require_safetensors_file("model.model_path", model_cfg.model_path)

        component_paths = model_cfg.component_paths
        for name in (
            "transformer",
            "embeddings_processor",
            "video_vae",
            "video_vae_encoder",
            "video_vae_decoder",
            "audio_vae",
            "audio_vae_encoder",
            "audio_vae_decoder",
            "vocoder",
        ):
            path = getattr(component_paths, name)
            if path is not None:
                cls._require_safetensors_file(f"model.component_paths.{name}", path)

        text_encoder_path = component_paths.text_encoder or model_cfg.text_encoder_path
        if text_encoder_path is not None:
            text_encoder_path = cls._require_existing_file_or_dir("model.text_encoder_path", text_encoder_path)
        elif trainer_config.validation.prompts:
            raise ValueError("model.text_encoder_path or model.component_paths.text_encoder is required for validation")

        if trainer_config.acceleration.load_text_encoder_in_8bit:
            if text_encoder_path is None:
                raise ValueError("load_text_encoder_in_8bit requires a text encoder directory")
            if not text_encoder_path.is_dir():
                raise ValueError(
                    "load_text_encoder_in_8bit currently requires model.text_encoder_path to be a directory"
                )
            if importlib.util.find_spec("bitsandbytes") is None:
                raise ValueError("load_text_encoder_in_8bit requires bitsandbytes to be installed")

        training_strategy = get_training_strategy(trainer_config.training_strategy)
        need_vae_encoder = (
            trainer_config.validation.images is not None or trainer_config.validation.reference_videos is not None
        )
        load_audio = training_strategy.requires_audio or trainer_config.validation.generate_audio

        required_component_paths = [
            ("model.component_paths.transformer", cls._component_path_for_config(trainer_config, "transformer")),
            (
                "model.component_paths.embeddings_processor",
                cls._component_path_for_config(trainer_config, "embeddings_processor"),
            ),
            (
                "model.component_paths.video_vae_decoder",
                cls._video_vae_component_path_for_config(trainer_config, "decoder"),
            ),
        ]
        if need_vae_encoder:
            required_component_paths.append(
                (
                    "model.component_paths.video_vae_encoder",
                    cls._video_vae_component_path_for_config(trainer_config, "encoder"),
                )
            )
        if load_audio:
            required_component_paths.extend(
                [
                    (
                        "model.component_paths.audio_vae_decoder",
                        cls._audio_vae_component_path_for_config(trainer_config, "decoder"),
                    ),
                    ("model.component_paths.vocoder", cls._component_path_for_config(trainer_config, "vocoder")),
                ]
            )

        for label, path in required_component_paths:
            cls._require_safetensors_file(label, path)

    @classmethod
    def _preflight_dataset(cls, trainer_config: LtxTrainerConfig) -> None:
        training_strategy = get_training_strategy(trainer_config.training_strategy)
        data_sources = training_strategy.get_data_sources()
        optional_data_sources = set(training_strategy.get_optional_data_sources())

        if cls._config_nsync_requires_negatives(trainer_config):
            if isinstance(data_sources, list):
                data_sources = {source: source for source in data_sources}
            data_sources[trainer_config.lora.nsync.negative_latents_dir] = "negative_latents"

        dataset = PrecomputedDataset(
            trainer_config.data.preprocessed_data_root,
            data_sources=data_sources,
            optional_data_sources=optional_data_sources,
        )

        if cls._config_nsync_requires_anchor(trainer_config) and len(dataset) < 2:
            raise ValueError("NSYNC anchor training requires at least two positive samples")

        if trainer_config.data.validation_data_root is not None:
            PrecomputedDataset(
                trainer_config.data.validation_data_root,
                data_sources=training_strategy.get_data_sources(),
                optional_data_sources=set(training_strategy.get_optional_data_sources()),
            )

    @staticmethod
    def _preflight_output_dir(trainer_config: LtxTrainerConfig) -> None:
        output_dir = Path(trainer_config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if not output_dir.is_dir():
            raise ValueError(f"output_dir is not a directory: {output_dir}")

        probe_path = output_dir / ".ltx_preflight_write_test"
        try:
            probe_path.write_text("ok", encoding="utf-8")
        finally:
            if probe_path.exists():
                probe_path.unlink()

    @classmethod
    def _preflight_checkpoint(cls, trainer_config: LtxTrainerConfig) -> Path | None:
        if not trainer_config.model.load_checkpoint:
            return None

        checkpoint_path = cls._find_checkpoint(trainer_config.model.load_checkpoint)
        if checkpoint_path is None:
            logger.warning(f"⚠️ Could not find checkpoint at {trainer_config.model.load_checkpoint}")
        return checkpoint_path

    @staticmethod
    def _expand_path(path: str | Path) -> Path:
        return Path(path).expanduser()

    @classmethod
    def _require_existing_file_or_dir(cls, label: str, path: str | Path) -> Path:
        resolved = cls._expand_path(path)
        if not resolved.exists():
            raise FileNotFoundError(f"{label} does not exist: {resolved}")
        if not resolved.is_file() and not resolved.is_dir():
            raise ValueError(f"{label} must be a file or directory: {resolved}")
        return resolved

    @classmethod
    def _require_safetensors_file(cls, label: str, path: str | Path) -> Path:
        resolved = cls._expand_path(path)
        if not resolved.exists():
            raise FileNotFoundError(f"{label} does not exist: {resolved}")
        if not resolved.is_file():
            raise ValueError(f"{label} must be a .safetensors file: {resolved}")
        if resolved.suffix != ".safetensors":
            raise ValueError(f"{label} must have a .safetensors extension: {resolved}")
        return resolved

    @staticmethod
    def _component_path_for_config(trainer_config: LtxTrainerConfig, name: str) -> str | Path:
        path = getattr(trainer_config.model.component_paths, name)
        if path is not None:
            return path
        if trainer_config.model.model_path is not None:
            return trainer_config.model.model_path
        raise ValueError(f"model.component_paths.{name} or model.model_path must be provided")

    @staticmethod
    def _video_vae_component_path_for_config(
        trainer_config: LtxTrainerConfig,
        direction: Literal["encoder", "decoder"],
    ) -> str | Path:
        paths = trainer_config.model.component_paths
        specific = paths.video_vae_encoder if direction == "encoder" else paths.video_vae_decoder
        path = specific or paths.video_vae
        if path is not None:
            return path
        if trainer_config.model.model_path is not None:
            return trainer_config.model.model_path
        raise ValueError(
            f"model.component_paths.video_vae_{direction}, model.component_paths.video_vae, "
            "or model.model_path must be provided"
        )

    @staticmethod
    def _audio_vae_component_path_for_config(
        trainer_config: LtxTrainerConfig,
        direction: Literal["encoder", "decoder"],
    ) -> str | Path:
        paths = trainer_config.model.component_paths
        specific = paths.audio_vae_encoder if direction == "encoder" else paths.audio_vae_decoder
        path = specific or paths.audio_vae
        if path is not None:
            return path
        if trainer_config.model.model_path is not None:
            return trainer_config.model.model_path
        raise ValueError(
            f"model.component_paths.audio_vae_{direction}, model.component_paths.audio_vae, "
            "or model.model_path must be provided"
        )

    @staticmethod
    def _config_nsync_enabled(trainer_config: LtxTrainerConfig) -> bool:
        return trainer_config.lora is not None and trainer_config.lora.nsync.enabled

    @classmethod
    def _config_nsync_requires_negatives(cls, trainer_config: LtxTrainerConfig) -> bool:
        return cls._config_nsync_enabled(trainer_config) and trainer_config.lora.nsync.negative_strength > 0.0

    @classmethod
    def _config_nsync_requires_anchor(cls, trainer_config: LtxTrainerConfig) -> bool:
        return cls._config_nsync_enabled(trainer_config) and trainer_config.lora.nsync.anchor_strength > 0.0

    def __init__(
        self,
        trainer_config: LtxTrainerConfig,
        preflight_result: TrainerPreflightResult | None = None,
        model_pool: WarmModelPool | None = None,
    ) -> None:
        self._config = trainer_config
        self._preflight_result = preflight_result
        self._model_pool = model_pool
        self._warm_transformer_key: ModelCacheKey | None = None
        self._warm_embeddings_key: ModelCacheKey | None = None
        self._training_ended = False
        self._released = False
        self._loaded_checkpoint_path: Path | None = None
        self._resolve_lora_rank_from_checkpoint()
        if IS_MAIN_PROCESS:
            print_config(self._config)
        self._training_strategy = get_training_strategy(self._config.training_strategy)
        self._cached_validation_embeddings = self._load_text_encoder_and_cache_embeddings()
        self._load_models()
        self._setup_accelerator()
        self._collect_trainable_params()
        self._load_checkpoint()
        self._prepare_models_for_training()
        self._dataset = None
        self._validation_dataset = None
        self._validation_dataloader = None
        self._global_step = -1
        self._checkpoint_paths: list[Path] = []
        self._training_state_paths: list[Path] = []
        self._training_state_size_warned = False
        self._wandb_run = None
        self._tensorboard_writer: SummaryWriter | None = None
        self._sigma_tracker = SigmaBucketTracker()
        self._noise_experts: list[NoiseExpert] = []
        self._active_noise_expert: NoiseExpert | None = None

    def train(  # noqa: PLR0912, PLR0915
        self,
        disable_progress_bars: bool = False,
        step_callback: StepCallback | None = None,
    ) -> tuple[Path, TrainingStats]:
        """
        Start the training process.
        Args:
            disable_progress_bars: Disable Rich progress bars (useful for multi-process runs).
            step_callback: Optional callback invoked after each optimization step.
        Returns:
            Tuple of (saved_model_path, training_stats)
        """
        device = self._accelerator.device
        cfg = self._config
        start_mem = get_gpu_memory_gb(device)

        train_start_time = time.time()

        initial_step, training_state = self._resume_state
        resuming = training_state is not None

        set_seed(cfg.seed)
        logger.debug(f"Process {self._accelerator.process_index} using seed: {cfg.seed}")

        self._init_optimizer()

        if training_state is not None and not self._restore_training_state(training_state):
            initial_step = 0
            resuming = False

        # Initialize loggers after restore so we only resume W&B when state restore succeeds.
        resume_run_id = training_state.wandb_run_id if resuming and training_state is not None else None
        self._init_wandb(resume_run_id=resume_run_id)
        self._init_tensorboard()

        self._init_dataloader()
        data_iter = iter(self._dataloader)
        self._init_timestep_sampler()
        self._init_noise_experts()

        # Synchronize all processes after initialization
        self._accelerator.wait_for_everyone()

        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

        # Save the training configuration as YAML
        self._save_config()

        remaining_steps = cfg.optimization.steps - initial_step
        if remaining_steps <= 0:
            raise ValueError(
                f"No remaining training steps: initial_step={initial_step} >= "
                f"target_steps={cfg.optimization.steps}. Nothing to train."
            )

        if resuming:
            logger.info(f"🚀 Resuming training from step {initial_step} → {cfg.optimization.steps}")
        else:
            logger.info("🚀 Starting training...")

        # Create progress tracking (disabled for non-main processes or when explicitly disabled)
        progress_enabled = IS_MAIN_PROCESS and not disable_progress_bars
        progress = TrainingProgress(
            enabled=progress_enabled,
            total_steps=remaining_steps,
        )

        if IS_MAIN_PROCESS and disable_progress_bars:
            logger.warning("Progress bars disabled. Intermediate status messages will be logged instead.")

        self._transformer.train()
        self._global_step = initial_step

        peak_mem_during_training = start_mem

        sampled_videos_paths = None

        with progress:
            if cfg.validation.loss_interval and not cfg.validation.skip_initial_validation:
                self._run_validation_loss()

            if cfg.validation.interval and not cfg.validation.skip_initial_validation:
                with self._offloaded_optimizer_state():
                    sampled_videos_paths = self._run_distributed_validation(progress)

            self._accelerator.wait_for_everyone()

            for step in range(remaining_steps * cfg.optimization.gradient_accumulation_steps):
                # Get next batch, reset the dataloader if needed
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self._dataloader)
                    batch = next(data_iter)

                step_start_time = time.time()
                with self._accelerator.accumulate(self._transformer):
                    is_optimization_step = (step + 1) % cfg.optimization.gradient_accumulation_steps == 0
                    if is_optimization_step:
                        self._global_step += 1

                    expert = self._select_noise_expert(step)
                    self._set_active_lora_adapter(expert.name)

                    if self._nsync_enabled:
                        output = self._nsync_training_step(batch, expert.sampler)
                    else:
                        output = self._training_step(batch, expert.sampler)
                        self._accelerator.backward(output.loss.mean())

                    if self._accelerator.sync_gradients and cfg.optimization.max_grad_norm > 0:
                        self._accelerator.clip_grad_norm_(
                            self._trainable_params,
                            cfg.optimization.max_grad_norm,
                        )

                    self._optimizer.step()
                    self._optimizer.zero_grad()

                    if self._lr_scheduler is not None:
                        self._lr_scheduler.step()

                    # Evaluate held-out loss if needed
                    if (
                        cfg.validation.loss_interval
                        and self._global_step > 0
                        and self._global_step % cfg.validation.loss_interval == 0
                        and is_optimization_step
                    ):
                        self._run_validation_loss()

                    # Generate validation samples if needed
                    if (
                        cfg.validation.interval
                        and self._global_step > 0
                        and self._global_step % cfg.validation.interval == 0
                        and is_optimization_step
                    ):
                        with self._offloaded_optimizer_state():
                            sampled_videos_paths = self._run_distributed_validation(progress)

                    # Save checkpoint if needed
                    if (
                        cfg.checkpoints.interval
                        and self._global_step > 0
                        and self._global_step % cfg.checkpoints.interval == 0
                        and is_optimization_step
                    ):
                        self._save_checkpoint()

                    self._accelerator.wait_for_everyone()

                    # Call step callback if provided
                    if step_callback and is_optimization_step:
                        step_callback(self._global_step, cfg.optimization.steps, sampled_videos_paths)

                    self._accelerator.wait_for_everyone()

                    # Update progress and log metrics
                    current_lr = self._optimizer.param_groups[0]["lr"]
                    step_time = (time.time() - step_start_time) * cfg.optimization.gradient_accumulation_steps
                    step_loss = output.loss.detach().mean().item()

                    progress.update_training(
                        loss=step_loss,
                        lr=current_lr,
                        step_time=step_time,
                        advance=is_optimization_step,
                    )

                    # Log metrics (only on main process and optimization steps)
                    if IS_MAIN_PROCESS and is_optimization_step:
                        # Track per-element loss by sigma bucket
                        self._sigma_tracker.update(output.sigma.cpu().tolist(), output.loss.detach().cpu().tolist())
                        metrics = {
                            "train/loss": step_loss,
                            "train/learning_rate": current_lr,
                            "train/step_time": step_time,
                            "train/global_step": self._global_step,
                        }
                        metrics.update(self._sigma_tracker.get_metrics())
                        self._log_metrics(metrics)

                    # Fallback logging when progress bars are disabled
                    if disable_progress_bars and IS_MAIN_PROCESS and self._global_step % 20 == 0:
                        elapsed = time.time() - train_start_time
                        steps_done = self._global_step - initial_step
                        if steps_done > 0:
                            total_estimated = elapsed / steps_done * remaining_steps
                            total_time = f"{total_estimated // 3600:.0f}h {(total_estimated % 3600) // 60:.0f}m"
                        else:
                            total_time = "calculating..."
                        logger.info(
                            f"Step {self._global_step}/{cfg.optimization.steps} - "
                            f"Loss: {step_loss:.4f}, LR: {current_lr:.2e}, "
                            f"Time/Step: {step_time:.2f}s, Total Time: {total_time}",
                        )

                    # Sample GPU memory periodically
                    if step % MEMORY_CHECK_INTERVAL == 0:
                        current_mem = get_gpu_memory_gb(device)
                        peak_mem_during_training = max(peak_mem_during_training, current_mem)

        # Collect final stats
        train_end_time = time.time()
        end_mem = get_gpu_memory_gb(device)
        peak_mem = max(start_mem, end_mem, peak_mem_during_training)

        # Calculate steps/second over entire training
        total_time_seconds = train_end_time - train_start_time
        steps_per_second = remaining_steps / total_time_seconds

        samples_per_second = steps_per_second * self._accelerator.num_processes * cfg.optimization.batch_size

        stats = TrainingStats(
            total_time_seconds=total_time_seconds,
            steps_per_second=steps_per_second,
            samples_per_second=samples_per_second,
            peak_gpu_memory_gb=peak_mem,
            num_processes=self._accelerator.num_processes,
            global_batch_size=cfg.optimization.batch_size * self._accelerator.num_processes,
        )

        saved_path = self._save_checkpoint()

        if IS_MAIN_PROCESS:
            # Log the training statistics
            self._log_training_stats(stats)

            # Upload artifacts to hub if enabled
            if cfg.hub.push_to_hub:
                push_to_hub(saved_path, sampled_videos_paths, self._config)

            # Log final stats and close experiment loggers.
            self._log_metrics(
                {
                    "stats/total_time_minutes": stats.total_time_seconds / 60,
                    "stats/steps_per_second": stats.steps_per_second,
                    "stats/samples_per_second": stats.samples_per_second,
                    "stats/peak_gpu_memory_gb": stats.peak_gpu_memory_gb,
                }
            )
            if self._wandb_run is not None:
                self._wandb_run.finish()
            if self._tensorboard_writer is not None:
                self._tensorboard_writer.close()
                self._tensorboard_writer = None

        self._accelerator.wait_for_everyone()
        self._accelerator.end_training()
        self._training_ended = True

        return saved_path, stats

    def release_warm_models(self) -> None:
        """Return a pooled LoRA base model to a clean, reusable state.

        Per-run objects (PEFT adapters, optimizer, scheduler, dataloaders and
        Accelerate wrappers) are discarded. Frozen base weights and inference
        components remain owned by the warm model pool.
        """
        if self._released or self._model_pool is None or self._warm_transformer_key is None:
            return

        accelerator = getattr(self, "_accelerator", None)
        wrapped_transformer = getattr(self, "_transformer", None)
        if wrapped_transformer is None:
            return

        unwrapped = (
            accelerator.unwrap_model(wrapped_transformer, keep_torch_compile=False)
            if accelerator is not None
            else wrapped_transformer
        )

        transient_objects = [wrapped_transformer]
        for name in ("_optimizer", "_lr_scheduler", "_dataloader", "_validation_dataloader"):
            value = getattr(self, name, None)
            if value is not None:
                transient_objects.append(value)
        if accelerator is not None:
            accelerator.free_memory(*transient_objects)

        # PeftModel.base_model is the tuner object; unload() removes all injected
        # LoRA layers without merging their trained values into the frozen base.
        tuner = getattr(unwrapped, "base_model", None)
        base_transformer = tuner.unload() if tuner is not None and hasattr(tuner, "unload") else unwrapped

        base_transformer.requires_grad_(False)
        base_transformer.eval()
        if hasattr(base_transformer, "set_gradient_checkpointing"):
            base_transformer.set_gradient_checkpointing(False)
        for parameter in base_transformer.parameters():
            parameter.grad = None

        self._transformer = base_transformer
        self._model_pool.replace(self._warm_transformer_key, base_transformer)
        for name in ("_optimizer", "_lr_scheduler", "_dataloader", "_validation_dataloader"):
            if hasattr(self, name):
                setattr(self, name, None)

        if accelerator is not None and not self._training_ended:
            with contextlib.suppress(Exception):
                accelerator.end_training()
            self._training_ended = True

        free_gpu_memory()
        self._released = True

    @property
    def _nsync_enabled(self) -> bool:
        return self._config.lora is not None and self._config.lora.nsync.enabled

    @property
    def _nsync_negative_strength(self) -> float:
        if self._config.lora is None:
            return 0.0
        return self._config.lora.nsync.negative_strength

    @property
    def _nsync_anchor_strength(self) -> float:
        if self._config.lora is None:
            return 0.0
        return getattr(self._config.lora.nsync, "anchor_strength", 0.0)

    @property
    def _nsync_requires_negatives(self) -> bool:
        return self._nsync_enabled and self._nsync_negative_strength > 0.0

    @property
    def _nsync_requires_anchor(self) -> bool:
        return self._nsync_enabled and self._nsync_anchor_strength > 0.0

    def _training_step(
        self,
        batch: dict[str, dict[str, Tensor]],
        timestep_sampler: TimestepSampler | None = None,
    ) -> TrainingStepOutput:
        """Perform a single training step using the configured strategy."""
        timestep_sampler = timestep_sampler or self._timestep_sampler

        # Apply embedding connectors to transform pre-computed text embeddings
        conditions = batch["conditions"]

        if not conditions.get("_embeddings_processed", False):
            if "video_prompt_embeds" in conditions:
                # New format: separate video/audio features from precompute()
                video_features = conditions["video_prompt_embeds"]
                audio_features = conditions.get("audio_prompt_embeds")
            else:
                # Legacy format: single prompt_embeds tensor — duplicate for both modalities
                video_features = conditions["prompt_embeds"]
                audio_features = conditions["prompt_embeds"]

            with torch.no_grad():
                mask = conditions["prompt_attention_mask"]
                additive_mask = convert_to_additive_mask(mask, video_features.dtype)
                video_embeds, audio_embeds, attention_mask = self._embeddings_processor.create_embeddings(
                    video_features, audio_features, additive_mask
                )

            conditions["video_prompt_embeds"] = video_embeds.detach()
            conditions["audio_prompt_embeds"] = audio_embeds.detach() if audio_embeds is not None else None
            conditions["prompt_attention_mask"] = attention_mask.detach()
            conditions["_embeddings_processed"] = True

        # Use strategy to prepare training inputs (returns ModelInputs with Modality objects)
        model_inputs = self._training_strategy.prepare_training_inputs(batch, timestep_sampler)

        # Run transformer forward pass with Modality-based interface
        video_pred, audio_pred = self._transformer(
            video=model_inputs.video,
            audio=model_inputs.audio,
            perturbations=None,
        )

        # Use strategy to compute loss
        loss = self._training_strategy.compute_loss(video_pred, audio_pred, model_inputs)
        sigma = model_inputs.video.sigma.detach() if model_inputs.video.enabled else model_inputs.audio.sigma.detach()

        return TrainingStepOutput(loss=loss, sigma=sigma)

    def _nsync_training_step(
        self,
        batch: dict[str, dict[str, Tensor]],
        timestep_sampler: TimestepSampler,
    ) -> TrainingStepOutput:
        """Run NSYNC positive/negative gradient projection for one micro-batch."""
        if self._nsync_negative_strength <= 0.0 and self._nsync_anchor_strength <= 0.0:
            positive_output = self._training_step(batch, timestep_sampler)
            self._accelerator.backward(positive_output.loss.mean())
            return positive_output

        accumulated_grads = self._clone_current_grads()
        negative_grads: dict[torch.nn.Parameter, Tensor] = {}
        anchor_grads: dict[torch.nn.Parameter, Tensor] = {}

        if self._nsync_negative_strength > 0.0:
            if batch.get("negative_latents") is None:
                raise ValueError("NSYNC is enabled, but the batch does not contain negative_latents")

            self._clear_trainable_grads()
            negative_batch = dict(batch)
            negative_batch["latents"] = batch["negative_latents"]
            negative_batch["_force_video_only"] = True
            negative_output = self._training_step(negative_batch, timestep_sampler)
            self._accelerator.backward(negative_output.loss.mean())
            negative_grads = self._clone_current_grads()

        if self._nsync_anchor_strength > 0.0:
            if batch.get("anchor_latents") is None or batch.get("anchor_conditions") is None:
                raise ValueError("NSYNC anchor training is enabled, but the batch does not contain anchor data")

            self._clear_trainable_grads()
            anchor_batch = dict(batch)
            anchor_batch["latents"] = batch["anchor_latents"]
            anchor_batch["conditions"] = batch["anchor_conditions"]
            anchor_batch["_force_video_only"] = True
            anchor_output = self._training_step(anchor_batch, timestep_sampler)
            self._accelerator.backward(anchor_output.loss.mean())
            anchor_grads = self._clone_current_grads()

        self._clear_trainable_grads()
        positive_output = self._training_step(batch, timestep_sampler)
        self._accelerator.backward(positive_output.loss.mean())
        self._apply_nsync_projection(negative_grads, anchor_grads)
        self._add_grads(accumulated_grads)

        return positive_output

    def _clone_current_grads(self) -> dict[torch.nn.Parameter, Tensor]:
        """Clone current trainable gradients, preserving only parameters with gradients."""
        return {param: param.grad.detach().clone() for param in self._trainable_params if param.grad is not None}

    def _clear_trainable_grads(self) -> None:
        """Clear gradients for trainable parameters only."""
        for param in self._trainable_params:
            param.grad = None

    def _add_grads(self, grads: dict[torch.nn.Parameter, Tensor]) -> None:
        """Add previously accumulated gradients back after NSYNC projection."""
        for param, grad in grads.items():
            if param.grad is None:
                param.grad = grad
            else:
                param.grad.add_(grad.to(device=param.grad.device, dtype=param.grad.dtype))

    def _apply_nsync_projection(
        self,
        negative_grads: dict[torch.nn.Parameter, Tensor],
        anchor_grads: dict[torch.nn.Parameter, Tensor] | None = None,
    ) -> None:
        """Refine positive gradients with negative and anchor projections per configured group."""
        eps = self._config.lora.nsync.eps
        negative_strength = self._nsync_negative_strength
        anchor_strength = self._nsync_anchor_strength if anchor_grads else 0.0
        anchor_grads = anchor_grads or {}
        if negative_strength <= 0.0 and anchor_strength <= 0.0:
            return

        for params in self._gradient_projection_groups.values():
            negative_scale = (
                self._nsync_projection_scale(params, negative_grads, eps) if negative_strength > 0.0 else None
            )
            anchor_scale = self._nsync_projection_scale(params, anchor_grads, eps) if anchor_strength > 0.0 else None
            if negative_scale is None and anchor_scale is None:
                continue

            for param in params:
                if param.grad is None:
                    continue
                if negative_scale is not None and param in negative_grads:
                    neg_grad = negative_grads[param].to(device=param.grad.device, dtype=param.grad.dtype)
                    projection = negative_scale.to(device=param.grad.device, dtype=param.grad.dtype) * neg_grad
                    param.grad.sub_(negative_strength * projection)
                if anchor_scale is not None and param in anchor_grads:
                    anchor_grad = anchor_grads[param].to(device=param.grad.device, dtype=param.grad.dtype)
                    projection = anchor_scale.to(device=param.grad.device, dtype=param.grad.dtype) * anchor_grad
                    param.grad.add_(anchor_strength * projection)

    @staticmethod
    def _nsync_projection_scale(
        params: list[torch.nn.Parameter],
        reference_grads: dict[torch.nn.Parameter, Tensor],
        eps: float,
    ) -> Tensor | None:
        """Return the group projection scale of positive gradients onto reference gradients."""
        dot = None
        denom = None
        for param in params:
            if param.grad is None or param not in reference_grads:
                continue
            pos_grad = param.grad.detach().float()
            ref_grad = reference_grads[param].to(device=param.grad.device).float()
            group_dot = torch.sum(pos_grad * ref_grad)
            group_denom = torch.sum(ref_grad * ref_grad)
            dot = group_dot if dot is None else dot + group_dot
            denom = group_denom if denom is None else denom + group_denom

        if dot is None or denom is None or denom.item() <= 0.0:
            return None

        return dot / (denom + eps)

    @property
    def _text_encoder_path(self) -> str | Path:
        """Resolve Gemma root with component-path override support."""
        path = self._config.model.component_paths.text_encoder or self._config.model.text_encoder_path
        if path is None:
            raise ValueError("text_encoder_path or model.component_paths.text_encoder must be provided")
        return path

    def _component_path(self, name: str) -> str | Path:
        """Resolve a component checkpoint path, falling back to the base model path."""
        path = getattr(self._config.model.component_paths, name)
        if path is not None:
            return path
        if self._config.model.model_path is not None:
            return self._config.model.model_path
        raise ValueError(f"model.component_paths.{name} or model.model_path must be provided")

    def _video_vae_component_path(self, direction: Literal["encoder", "decoder"]) -> str | Path:
        """Resolve video VAE encoder/decoder checkpoint paths."""
        paths = self._config.model.component_paths
        specific = paths.video_vae_encoder if direction == "encoder" else paths.video_vae_decoder
        path = specific or paths.video_vae
        if path is not None:
            return path
        if self._config.model.model_path is not None:
            return self._config.model.model_path
        raise ValueError(
            f"model.component_paths.video_vae_{direction}, model.component_paths.video_vae, "
            "or model.model_path must be provided"
        )

    def _audio_vae_component_path(self, direction: Literal["encoder", "decoder"]) -> str | Path:
        """Resolve audio VAE encoder/decoder checkpoint paths."""
        paths = self._config.model.component_paths
        specific = paths.audio_vae_encoder if direction == "encoder" else paths.audio_vae_decoder
        path = specific or paths.audio_vae
        if path is not None:
            return path
        if self._config.model.model_path is not None:
            return self._config.model.model_path
        raise ValueError(
            f"model.component_paths.audio_vae_{direction}, model.component_paths.audio_vae, "
            "or model.model_path must be provided"
        )

    @free_gpu_memory_context(after=True)
    def _load_text_encoder_and_cache_embeddings(self) -> list[CachedPromptEmbeddings] | None:  # noqa: PLR0912
        """Load text encoder + embeddings processor, compute and cache validation embeddings."""

        # This method:
        #   1. Loads the embeddings processor (feature extractor + connectors)
        #   2. Loads the pure Gemma text encoder only when validation prompts need caching
        #   3. Computes and caches validation embeddings when prompts are configured
        #   4. Unloads Gemma/feature extractor, keeps the connectors for training

        # LOCAL_RANK before Accelerator exists
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        init_device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

        # Load embeddings processor (feature extractor + connectors)
        logger.debug("Loading embeddings processor...")
        embeddings_path = self._component_path("embeddings_processor")
        text_key = None
        if self._model_pool is None:
            self._embeddings_processor = load_embeddings_processor(
                checkpoint_path=embeddings_path,
                device=init_device,
                dtype=torch.bfloat16,
            )
        else:
            self._warm_embeddings_key = ModelCacheKey.create(
                "embeddings_processor",
                embeddings_path,
                torch.bfloat16,
            )
            active_keys = {self._warm_embeddings_key}
            if self._config.validation.prompts:
                text_key = ModelCacheKey.create(
                    "text_encoder",
                    self._text_encoder_path,
                    torch.bfloat16,
                    load_in_8bit=self._config.acceleration.load_text_encoder_in_8bit,
                )
                active_keys.add(text_key)
            # Gemma needs substantial VRAM, so clear other cached components only
            # when prompts actually require it. With no prompts, this preserves a
            # transformer already warm on GPU between consecutive LoRA jobs.
            if self._config.validation.prompts:
                self._model_pool.offload_all(exclude=active_keys)
            self._embeddings_processor = self._model_pool.get_or_load(
                self._warm_embeddings_key,
                lambda target: load_embeddings_processor(
                    checkpoint_path=embeddings_path,
                    device=target,
                    dtype=torch.bfloat16,
                ),
                init_device,
            )

        # Cache validation embeddings if prompts are configured
        cached_embeddings = None
        if self._config.validation.prompts:
            logger.debug("Loading text encoder...")
            if self._model_pool is None:
                text_encoder = load_text_encoder(
                    gemma_model_path=self._text_encoder_path,
                    device=init_device,
                    dtype=torch.bfloat16,
                    load_in_8bit=self._config.acceleration.load_text_encoder_in_8bit,
                )
            else:
                text_encoder = self._model_pool.get_or_load(
                    text_key,
                    lambda target: load_text_encoder(
                        gemma_model_path=self._text_encoder_path,
                        device=target,
                        dtype=torch.bfloat16,
                        load_in_8bit=self._config.acceleration.load_text_encoder_in_8bit,
                    ),
                    init_device,
                    offloadable=not self._config.acceleration.load_text_encoder_in_8bit,
                )
            logger.info(f"Pre-computing embeddings for {len(self._config.validation.prompts)} validation prompts...")
            cached_embeddings = []
            with torch.inference_mode():
                for prompt in self._config.validation.prompts:
                    pos_hs, pos_mask = text_encoder.encode(prompt)
                    pos_out = self._embeddings_processor.process_hidden_states(pos_hs, pos_mask)

                    neg_hs, neg_mask = text_encoder.encode(self._config.validation.negative_prompt)
                    neg_out = self._embeddings_processor.process_hidden_states(neg_hs, neg_mask)

                    cached_embeddings.append(
                        CachedPromptEmbeddings(
                            video_context_positive=pos_out.video_encoding.cpu(),
                            audio_context_positive=pos_out.audio_encoding.cpu(),
                            video_context_negative=neg_out.video_encoding.cpu(),
                            audio_context_negative=(
                                neg_out.audio_encoding.cpu() if neg_out.audio_encoding is not None else None
                            ),
                        )
                    )
            del text_encoder
            if self._model_pool is not None:
                self._model_pool.offload(text_key)

        # Unload feature extractor, keep only connectors for training
        if self._model_pool is None:
            self._embeddings_processor.feature_extractor = None
        elif self._embeddings_processor.feature_extractor is not None:
            # Preserve it for later preprocessing/validation jobs while keeping
            # only the small connector modules resident on the training device.
            self._embeddings_processor.feature_extractor.to("cpu")
        self._embeddings_processor.requires_grad_(False)
        self._embeddings_processor.eval()

        if cached_embeddings is None:
            logger.debug("No validation prompts configured. Gemma text encoder load skipped")
        else:
            logger.debug("Validation prompt embeddings cached. Gemma model unloaded")
        return cached_embeddings

    def _load_models(self) -> None:
        """Load the LTX-2 model components."""
        # Load audio components if:
        # 1. Training strategy requires audio (training the audio branch), OR
        # 2. Validation is configured to generate audio (even if not training audio)
        load_audio = self._training_strategy.requires_audio or self._config.validation.generate_audio

        # Check if we need VAE encoder (for image or reference video conditioning)
        need_vae_encoder = (
            self._config.validation.images is not None or self._config.validation.reference_videos is not None
        )

        if self._model_pool is not None and self._config.model.training_mode == "lora":
            self._load_warm_lora_models(load_audio=load_audio, need_vae_encoder=need_vae_encoder)
            return

        # Load all model components (except text encoder - already handled)
        video_vae_encoder_path = self._video_vae_component_path("encoder") if need_vae_encoder else None
        audio_vae_decoder_path = self._audio_vae_component_path("decoder") if load_audio else None
        vocoder_path = self._component_path("vocoder") if load_audio else None
        components = load_ltx_model(
            checkpoint_path=self._config.model.model_path,
            device="cpu",
            dtype=torch.bfloat16,
            with_video_vae_encoder=need_vae_encoder,  # Needed for image conditioning
            with_video_vae_decoder=True,  # Needed for validation sampling
            with_audio_vae_decoder=load_audio,
            with_vocoder=load_audio,
            with_text_encoder=False,  # Text encoder handled separately
            transformer_path=self._component_path("transformer"),
            video_vae_encoder_path=video_vae_encoder_path,
            video_vae_decoder_path=self._video_vae_component_path("decoder"),
            audio_vae_decoder_path=audio_vae_decoder_path,
            vocoder_path=vocoder_path,
        )

        # Extract components
        self._transformer = components.transformer
        self._vae_decoder = components.video_vae_decoder.to(dtype=torch.bfloat16)
        self._vae_encoder = components.video_vae_encoder
        if self._vae_encoder is not None:
            self._vae_encoder = self._vae_encoder.to(dtype=torch.bfloat16)
        self._scheduler = components.scheduler
        self._audio_vae = components.audio_vae_decoder
        self._vocoder = components.vocoder
        # Note: self._embeddings_processor was set in _load_text_encoder_and_cache_embeddings

        # Determine initial dtype based on training mode.
        # Note: For FSDP + LoRA, we'll cast to FP32 later in _prepare_models_for_training()
        # after the accelerator is set up, and we can detect FSDP.
        transformer_dtype = torch.bfloat16 if self._config.model.training_mode == "lora" else torch.float32
        self._transformer = self._transformer.to(dtype=transformer_dtype)

        if self._config.acceleration.quantization is not None:
            if self._config.model.training_mode == "full":
                raise ValueError("Quantization is not supported in full training mode.")

            logger.info(f'Quantizing model with "{self._config.acceleration.quantization}". This may take a while...')
            self._transformer = quantize_model(
                self._transformer,
                precision=self._config.acceleration.quantization,
            )

        # Freeze all models. We later unfreeze the transformer based on training mode.
        # Note: embedding_connectors are already frozen (they come from the frozen text encoder)
        self._vae_decoder.requires_grad_(False)
        if self._vae_encoder is not None:
            self._vae_encoder.requires_grad_(False)
        self._transformer.requires_grad_(False)
        if self._audio_vae is not None:
            self._audio_vae.requires_grad_(False)
        if self._vocoder is not None:
            self._vocoder.requires_grad_(False)

    def _load_warm_lora_models(self, *, load_audio: bool, need_vae_encoder: bool) -> None:
        """Load or reuse frozen components for a compatible LoRA training job."""
        pool = self._model_pool
        transformer_path = self._component_path("transformer")
        quantization = self._config.acceleration.quantization
        self._warm_transformer_key = ModelCacheKey.create(
            "training_transformer",
            transformer_path,
            torch.bfloat16,
            quantization=quantization or "none",
        )

        def _load_base_transformer(target: torch.device) -> torch.nn.Module:
            transformer = load_transformer(transformer_path, device=target, dtype=torch.bfloat16)
            if quantization is not None:
                logger.info(f'Quantizing model with "{quantization}". This may take a while...')
                transformer = quantize_model(transformer, precision=quantization)
            return transformer

        # A cache hit deliberately retains its current device. This avoids a needless
        # GPU -> CPU -> GPU round trip between consecutive LoRA jobs.
        self._transformer = pool.get_or_load(
            self._warm_transformer_key,
            _load_base_transformer,
            "cpu",
            move_cached=False,
        )

        decoder_path = self._video_vae_component_path("decoder")
        decoder_key = ModelCacheKey.create("video_vae_decoder", decoder_path, torch.bfloat16)
        self._vae_decoder = pool.get_or_load(
            decoder_key,
            lambda target: load_video_vae_decoder(decoder_path, device=target, dtype=torch.bfloat16),
            "cpu",
        )

        self._vae_encoder = None
        if need_vae_encoder:
            encoder_path = self._video_vae_component_path("encoder")
            encoder_key = ModelCacheKey.create("video_vae_encoder", encoder_path, torch.bfloat16)
            self._vae_encoder = pool.get_or_load(
                encoder_key,
                lambda target: load_video_vae_encoder(encoder_path, device=target, dtype=torch.bfloat16),
                "cpu",
            )

        self._audio_vae = None
        self._vocoder = None
        if load_audio:
            audio_path = self._audio_vae_component_path("decoder")
            audio_key = ModelCacheKey.create("audio_vae_decoder", audio_path, torch.bfloat16)
            self._audio_vae = pool.get_or_load(
                audio_key,
                lambda target: load_audio_vae_decoder(audio_path, device=target, dtype=torch.bfloat16),
                "cpu",
            )
            vocoder_path = self._component_path("vocoder")
            vocoder_key = ModelCacheKey.create("vocoder", vocoder_path, torch.bfloat16)
            self._vocoder = pool.get_or_load(
                vocoder_key,
                lambda target: load_vocoder(vocoder_path, device=target, dtype=torch.bfloat16),
                "cpu",
            )

        self._scheduler = LTX2Scheduler()
        self._vae_decoder.requires_grad_(False)
        if self._vae_encoder is not None:
            self._vae_encoder.requires_grad_(False)
        self._transformer.requires_grad_(False)
        if self._audio_vae is not None:
            self._audio_vae.requires_grad_(False)
        if self._vocoder is not None:
            self._vocoder.requires_grad_(False)

    def _collect_trainable_params(self) -> None:
        """Collect trainable parameters based on training mode."""
        if self._config.model.training_mode == "lora":
            # For LoRA training, first set up LoRA layers
            self._setup_lora()
        elif self._config.model.training_mode == "full":
            # For full training, unfreeze all transformer parameters
            self._transformer.requires_grad_(True)
        else:
            raise ValueError(f"Unknown training mode: {self._config.model.training_mode}")

        self._trainable_params = [p for p in self._transformer.parameters() if p.requires_grad]
        self._trainable_param_names = {
            param: name for name, param in self._transformer.named_parameters() if param.requires_grad
        }
        self._gradient_projection_groups = self._build_gradient_projection_groups()
        logger.debug(f"Trainable params count: {sum(p.numel() for p in self._trainable_params):,}")

    def _init_timestep_sampler(self) -> None:
        """Initialize the timestep sampler based on the config."""
        sampler_cls = SAMPLERS[self._config.flow_matching.timestep_sampling_mode]
        self._timestep_sampler = sampler_cls(**self._config.flow_matching.timestep_sampling_params)

    def _init_noise_experts(self) -> None:
        """Initialize sigma-range samplers for LoRA experts."""
        experts = self._configured_noise_expert_ranges()
        self._noise_experts = [
            NoiseExpert(
                name=name,
                min_sigma=min_sigma,
                max_sigma=max_sigma,
                sampler=RangeScaledTimestepSampler(self._timestep_sampler, min_sigma, max_sigma),
            )
            for name, (min_sigma, max_sigma) in experts.items()
        ]
        self._active_noise_expert = self._noise_experts[0]

    def _configured_noise_expert_ranges(self) -> dict[str, tuple[float, float]]:
        """Return configured LoRA expert ranges, including the default single expert."""
        if self._config.model.training_mode != "lora" or self._config.lora is None:
            return {"default": (0.0, 1.0)}
        return self._config.lora.noise_experts or {"default": (0.0, 1.0)}

    def _select_noise_expert(self, step: int) -> NoiseExpert:
        """Select the expert for this micro-batch."""
        if not self._noise_experts:
            self._init_noise_experts()
        expert = self._noise_experts[step % len(self._noise_experts)]
        self._active_noise_expert = expert
        return expert

    def _setup_lora(self) -> None:
        """Configure LoRA adapters for the transformer. Only called in LoRA training mode."""
        logger.debug(f"Adding LoRA adapter(s) with rank {self._config.lora.rank}")
        lora_config = LoraConfig(
            r=self._config.lora.rank,
            lora_alpha=self._config.lora.alpha,
            target_modules=self._config.lora.target_modules,
            lora_dropout=self._config.lora.dropout,
            init_lora_weights=True,
        )
        expert_names = list(self._configured_noise_expert_ranges())
        # Wrap the transformer with PEFT to add LoRA layers
        # noinspection PyTypeChecker
        self._transformer = get_peft_model(self._transformer, lora_config, adapter_name=expert_names[0])
        for adapter_name in expert_names[1:]:
            self._transformer.add_adapter(adapter_name, lora_config)
        self._lora_adapter_names = expert_names
        self._set_active_lora_adapter(expert_names[0])

        # Ensure every adapter is included in the optimizer even if PEFT toggles
        # inactive adapter parameters when switching active adapters.
        for name, param in self._transformer.named_parameters():
            if ".lora_" in name:
                param.requires_grad_(True)

    def _set_active_lora_adapter(self, adapter_name: str) -> None:
        """Activate a named LoRA adapter when training LoRA experts."""
        if self._config.model.training_mode != "lora" or not hasattr(self, "_transformer"):
            return
        if hasattr(self._transformer, "set_adapter"):
            self._transformer.set_adapter(adapter_name)
            for name, param in self._transformer.named_parameters():
                if ".lora_" in name:
                    param.requires_grad_(True)

    def _build_gradient_projection_groups(self) -> dict[str, list[torch.nn.Parameter]]:
        """Build NSYNC gradient projection groups from trainable parameter names."""
        groups: dict[str, list[torch.nn.Parameter]] = {}
        projection_scope = self._config.lora.nsync.projection_scope if self._config.lora is not None else "layer"
        for param, name in self._trainable_param_names.items():
            group_name = name if projection_scope == "parameter" else self._layer_projection_group_name(name)
            groups.setdefault(group_name, []).append(param)
        return groups

    @staticmethod
    def _layer_projection_group_name(param_name: str) -> str:
        """Map a parameter name to its transformer-block projection group."""
        match = re.search(r"transformer_blocks\.(\d+)", param_name)
        if match:
            return f"transformer_blocks.{match.group(1)}"
        return param_name.rsplit(".", 2)[0]

    def _load_checkpoint(self) -> None:
        """Load checkpoint if specified in config, then resolve resume state."""
        if not self._config.model.load_checkpoint:
            self._resume_state: tuple[int, TrainingState | None] = (0, None)
            return

        preflight_result = getattr(self, "_preflight_result", None)
        checkpoint_path = (
            preflight_result.checkpoint_path
            if preflight_result is not None
            else self._find_checkpoint(self._config.model.load_checkpoint)
        )
        if not checkpoint_path:
            logger.warning(f"⚠️ Could not find checkpoint at {self._config.model.load_checkpoint}")
            self._resume_state = (0, None)
            return

        self._loaded_checkpoint_path = checkpoint_path
        logger.info(f"📥 Loading checkpoint from {checkpoint_path}")

        if self._config.model.training_mode == "full":
            self._load_full_checkpoint(checkpoint_path)
        else:  # LoRA mode
            self._load_lora_checkpoint(checkpoint_path)

        if preflight_result is not None and preflight_result.resume_state is not None:
            self._resume_state = preflight_result.resume_state
        else:
            self._resume_state = self._resolve_resume_state()

    def _load_full_checkpoint(self, checkpoint_path: Path) -> None:
        """Load full model checkpoint."""
        state_dict = load_file(checkpoint_path)
        self._transformer.load_state_dict(state_dict, strict=True)

        logger.info("✅ Full model checkpoint loaded successfully")

    def _load_lora_checkpoint(self, checkpoint_path: Path) -> None:
        """Load LoRA checkpoint with DDP/FSDP compatibility."""
        if len(getattr(self, "_lora_adapter_names", ["default"])) > 1:
            step_match = re.search(r"step_(\d+)", checkpoint_path.name)
            step_suffix = step_match.group(1) if step_match else None
            for adapter_name in self._lora_adapter_names:
                adapter_checkpoint = checkpoint_path
                if step_suffix is not None:
                    candidate = checkpoint_path.parent / f"lora_weights_{adapter_name}_step_{step_suffix}.safetensors"
                    if candidate.exists():
                        adapter_checkpoint = candidate
                    else:
                        logger.warning(f"⚠️ Missing LoRA expert checkpoint for '{adapter_name}': {candidate}")
                        continue
                self._load_lora_adapter_checkpoint(adapter_checkpoint, adapter_name)
            return

        self._load_lora_adapter_checkpoint(checkpoint_path, self._lora_adapter_names[0])

    def _load_lora_adapter_checkpoint(self, checkpoint_path: Path, adapter_name: str) -> None:
        """Load a single LoRA adapter checkpoint."""
        state_dict = load_file(checkpoint_path)

        # Adjust layer names to match internal format.
        # (Weights are saved in ComfyUI-compatible format, with "diffusion_model." prefix)
        state_dict = {k.replace("diffusion_model.", "", 1): v for k, v in state_dict.items()}

        # Load LoRA weights and verify all weights were loaded
        base_model = self._transformer.get_base_model()
        set_peft_model_state_dict(base_model, state_dict, adapter_name=adapter_name)

        logger.info(f"✅ LoRA checkpoint loaded successfully for adapter '{adapter_name}'")

    def _resolve_lora_rank_from_checkpoint(self) -> None:
        """Use the checkpoint LoRA rank as the effective training rank when resuming LoRA weights."""
        if (
            self._config.model.training_mode != "lora"
            or self._config.lora is None
            or not self._config.model.load_checkpoint
        ):
            return

        preflight_result = getattr(self, "_preflight_result", None)
        checkpoint_path = (
            preflight_result.checkpoint_path
            if preflight_result is not None
            else self._find_checkpoint(self._config.model.load_checkpoint)
        )
        if checkpoint_path is None:
            return

        ranks_by_path = {
            path: self._detect_lora_rank_from_checkpoint(path)
            for path in self._lora_rank_detection_checkpoint_paths(checkpoint_path)
        }
        if not ranks_by_path:
            return

        ranks = set(ranks_by_path.values())
        if len(ranks) > 1:
            details = ", ".join(f"{path.name}: rank {rank}" for path, rank in sorted(ranks_by_path.items()))
            raise ValueError(f"LoRA expert checkpoints must all use the same rank, but found {details}")

        checkpoint_rank = ranks.pop()
        config_rank = self._config.lora.rank
        if checkpoint_rank != config_rank:
            logger.info(
                f"Using LoRA rank {checkpoint_rank} detected from checkpoint instead of configured rank {config_rank}"
            )
            self._config.lora.rank = checkpoint_rank

    def _lora_rank_detection_checkpoint_paths(self, checkpoint_path: Path) -> list[Path]:
        """Return existing LoRA files whose ranks should agree for this training run."""
        expert_names = list(self._configured_noise_expert_ranges())
        if len(expert_names) <= 1:
            return [checkpoint_path]

        step_match = re.search(r"step_(\d+)", checkpoint_path.name)
        if step_match is None:
            return [checkpoint_path]

        step_suffix = step_match.group(1)
        expert_paths = [
            checkpoint_path.parent / f"lora_weights_{adapter_name}_step_{step_suffix}.safetensors"
            for adapter_name in expert_names
        ]
        existing_paths = [path for path in expert_paths if path.exists()]
        return existing_paths or [checkpoint_path]

    @classmethod
    def _detect_lora_rank_from_checkpoint(cls, checkpoint_path: Path) -> int:
        """Detect the single LoRA rank encoded in a safetensors checkpoint."""
        state_dict = load_file(checkpoint_path)
        return cls._detect_lora_rank_from_state_dict(state_dict, checkpoint_path)

    @staticmethod
    def _detect_lora_rank_from_state_dict(state_dict: Mapping[str, Tensor], checkpoint_path: Path | str) -> int:
        """Infer LoRA rank from paired ``lora_A``/``lora_B`` factor tensors."""
        suffix_a = ".lora_A.weight"
        ranks_by_prefix: dict[str, int] = {}

        for key, lora_a in state_dict.items():
            if not key.endswith(suffix_a):
                continue

            prefix = key[: -len(suffix_a)]
            key_b = f"{prefix}.lora_B.weight"
            lora_b = state_dict.get(key_b)
            if lora_b is None:
                raise ValueError(f"LoRA checkpoint {checkpoint_path} is missing paired tensor {key_b}")
            if lora_a.ndim != 2 or lora_b.ndim != 2:
                raise ValueError(f"LoRA checkpoint {checkpoint_path} has non-matrix LoRA tensors for {prefix}")

            rank_a = lora_a.shape[0]
            rank_b = lora_b.shape[1]
            if rank_a != rank_b:
                raise ValueError(
                    f"LoRA checkpoint {checkpoint_path} has inconsistent rank for {prefix}: "
                    f"lora_A rank {rank_a}, lora_B rank {rank_b}"
                )
            ranks_by_prefix[prefix] = rank_a

        if not ranks_by_prefix:
            raise ValueError(f"Could not detect LoRA rank from checkpoint {checkpoint_path}: no lora_A tensors found")

        ranks = set(ranks_by_prefix.values())
        if len(ranks) > 1:
            details = ", ".join(f"{prefix}: rank {rank}" for prefix, rank in sorted(ranks_by_prefix.items()))
            raise ValueError(f"LoRA checkpoint {checkpoint_path} contains mixed ranks: {details}")

        return ranks.pop()

    def _resolve_resume_state(self) -> tuple[int, TrainingState | None]:
        """Determine resume state by looking for a training state file next to the loaded checkpoint.
        Returns (initial_step, TrainingState or None).
        If no_resume config is set, no checkpoint loaded, or no state file found: returns (0, None).
        """
        if self._config.checkpoints.no_resume or self._loaded_checkpoint_path is None:
            return 0, None

        return self._resolve_resume_state_from_checkpoint(self._config, self._loaded_checkpoint_path)

    @classmethod
    def _resolve_resume_state_from_checkpoint(
        cls,
        trainer_config: LtxTrainerConfig,
        checkpoint_path: Path,
    ) -> tuple[int, TrainingState | None]:
        if trainer_config.checkpoints.no_resume:
            return 0, None

        state = cls._load_training_state(checkpoint_path)
        if state is None:
            return 0, None

        mismatches = cls._training_state_mismatches(trainer_config, state)
        if mismatches:
            logger.warning(
                f"⚠️ Training state config mismatch ({', '.join(mismatches)}). "
                "Starting from step 0. Set checkpoints.no_resume=true to silence this warning."
            )
            return 0, None

        if state.global_step < 0:
            logger.warning(f"⚠️ Training state has invalid global_step={state.global_step!r}. Starting from step 0.")
            return 0, None
        logger.info(f"📌 Resuming from step {state.global_step}")
        return state.global_step, state

    @staticmethod
    def _training_state_mismatches(trainer_config: LtxTrainerConfig, state: TrainingState) -> list[str]:
        fp = state.config_fingerprint
        cfg = trainer_config
        mismatches: list[str] = []
        if fp.optimizer_type != cfg.optimization.optimizer_type:
            mismatches.append(f"optimizer_type: {fp.optimizer_type} → {cfg.optimization.optimizer_type}")
        if fp.scheduler_type != cfg.optimization.scheduler_type:
            mismatches.append(f"scheduler_type: {fp.scheduler_type} → {cfg.optimization.scheduler_type}")
        if fp.training_mode != cfg.model.training_mode:
            mismatches.append(f"training_mode: {fp.training_mode} → {cfg.model.training_mode}")
        if (
            cfg.model.training_mode == "lora"
            and cfg.lora is not None
            and fp.lora_rank is not None
            and fp.lora_rank != cfg.lora.rank
        ):
            mismatches.append(f"lora_rank: {fp.lora_rank} → {cfg.lora.rank}")
        if cfg.model.training_mode == "lora" and cfg.lora is not None:
            current_experts = cfg.lora.noise_experts
            if fp.lora_noise_experts != current_experts:
                mismatches.append(f"lora_noise_experts: {fp.lora_noise_experts} → {current_experts}")
            if fp.nsync_enabled != cfg.lora.nsync.enabled:
                mismatches.append(f"nsync_enabled: {fp.nsync_enabled} → {cfg.lora.nsync.enabled}")
            if (fp.nsync_enabled or cfg.lora.nsync.enabled) and (
                fp.nsync_negative_strength != cfg.lora.nsync.negative_strength
            ):
                mismatches.append(
                    f"nsync_negative_strength: {fp.nsync_negative_strength} → {cfg.lora.nsync.negative_strength}"
                )
            if (fp.nsync_enabled or cfg.lora.nsync.enabled) and (
                fp.nsync_anchor_strength != cfg.lora.nsync.anchor_strength
            ):
                mismatches.append(
                    f"nsync_anchor_strength: {fp.nsync_anchor_strength} → {cfg.lora.nsync.anchor_strength}"
                )
        return mismatches

    @staticmethod
    def _load_training_state(checkpoint_path: Path) -> TrainingState | None:
        """Load training state file that corresponds to a checkpoint weights file."""
        match = re.search(r"step_(\d+)", checkpoint_path.name)
        if not match:
            return None

        step_str = match.group(1)
        state_path = checkpoint_path.parent / f"training_state_step_{step_str}.pt"

        if not state_path.exists():
            return None

        try:
            raw: dict = torch.load(state_path, map_location="cpu", weights_only=False)
            state = TrainingState.from_save_dict(raw)
            logger.info(f"📥 Loaded training state from {state_path}")
            return state
        except Exception as e:
            logger.warning(f"⚠️ Failed to load training state from {state_path}: {e}. Starting from step 0.")
            return None

    def _restore_training_state(self, training_state: TrainingState) -> bool:
        """Restore optimizer, scheduler, and RNG states from a loaded TrainingState.
        Must be called after _init_optimizer() (which calls accelerator.prepare).
        Returns True if restore succeeded, False if it failed (caller should fall back to step 0).
        """
        try:
            if training_state.optimizer_state_dict is not None:
                self._optimizer.load_state_dict(training_state.optimizer_state_dict)
                logger.debug("Restored optimizer state (full mode)")

            if training_state.lr_scheduler_state_dict is not None and self._lr_scheduler is not None:
                self._lr_scheduler.load_state_dict(training_state.lr_scheduler_state_dict)
                logger.debug("Restored LR scheduler state")
        except Exception as e:
            logger.warning(f"⚠️ Failed to restore training state: {e}. Starting from step 0.")
            return False

        rng = training_state.rng_states
        if self._accelerator.num_processes > 1:
            logger.debug("Skipping RNG restore in multi-process mode (only main process state was saved)")
        else:
            if rng.torch_state is not None:
                torch.random.set_rng_state(rng.torch_state)
            if rng.cuda_state is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state(rng.cuda_state)
            logger.debug("Restored RNG states")

        return True

    def _prepare_models_for_training(self) -> None:
        """Prepare models for training with Accelerate."""

        if self._model_pool is not None and self._warm_transformer_key is not None:
            keep_warm = {self._warm_transformer_key}
            if self._warm_embeddings_key is not None:
                keep_warm.add(self._warm_embeddings_key)
            self._model_pool.offload_all(exclude=keep_warm)

        # For FSDP + LoRA: Cast entire model to FP32.
        # FSDP requires uniform dtype across all parameters in wrapped modules.
        # In LoRA mode, PEFT creates LoRA params in FP32 while base model is BF16.
        # We cast the base model to FP32 to match the LoRA params.
        if self._accelerator.distributed_type == DistributedType.FSDP and self._config.model.training_mode == "lora":
            logger.debug("FSDP: casting transformer to FP32 for uniform dtype")
            self._transformer = self._transformer.to(dtype=torch.float32)

        # Enable gradient checkpointing if requested
        # For PeftModel, we need to access the underlying base model
        transformer = (
            self._transformer.get_base_model() if hasattr(self._transformer, "get_base_model") else self._transformer
        )

        transformer.set_gradient_checkpointing(self._config.optimization.enable_gradient_checkpointing)

        # Keep frozen models on CPU for memory efficiency
        self._vae_decoder = self._vae_decoder.to("cpu")
        if self._vae_encoder is not None:
            self._vae_encoder = self._vae_encoder.to("cpu")

        # Embedding connectors are already on GPU from _load_text_encoder_and_cache_embeddings

        # noinspection PyTypeChecker
        self._transformer = self._accelerator.prepare(self._transformer)

        # Log GPU memory usage after model preparation
        vram_usage_gb = torch.cuda.memory_allocated() / 1024**3
        logger.debug(f"GPU memory usage after models preparation: {vram_usage_gb:.2f} GB")

    @staticmethod
    def _find_checkpoint(checkpoint_path: str | Path) -> Path | None:
        """Find the checkpoint file to load, handling both file and directory paths."""
        checkpoint_path = Path(checkpoint_path)

        if checkpoint_path.is_file():
            if not checkpoint_path.suffix == ".safetensors":
                raise ValueError(f"Checkpoint file must have a .safetensors extension: {checkpoint_path}")
            return checkpoint_path

        if checkpoint_path.is_dir():
            # Look for checkpoint files in the directory
            checkpoints = list(checkpoint_path.rglob("*step_*.safetensors"))

            if not checkpoints:
                return None

            # Sort by step number and return the latest
            def _get_step_num(p: Path) -> int:
                try:
                    return int(p.stem.split("step_")[1])
                except (IndexError, ValueError):
                    return -1

            latest = max(checkpoints, key=_get_step_num)
            return latest

        else:
            raise ValueError(f"Invalid checkpoint path: {checkpoint_path}. Must be a file or directory.")

    def _init_dataloader(self) -> None:
        """Initialize training and optional held-out validation data loaders."""
        if self._dataset is None:
            # Get data sources from the training strategy
            data_sources = self._training_strategy.get_data_sources()
            optional_data_sources = set(self._training_strategy.get_optional_data_sources())

            if self._nsync_requires_negatives:
                if isinstance(data_sources, list):
                    data_sources = {source: source for source in data_sources}
                data_sources[self._config.lora.nsync.negative_latents_dir] = "negative_latents"

            self._dataset = PrecomputedDataset(
                self._config.data.preprocessed_data_root,
                data_sources=data_sources,
                optional_data_sources=optional_data_sources,
            )
            if self._nsync_requires_anchor:
                self._dataset = AnchorSampleDataset(self._dataset)
            logger.debug(f"Loaded dataset with {len(self._dataset):,} samples from sources: {list(data_sources)}")

        dataloader = self._create_dataloader(self._dataset, shuffle=True, drop_last=True)
        self._dataloader = self._accelerator.prepare(dataloader)

        validation_root = self._config.data.validation_data_root
        if validation_root is None or self._config.validation.loss_interval is None:
            self._validation_dataloader = None
            return

        self._validation_dataset = PrecomputedDataset(
            validation_root,
            data_sources=self._training_strategy.get_data_sources(),
            optional_data_sources=set(self._training_strategy.get_optional_data_sources()),
        )
        validation_dataloader = self._create_dataloader(
            self._validation_dataset,
            shuffle=False,
            drop_last=False,
        )
        self._validation_dataloader = self._accelerator.prepare(validation_dataloader)
        logger.info(f"Loaded held-out validation dataset with {len(self._validation_dataset):,} samples")

    def _create_dataloader(
        self,
        dataset: PrecomputedDataset | AnchorSampleDataset,
        *,
        shuffle: bool,
        drop_last: bool,
    ) -> DataLoader:
        """Create a training-compatible dataloader for a precomputed dataset."""
        num_workers = self._config.data.num_dataloader_workers
        if self._is_mixed_audio_training:
            batch_sampler = OptionalSourceGroupedBatchSampler(
                dataset,
                output_key="audio_latents",
                batch_size=self._config.optimization.batch_size,
                drop_last=drop_last,
                shuffle=shuffle,
            )
            return DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                num_workers=num_workers,
                pin_memory=num_workers > 0,
                persistent_workers=num_workers > 0,
                collate_fn=collate_precomputed_samples,
            )
        else:
            return DataLoader(
                dataset,
                batch_size=self._config.optimization.batch_size,
                shuffle=shuffle,
                drop_last=drop_last,
                num_workers=num_workers,
                pin_memory=num_workers > 0,
                persistent_workers=num_workers > 0,
                collate_fn=collate_precomputed_samples,
            )

    @property
    def _is_mixed_audio_training(self) -> bool:
        return bool(getattr(self._config.training_strategy, "mixed_audio", False))

    def _init_lora_weights(self) -> None:
        """Initialize LoRA weights for the transformer."""
        logger.debug("Initializing LoRA weights...")
        for _, module in self._transformer.named_modules():
            if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper)):
                module.reset_lora_parameters(adapter_name="default", init_lora_weights=True)

    def _init_optimizer(self) -> None:
        """Initialize the optimizer and learning rate scheduler."""
        opt_cfg = self._config.optimization

        lr = opt_cfg.learning_rate
        if opt_cfg.optimizer_type == "adamw":
            optimizer = AdamW(self._trainable_params, lr=lr)
        elif opt_cfg.optimizer_type == "adamw8bit":
            # noinspection PyUnresolvedReferences
            from bitsandbytes.optim import AdamW8bit  # noqa: PLC0415

            optimizer = AdamW8bit(self._trainable_params, lr=lr)
        else:
            raise ValueError(f"Unknown optimizer type: {opt_cfg.optimizer_type}")

        lr_scheduler = self._create_scheduler(optimizer)

        # noinspection PyTypeChecker
        self._optimizer, self._lr_scheduler = self._accelerator.prepare(optimizer, lr_scheduler)

    @contextlib.contextmanager
    def _offloaded_optimizer_state(self) -> Iterator[None]:
        """Context manager that offloads optimizer state to CPU during validation.
        Opt-in via `acceleration.offload_optimizer_during_validation`. Frees VRAM for
        validation video generation when optimizer state is large (e.g. full fine-tune
        AdamW, high-rank LoRA). No-op for FSDP (sharded state -- manual `.cpu()` breaks
        metadata).
        """
        enabled = (
            self._config.acceleration.offload_optimizer_during_validation
            and self._accelerator.distributed_type != DistributedType.FSDP
        )

        # Track exactly which tensors we move so we don't promote ones that were
        # intentionally on CPU (e.g. AdamW's `step` scalar on recent PyTorch).
        offloaded: list[tuple[dict, str]] = []
        if enabled:
            offloaded_bytes = 0
            for state in self._optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor) and v.is_cuda:
                        offloaded.append((state, k))
                        offloaded_bytes += v.nbytes
            if offloaded:
                logger.info(f"Offloading optimizer state to CPU ({offloaded_bytes / 1e9:.1f} GB)")
                for state, k in offloaded:
                    state[k] = state[k].cpu()

        try:
            yield
        finally:
            device = self._accelerator.device
            for state, k in offloaded:
                state[k] = state[k].to(device)

    def _create_scheduler(self, optimizer: torch.optim.Optimizer) -> LRScheduler | None:
        """Create learning rate scheduler based on config."""
        scheduler_type = self._config.optimization.scheduler_type
        steps = self._config.optimization.steps
        params = dict(self._config.optimization.scheduler_params or {})

        if scheduler_type is None:
            return None

        if scheduler_type == "linear":
            scheduler = LinearLR(
                optimizer,
                start_factor=params.pop("start_factor", 1.0),
                end_factor=params.pop("end_factor", 0.1),
                total_iters=steps,
                **params,
            )
        elif scheduler_type == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=steps,
                eta_min=params.pop("eta_min", 0),
                **params,
            )
        elif scheduler_type == "cosine_with_restarts":
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=params.pop("T_0", steps // 4),
                T_mult=params.pop("T_mult", 1),
                eta_min=params.pop("eta_min", 5e-5),
                **params,
            )
        elif scheduler_type == "polynomial":
            scheduler = PolynomialLR(
                optimizer,
                total_iters=steps,
                power=params.pop("power", 1.0),
                **params,
            )
        elif scheduler_type == "step":
            scheduler = StepLR(
                optimizer,
                step_size=params.pop("step_size", steps // 2),
                gamma=params.pop("gamma", 0.1),
                **params,
            )
        elif scheduler_type == "constant":
            scheduler = None
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

        return scheduler

    def _setup_accelerator(self) -> None:
        """Initialize the Accelerator with the appropriate settings."""

        # find_unused_parameters=True keeps DDP happy when LoRA targets a branch the forward
        # pass skips (e.g. audio LoRA with `with_audio: false`, or short module patterns like
        # "to_k" that match the audio branch unintentionally). It's a no-op for FSDP and
        # single-GPU runs. The probing cost is paid only on the first step.
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

        # All distributed setup (DDP/FSDP, number of processes, etc.) is controlled by
        # the user's Accelerate configuration (accelerate config / accelerate launch).
        self._accelerator = Accelerator(
            mixed_precision=self._config.acceleration.mixed_precision_mode,
            gradient_accumulation_steps=self._config.optimization.gradient_accumulation_steps,
            kwargs_handlers=[ddp_kwargs],
        )

        if self._accelerator.num_processes > 1:
            logger.info(
                f"{self._accelerator.distributed_type.value} distributed training enabled "
                f"with {self._accelerator.num_processes} processes"
            )

            local_batch = self._config.optimization.batch_size
            global_batch = self._config.optimization.batch_size * self._accelerator.num_processes
            logger.info(f"Local batch size: {local_batch}, global batch size: {global_batch}")

        # Log torch.compile status from Accelerate's dynamo plugin
        is_compile_enabled = (
            hasattr(self._accelerator.state, "dynamo_plugin") and self._accelerator.state.dynamo_plugin.backend != "NO"
        )
        if is_compile_enabled:
            plugin = self._accelerator.state.dynamo_plugin
            logger.info(f"🔥 torch.compile enabled via Accelerate: backend={plugin.backend}, mode={plugin.mode}")

            if self._accelerator.distributed_type == DistributedType.FSDP:
                logger.warning(
                    "⚠️ FSDP + torch.compile is experimental and may hang on the first training iteration. "
                    "If this occurs, disable torch.compile by removing dynamo_config from your Accelerate config."
                )

        if self._accelerator.distributed_type == DistributedType.FSDP and self._config.acceleration.quantization:
            logger.warning(
                f"FSDP with quantization ({self._config.acceleration.quantization}) may have compatibility issues."
                "Monitor training stability and consider disabling quantization if issues arise."
            )

    @torch.no_grad()
    def _run_validation_loss(self) -> dict[str, float]:
        """Evaluate the training objective on the held-out dataset across all ranks."""
        if self._validation_dataloader is None:
            return {}

        was_training = self._transformer.training
        previous_expert = self._active_noise_expert
        self._transformer.eval()

        device = self._accelerator.device
        cuda_devices = []
        if device.type == "cuda":
            cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]

        expert_losses: dict[str, float] = {}
        try:
            # Restore training RNG state on exit so evaluation never changes subsequent training batches.
            with torch.random.fork_rng(devices=cuda_devices):
                for expert_index, expert in enumerate(self._noise_experts):
                    validation_seed = self._config.validation.seed + self._accelerator.process_index + expert_index
                    torch.manual_seed(validation_seed)
                    self._active_noise_expert = expert
                    self._set_active_lora_adapter(expert.name)

                    total_loss = 0.0
                    total_samples = 0
                    max_batches = self._config.validation.max_loss_batches
                    for batch_index, batch in enumerate(self._validation_dataloader):
                        if max_batches is not None and batch_index >= max_batches:
                            break

                        output = self._training_step(batch, expert.sampler)
                        gathered_loss = self._accelerator.gather_for_metrics(output.loss.detach().float())
                        total_loss += gathered_loss.sum().item()
                        total_samples += gathered_loss.numel()

                    if total_samples == 0:
                        raise RuntimeError("Held-out validation dataloader produced no samples")
                    expert_losses[expert.name] = total_loss / total_samples
        finally:
            if previous_expert is not None:
                self._active_noise_expert = previous_expert
                self._set_active_lora_adapter(previous_expert.name)
            self._transformer.train(was_training)

        if len(expert_losses) == 1:
            metrics = {"validation/loss": next(iter(expert_losses.values()))}
        else:
            metrics = {
                "validation/loss": sum(expert_losses.values()) / len(expert_losses),
                **{f"validation/loss/{name}": loss for name, loss in expert_losses.items()},
            }

        if self._accelerator.is_main_process:
            loss_summary = ", ".join(f"{name}={loss:.6f}" for name, loss in expert_losses.items())
            logger.info(f"Held-out validation loss at step {self._global_step}: {loss_summary}")
            self._log_metrics(metrics, step=self._global_step)

        self._accelerator.wait_for_everyone()
        return metrics

    def _run_distributed_validation(self, progress: TrainingProgress) -> list[Path]:
        """Run validation across all ranks and log gathered results on rank 0.
        Each rank generates only its assigned subset of prompts (see `_sample_videos`),
        so all GPUs stay busy and no rank idles long enough to trigger NCCL timeouts.
        Paths are gathered across ranks so rank 0 has the full list for W&B logging.
        Note: Multi-node training requires a shared filesystem so rank 0 can read
        videos written by other ranks.
        """
        sampled = self._sample_videos(progress)

        if self._accelerator.num_processes > 1:
            # gather_object returns a flat list from all ranks
            sampled = sorted(gather_object(sampled), key=lambda x: x[0])

        paths = [p for _, p in sampled]

        if self._accelerator.is_main_process and paths:
            self._log_validation_samples(paths, self._config.validation.prompts)

        # Non-main ranks must not reach checkpoint collectives while main is still logging to W&B.
        self._accelerator.wait_for_everyone()

        return paths

    # Note: Use @torch.no_grad() instead of @torch.inference_mode() to avoid FSDP inplace update errors after validation
    @torch.no_grad()
    @free_gpu_memory_context(after=True)
    def _sample_videos(self, progress: TrainingProgress) -> list[tuple[int, Path]]:
        """Run validation by generating videos from this rank's share of the validation prompts.
        Prompts are split round-robin across ranks via `process_index` / `num_processes`,
        which collapses to "all prompts" when running on a single GPU. Returns
        (prompt_idx, path) tuples so the caller can reconstruct global order without
        relying on filename conventions.
        Under FSDP with multiple processes, ranks pad with extra generate passes (same prompt,
        no disk write) so every rank runs the same number of forwards — avoids collective mismatch.
        """
        use_images = self._config.validation.images is not None
        use_reference_videos = self._config.validation.reference_videos is not None
        generate_audio = self._config.validation.generate_audio
        inference_steps = self._config.validation.inference_steps

        # Zero gradients and free GPU memory to reclaim memory before validation sampling
        self._optimizer.zero_grad(set_to_none=True)
        free_gpu_memory()

        prompts = self._config.validation.prompts
        rank = self._accelerator.process_index
        world_size = self._accelerator.num_processes
        rank_indices = list(range(rank, len(prompts), world_size))

        # FSDP: every rank must run the same number of forwards; pad with duplicate generates (no save).
        work: list[tuple[int, bool]] = [(i, True) for i in rank_indices]
        if self._accelerator.distributed_type == DistributedType.FSDP and world_size > 1:
            max_per_rank = math.ceil(len(prompts) / world_size)
            pad_seed = rank_indices[-1] if rank_indices else 0
            work += [(pad_seed, False)] * (max_per_rank - len(work))

        sampling_ctx = progress.start_sampling(
            num_prompts=len(work),
            num_steps=inference_steps,
        )

        # Create a validation sampler with loaded models and progress tracking
        sampler = ValidationSampler(
            transformer=self._transformer,
            vae_decoder=self._vae_decoder,
            vae_encoder=self._vae_encoder,
            text_encoder=None,
            audio_decoder=self._audio_vae if generate_audio else None,
            vocoder=self._vocoder if generate_audio else None,
            sampling_context=sampling_ctx,
        )

        output_dir = Path(self._config.output_dir) / "samples"
        output_dir.mkdir(exist_ok=True, parents=True)

        results: list[tuple[int, Path]] = []
        width, height, num_frames = self._config.validation.video_dims

        for local_i, (prompt_idx, save_output) in enumerate(work):
            prompt = prompts[prompt_idx]
            sampling_ctx.start_video(local_i)

            # Load conditioning image if provided
            condition_image = None
            if use_images:
                image_path = self._config.validation.images[prompt_idx]
                image = open_image_as_srgb(image_path)
                # Convert PIL image to tensor [C, H, W] in [0, 1]
                condition_image = F.to_tensor(image)

            # Load reference video if provided (for IC-LoRA)
            reference_video = None
            if use_reference_videos:
                ref_video_path = self._config.validation.reference_videos[prompt_idx]
                # read_video returns [F, C, H, W] in [0, 1]
                reference_video, _ = read_video(ref_video_path, max_frames=num_frames)

            # Get cached embeddings for this prompt if available
            cached_embeddings = (
                self._cached_validation_embeddings[prompt_idx]
                if self._cached_validation_embeddings is not None
                else None
            )

            # Create generation config
            gen_config = GenerationConfig(
                prompt=prompt,
                negative_prompt=self._config.validation.negative_prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=self._config.validation.frame_rate,
                num_inference_steps=inference_steps,
                guidance_scale=self._config.validation.guidance_scale,
                seed=self._config.validation.seed,
                condition_image=condition_image,
                reference_video=reference_video,
                reference_downscale_factor=self._config.validation.reference_downscale_factor,
                generate_audio=generate_audio,
                include_reference_in_output=self._config.validation.include_reference_in_output,
                cached_embeddings=cached_embeddings,
                stg_scale=self._config.validation.stg_scale,
                stg_blocks=self._config.validation.stg_blocks,
                stg_mode=self._config.validation.stg_mode,
            )

            # Generate sample
            video, audio = sampler.generate(
                config=gen_config,
                device=self._accelerator.device,
            )

            if not save_output:
                continue

            # Save output (image for single frame, video otherwise)
            ext = "png" if num_frames == 1 else "mp4"
            output_path = output_dir / f"step_{self._global_step:06d}_{prompt_idx + 1:02d}.{ext}"
            if num_frames == 1:
                save_image(video, output_path)
            else:
                save_video(
                    video_tensor=video,
                    output_path=output_path,
                    fps=self._config.validation.frame_rate,
                    audio=audio,
                    audio_sample_rate=self._vocoder.output_sampling_rate if audio is not None else None,
                )
            results.append((prompt_idx, output_path))

        # Clean up progress tasks
        sampling_ctx.cleanup()

        rel_outputs_path = output_dir.relative_to(self._config.output_dir)
        logger.info(f"🎥 Validation samples for step {self._global_step} saved in {rel_outputs_path}")
        return results

    @staticmethod
    def _log_training_stats(stats: TrainingStats) -> None:
        """Log training statistics."""
        stats_str = (
            "📊 Training Statistics:\n"
            f" - Total time: {stats.total_time_seconds / 60:.1f} minutes\n"
            f" - Training speed: {stats.steps_per_second:.2f} steps/second\n"
            f" - Samples/second: {stats.samples_per_second:.2f}\n"
            f" - Peak GPU memory: {stats.peak_gpu_memory_gb:.2f} GB"
        )
        if stats.num_processes > 1:
            stats_str += f"\n - Number of processes: {stats.num_processes}\n"
            stats_str += f" - Global batch size: {stats.global_batch_size}"
        logger.info(stats_str)

    def _save_checkpoint(self) -> Path | None:
        """Save the model weights."""
        is_lora = self._config.model.training_mode == "lora"
        is_fsdp = self._accelerator.distributed_type == DistributedType.FSDP

        # Prepare paths
        save_dir = Path(self._config.output_dir) / "checkpoints"
        prefix = "lora" if is_lora else "model"
        filename = f"{prefix}_weights_step_{self._global_step:05d}.safetensors"
        saved_weights_path = save_dir / filename

        # Get state dict (collective operation - all processes must participate)
        self._accelerator.wait_for_everyone()
        full_state_dict = self._accelerator.get_state_dict(self._transformer)

        if not IS_MAIN_PROCESS:
            return None

        save_dir.mkdir(exist_ok=True, parents=True)

        # Determine save precision
        save_dtype = torch.bfloat16 if self._config.checkpoints.precision == "bfloat16" else torch.float32

        # For LoRA: extract only adapter weights; for full: use as-is
        if is_lora:
            unwrapped = self._accelerator.unwrap_model(self._transformer, keep_torch_compile=False)
            saved_paths = []
            for adapter_name in self._lora_adapter_names:
                adapter_path = self._lora_checkpoint_path(save_dir, adapter_name)
                state_dict = self._get_lora_state_dict_for_save(
                    unwrapped=unwrapped,
                    full_state_dict=full_state_dict,
                    is_fsdp=is_fsdp,
                    adapter_name=adapter_name,
                    save_dtype=save_dtype,
                )
                metadata = self._build_checkpoint_metadata(adapter_name=adapter_name)
                save_file(state_dict, adapter_path, metadata=metadata)
                saved_paths.append(adapter_path)

            saved_weights_path = saved_paths[0]
        else:
            # Cast to configured precision
            full_state_dict = {k: v.to(save_dtype) if isinstance(v, Tensor) else v for k, v in full_state_dict.items()}

            # Save to disk
            self._accelerator.save(full_state_dict, saved_weights_path)

        rel_path = saved_weights_path.relative_to(self._config.output_dir)
        logger.info(f"💾 {prefix.capitalize()} weights for step {self._global_step} saved in {rel_path}")

        self._checkpoint_paths.append(saved_weights_path)
        self._cleanup_checkpoints()

        self._save_training_state(save_dir)

        return saved_weights_path

    def _lora_checkpoint_path(self, save_dir: Path, adapter_name: str) -> Path:
        """Return the checkpoint path for a LoRA adapter."""
        if len(self._lora_adapter_names) == 1 and adapter_name == "default":
            return save_dir / f"lora_weights_step_{self._global_step:05d}.safetensors"
        return save_dir / f"lora_weights_{adapter_name}_step_{self._global_step:05d}.safetensors"

    @staticmethod
    def _get_lora_state_dict_for_save(
        unwrapped: torch.nn.Module,
        full_state_dict: dict[str, Tensor],
        is_fsdp: bool,
        adapter_name: str,
        save_dtype: torch.dtype,
    ) -> dict[str, Tensor]:
        """Extract and convert one LoRA adapter state dict for checkpoint saving."""
        state_dict = get_peft_model_state_dict(
            unwrapped,
            state_dict=full_state_dict if is_fsdp else None,
            adapter_name=adapter_name,
        )

        # Remove "base_model.model." prefix added by PEFT
        state_dict = {k.replace("base_model.model.", "", 1): v for k, v in state_dict.items()}

        # Convert to ComfyUI-compatible format (add "diffusion_model." prefix)
        state_dict = {f"diffusion_model.{k}": v for k, v in state_dict.items()}

        return {k: v.to(save_dtype) if isinstance(v, Tensor) else v for k, v in state_dict.items()}

    def _cleanup_checkpoints(self) -> None:
        """Clean up old checkpoints."""
        if 0 < self._config.checkpoints.keep_last_n < len(self._checkpoint_paths):
            checkpoints_to_remove = self._checkpoint_paths[: -self._config.checkpoints.keep_last_n]
            for old_checkpoint in checkpoints_to_remove:
                for path in self._checkpoint_family_paths(old_checkpoint):
                    if path.exists():
                        path.unlink()
                        logger.info(f"Removed old checkpoint: {path}")
            self._checkpoint_paths = self._checkpoint_paths[-self._config.checkpoints.keep_last_n :]

    def _checkpoint_family_paths(self, checkpoint_path: Path) -> list[Path]:
        """Return all sibling expert checkpoint paths for a saved step."""
        if len(getattr(self, "_lora_adapter_names", [])) <= 1:
            return [checkpoint_path]
        match = re.search(r"step_(\d+)", checkpoint_path.name)
        if not match:
            return [checkpoint_path]
        step_suffix = match.group(1)
        return [
            checkpoint_path.parent / f"lora_weights_{adapter_name}_step_{step_suffix}.safetensors"
            for adapter_name in self._lora_adapter_names
        ]

    def _save_training_state(self, save_dir: Path) -> None:
        """Save training state alongside checkpoint for resume.
        Respects checkpoints.save_training_state config:
        - "full": optimizer + scheduler + RNG + step + wandb_run_id
        - "minimal": scheduler + RNG + step + wandb_run_id
        - "off": skip entirely
        """
        if not IS_MAIN_PROCESS:
            return

        mode = self._config.checkpoints.save_training_state
        if mode == "off":
            return

        is_fsdp = self._accelerator.distributed_type == DistributedType.FSDP

        optimizer_state = None
        if mode == "full":
            if is_fsdp:
                logger.warning(
                    "⚠️ save_training_state='full' is not supported with FSDP. "
                    "Saving 'minimal' state (scheduler + RNG only)."
                )
            else:
                optimizer_state = self._optimizer.state_dict()

        state = TrainingState(
            global_step=self._global_step,
            config_fingerprint=ConfigFingerprint(
                optimizer_type=self._config.optimization.optimizer_type,
                scheduler_type=self._config.optimization.scheduler_type,
                training_mode=self._config.model.training_mode,
                lora_rank=self._config.lora.rank if self._config.lora is not None else None,
                lora_noise_experts=self._config.lora.noise_experts if self._config.lora is not None else None,
                nsync_enabled=self._config.lora.nsync.enabled if self._config.lora is not None else False,
                nsync_negative_strength=(
                    self._config.lora.nsync.negative_strength if self._config.lora is not None else 1.0
                ),
                nsync_anchor_strength=(
                    self._config.lora.nsync.anchor_strength if self._config.lora is not None else 0.0
                ),
            ),
            rng_states=RngStates(
                torch_state=torch.random.get_rng_state(),
                cuda_state=torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            ),
            lr_scheduler_state_dict=self._lr_scheduler.state_dict() if self._lr_scheduler is not None else None,
            optimizer_state_dict=optimizer_state,
            wandb_run_id=self._wandb_run.id if self._wandb_run is not None else None,
        )

        state_path = save_dir / f"training_state_step_{self._global_step:05d}.pt"
        tmp_path = state_path.with_suffix(".pt.tmp")
        try:
            torch.save(state.to_save_dict(), tmp_path)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise
        tmp_path.rename(state_path)

        file_size_gb = state_path.stat().st_size / (1024**3)
        if file_size_gb > 1.0 and not self._training_state_size_warned:
            self._training_state_size_warned = True
            logger.warning(
                f"⚠️ Training state file is {file_size_gb:.1f} GB (full mode includes optimizer state). "
                f'Set checkpoints.save_training_state="minimal" to save only scheduler/RNG/step (~few KB), '
                f'or "off" to disable entirely.'
            )

        if not self._training_state_paths or self._training_state_paths[-1] != state_path:
            self._training_state_paths.append(state_path)
        self._cleanup_training_states()

        rel_path = state_path.relative_to(self._config.output_dir)
        logger.debug(f"Training state saved to {rel_path}")

    def _cleanup_training_states(self) -> None:
        """Clean up old training state files, using the same keep_last_n as checkpoints."""
        keep_n = self._config.checkpoints.keep_last_n
        if 0 < keep_n < len(self._training_state_paths):
            to_remove = self._training_state_paths[:-keep_n]
            for old_state in to_remove:
                if old_state.exists():
                    old_state.unlink()
                    logger.debug(f"Removed old training state: {old_state}")
            self._training_state_paths = self._training_state_paths[-keep_n:]

    def _build_checkpoint_metadata(self, adapter_name: str | None = None) -> dict[str, str]:
        """Build metadata dictionary for safetensors checkpoint.
        Delegates to the training strategy to get strategy-specific metadata
        that downstream inference pipelines may need.
        Returns:
            Dictionary of string key-value pairs for safetensors metadata.
            Values are converted to strings for safetensors compatibility.
        """
        raw_metadata = self._training_strategy.get_checkpoint_metadata()
        if adapter_name is not None:
            raw_metadata["lora_adapter_name"] = adapter_name
            expert_ranges = self._configured_noise_expert_ranges()
            if adapter_name in expert_ranges:
                min_sigma, max_sigma = expert_ranges[adapter_name]
                raw_metadata["noise_expert_min_sigma"] = min_sigma
                raw_metadata["noise_expert_max_sigma"] = max_sigma
        if self._nsync_enabled:
            raw_metadata["nsync_enabled"] = True
            raw_metadata["nsync_projection_scope"] = self._config.lora.nsync.projection_scope
            raw_metadata["nsync_negative_strength"] = self._config.lora.nsync.negative_strength
            raw_metadata["nsync_anchor_strength"] = self._config.lora.nsync.anchor_strength
        # Convert all values to strings for safetensors compatibility
        metadata = {k: str(v) for k, v in raw_metadata.items()}
        if metadata:
            logger.info(f"Saving checkpoint metadata: {metadata}")
        return metadata

    def _save_config(self) -> None:
        """Save the training configuration as a YAML file in the output directory."""
        if not IS_MAIN_PROCESS:
            return

        config_path = Path(self._config.output_dir) / "training_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(self._config.model_dump(), f, default_flow_style=False, indent=2)

        logger.info(f"💾 Training configuration saved to: {config_path.relative_to(self._config.output_dir)}")

    def _init_wandb(self, resume_run_id: str | None = None) -> None:
        """Initialize Weights & Biases run."""
        if not self._config.wandb.enabled or not IS_MAIN_PROCESS:
            self._wandb_run = None
            return

        wandb_config = self._config.wandb
        init_kwargs: dict[str, Any] = {
            "project": wandb_config.project,
            "entity": wandb_config.entity,
            "name": Path(self._config.output_dir).name,
            "tags": wandb_config.tags,
            "config": self._config.model_dump(),
        }
        if resume_run_id is not None:
            init_kwargs["id"] = resume_run_id
            init_kwargs["resume"] = "allow"
        run = wandb.init(**init_kwargs)
        self._wandb_run = run

    def _init_tensorboard(self) -> None:
        """Initialize the TensorBoard writer on the main process."""
        if not self._config.tensorboard.enabled or not IS_MAIN_PROCESS:
            self._tensorboard_writer = None
            return

        tensorboard_config = self._config.tensorboard
        log_dir = (
            Path(tensorboard_config.log_dir).expanduser()
            if tensorboard_config.log_dir is not None
            else Path(self._config.output_dir) / "tensorboard"
        )
        log_dir.mkdir(parents=True, exist_ok=True)
        self._tensorboard_writer = SummaryWriter(
            log_dir=str(log_dir),
            flush_secs=tensorboard_config.flush_secs,
        )
        logger.info(f"TensorBoard logging enabled: {log_dir}")

    def _log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log scalar metrics to enabled experiment trackers."""
        if self._wandb_run is not None:
            if step is None:
                self._wandb_run.log(metrics)
            else:
                self._wandb_run.log(metrics, step=step, commit=False)
        if self._tensorboard_writer is not None:
            global_step = self._global_step if step is None else step
            for name, value in metrics.items():
                self._tensorboard_writer.add_scalar(name, value, global_step)

    def _log_validation_samples(self, sample_paths: list[Path], prompts: list[str]) -> None:
        """Log validation samples (videos or images) to Weights & Biases."""
        if not self._config.wandb.log_validation_videos or self._wandb_run is None:
            return

        # Determine if outputs are images or videos based on file extension
        is_image = sample_paths and sample_paths[0].suffix.lower() in (".png", ".jpg", ".jpeg", ".heic", ".webp")

        if is_image:
            samples = [
                wandb.Image(str(path), caption=prompt) for path, prompt in zip(sample_paths, prompts, strict=True)
            ]
        else:
            samples = [
                wandb.Video(str(path), caption=prompt, format=path.suffix.lower().lstrip("."))
                for path, prompt in zip(sample_paths, prompts, strict=True)
            ]
        self._wandb_run.log({"validation_samples": samples}, step=self._global_step)
