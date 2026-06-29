#!/usr/bin/env python3

"""Run multiple preprocessing/training jobs with one warm model cache."""

from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Annotated, Any

import torch
import typer
import yaml
from process_dataset import parse_resolution_buckets, preprocess_dataset
from rich.console import Console

from ltx_trainer.config import LtxTrainerConfig
from ltx_trainer.model_pool import WarmModelPool
from ltx_trainer.trainer import LtxvTrainer

console = Console()
app = typer.Typer(
    pretty_exceptions_enable=False,
    no_args_is_help=True,
    help="Run a sequence of LTX preprocessing and LoRA training jobs while keeping models warm.",
)


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    jobs = raw.get("jobs") if isinstance(raw, dict) else raw
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("Session manifest must contain a non-empty 'jobs' list")
    if not all(isinstance(job, dict) for job in jobs):
        raise ValueError("Every session job must be a mapping")
    return jobs


def _job_kind(job: dict[str, Any], index: int) -> str:
    kind = job.get("type")
    if kind not in {"preprocess", "train"}:
        raise ValueError(f"Job {index} has invalid type {kind!r}; expected 'preprocess' or 'train'")
    return kind


def _preprocess_args(job: dict[str, Any], index: int) -> dict[str, Any]:
    raw_args = job.get("args", {})
    if not isinstance(raw_args, dict):
        raise ValueError(f"Job {index} 'args' must be a mapping")
    args = dict(raw_args)
    if "dataset_path" in args:
        args["dataset_file"] = args.pop("dataset_path")

    required = ("dataset_file", "resolution_buckets", "model_path", "text_encoder_path")
    missing = [name for name in required if name not in args]
    if missing:
        raise ValueError(f"Preprocess job {index} is missing: {', '.join(missing)}")

    buckets = args["resolution_buckets"]
    if isinstance(buckets, str):
        args["resolution_buckets"] = parse_resolution_buckets(buckets)
    elif isinstance(buckets, list):
        args["resolution_buckets"] = [tuple(bucket) for bucket in buckets]
    else:
        raise ValueError(f"Preprocess job {index} resolution_buckets must be a string or list")

    defaults = {
        "caption_column": "caption",
        "video_column": "media_path",
        "batch_size": 1,
        "output_dir": None,
        "lora_trigger": None,
        "vae_tiling": False,
        "decode": False,
        "device": "cuda",
    }
    return {**defaults, **args}


def _training_config(job: dict[str, Any], index: int, manifest_dir: Path) -> tuple[LtxTrainerConfig, bool]:
    config_value = job.get("config")
    if not isinstance(config_value, str):
        raise ValueError(f"Training job {index} requires a string 'config' path")
    config_path = Path(config_value).expanduser()
    if not config_path.is_absolute():
        config_path = manifest_dir / config_path
    if not config_path.is_file():
        raise FileNotFoundError(f"Training config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as file:
        config = LtxTrainerConfig(**yaml.safe_load(file))
    return config, bool(job.get("disable_progress_bars", False))


def _validate_training_sequence(configs: list[LtxTrainerConfig]) -> None:
    mixed_precision_modes = {config.acceleration.mixed_precision_mode for config in configs}
    if len(mixed_precision_modes) > 1:
        modes = ", ".join(sorted(mixed_precision_modes))
        raise ValueError(
            f"All training jobs in one warm session must use the same acceleration.mixed_precision_mode; found: {modes}"
        )

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1 and len(configs) > 1:
        raise ValueError(
            "Multiple training jobs in one warm session are currently supported only on one process/GPU. "
            "Accelerate destroys the distributed process group after a training job."
        )


@app.command()
def main(
    manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
) -> None:
    """Execute all jobs in MANIFEST, retaining compatible models between jobs."""
    manifest = manifest.expanduser().resolve()
    jobs = _load_manifest(manifest)
    kinds = [_job_kind(job, index) for index, job in enumerate(jobs, start=1)]

    training_jobs: dict[int, tuple[LtxTrainerConfig, bool]] = {}
    for index, (kind, job) in enumerate(zip(kinds, jobs, strict=True), start=1):
        if kind == "train":
            training_jobs[index] = _training_config(job, index, manifest.parent)
    _validate_training_sequence([config for config, _ in training_jobs.values()])

    with WarmModelPool() as model_pool:
        for index, (kind, job) in enumerate(zip(kinds, jobs, strict=True), start=1):
            console.rule(f"Job {index}/{len(jobs)}: {kind}")
            if kind == "preprocess":
                preprocess_dataset(**_preprocess_args(job, index), model_pool=model_pool)
                continue

            config, disable_progress_bars = training_jobs[index]
            preflight = LtxvTrainer.preflight_config(config)
            use_warm_pool = config.model.training_mode == "lora"
            if not use_warm_pool:
                console.print(
                    "[yellow]Full fine-tuning mutates base weights; running this job with isolated model loading.[/]"
                )
                model_pool.offload_all()

            trainer = LtxvTrainer(
                config,
                preflight_result=preflight,
                model_pool=model_pool if use_warm_pool else None,
            )
            try:
                trainer.train(disable_progress_bars=disable_progress_bars)
            finally:
                trainer.release_warm_models()
                del trainer
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    app()
