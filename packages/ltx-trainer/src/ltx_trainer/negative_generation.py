from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from ltx_trainer import logger
from ltx_trainer.model_loader import load_embeddings_processor, load_text_encoder
from ltx_trainer.model_loader import load_model as load_ltx_model
from ltx_trainer.validation_sampler import GeneratedLatents, GenerationConfig, ValidationSampler
from ltx_trainer.video_utils import save_video

VIDEO_LATENT_SPATIAL_FACTOR = 32
VIDEO_LATENT_TEMPORAL_FACTOR = 8


@dataclass(frozen=True)
class NegativeSampleSpec:
    """A single row's paired NSYNC negative metadata."""

    media_path: str
    negative_caption: str
    negative_media_path: str | None = None


@dataclass(frozen=True)
class NegativeLatentGenerationSpec:
    """Description of one synthetic negative latent file to generate."""

    positive_media_path: str
    output_rel_path: str
    prompt: str


def load_negative_sample_specs(
    dataset_file: str | Path,
    *,
    media_column: str,
    negative_caption_column: str,
    negative_media_column: str,
) -> list[NegativeSampleSpec]:
    """Load paired negative specs from a metadata file.

    Returns an empty list when the dataset does not define either negative column,
    which keeps preprocessing backward compatible for existing datasets.
    """
    dataset_path = Path(dataset_file)
    records = _load_dataset_records(dataset_path)
    if not records:
        return []

    has_negative_caption = any(negative_caption_column in record for record in records)
    has_negative_media = any(negative_media_column in record for record in records)
    if not has_negative_caption and not has_negative_media:
        return []
    if not has_negative_caption:
        raise ValueError(
            f"Dataset defines '{negative_media_column}' but not '{negative_caption_column}'. "
            "negative_caption is required for NSYNC preprocessing."
        )

    specs: list[NegativeSampleSpec] = []
    for index, record in enumerate(records):
        if media_column not in record:
            raise ValueError(f"Key '{media_column}' not found in dataset row {index}")

        negative_caption = record.get(negative_caption_column)
        if _is_missing_value(negative_caption):
            raise ValueError(
                f"Missing '{negative_caption_column}' for dataset row {index}. "
                "negative_caption is required for NSYNC preprocessing."
            )

        negative_media_path = record.get(negative_media_column)
        specs.append(
            NegativeSampleSpec(
                media_path=str(record[media_column]).strip(),
                negative_caption=str(negative_caption).strip(),
                negative_media_path=None if _is_missing_value(negative_media_path) else str(negative_media_path).strip(),
            )
        )

    return specs


def split_negative_sample_specs(
    specs: list[NegativeSampleSpec],
) -> tuple[list[NegativeSampleSpec], list[NegativeSampleSpec]]:
    """Split specs into user-supplied and auto-generated negative subsets."""
    manual_specs = [spec for spec in specs if spec.negative_media_path is not None]
    generated_specs = [spec for spec in specs if spec.negative_media_path is None]
    return manual_specs, generated_specs


def write_negative_media_subset(
    specs: list[NegativeSampleSpec],
    *,
    output_path: str | Path,
    media_column: str,
    negative_media_column: str,
) -> Path:
    """Write the subset of rows with user-supplied negative media to a JSONL file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for spec in specs:
            if spec.negative_media_path is None:
                continue
            record = {
                media_column: spec.media_path,
                negative_media_column: spec.negative_media_path,
            }
            handle.write(json.dumps(record))
            handle.write("\n")

    return output_path


def generate_missing_negative_latents(  # noqa: PLR0913
    specs: list[NegativeSampleSpec],
    *,
    positive_latents_dir: str | Path,
    output_dir: str | Path,
    model_path: str | Path,
    text_encoder_path: str | Path,
    device: str = "cuda",
    with_audio: bool = False,
    audio_output_dir: str | Path | None = None,
    inference_steps: int = 30,
    guidance_scale: float = 4.0,
    negative_prompt: str = "",
    seed: int = 42,
    use_first_frame_conditioning: bool = False,
    save_previews: bool = False,
    preview_output_dir: str | Path | None = None,
    load_text_encoder_in_8bit: bool = False,
) -> None:
    """Generate paired NSYNC negatives directly into trainer-format latent files."""
    generation_specs = [
        NegativeLatentGenerationSpec(
            positive_media_path=spec.media_path,
            output_rel_path=str(Path(spec.media_path).with_suffix(".pt")),
            prompt=spec.negative_caption,
        )
        for spec in specs
    ]
    generate_negative_latents(
        generation_specs,
        positive_latents_dir=positive_latents_dir,
        output_dir=output_dir,
        model_path=model_path,
        text_encoder_path=text_encoder_path,
        device=device,
        with_audio=with_audio,
        audio_output_dir=audio_output_dir,
        inference_steps=inference_steps,
        guidance_scale=guidance_scale,
        negative_prompt=negative_prompt,
        seed=seed,
        use_first_frame_conditioning=use_first_frame_conditioning,
        save_previews=save_previews,
        preview_output_dir=preview_output_dir,
        load_text_encoder_in_8bit=load_text_encoder_in_8bit,
    )


def generate_negative_latents(  # noqa: PLR0913
    specs: list[NegativeLatentGenerationSpec],
    *,
    positive_latents_dir: str | Path,
    output_dir: str | Path,
    model_path: str | Path,
    text_encoder_path: str | Path,
    device: str = "cuda",
    with_audio: bool = False,
    audio_output_dir: str | Path | None = None,
    inference_steps: int = 30,
    guidance_scale: float = 4.0,
    negative_prompt: str = "",
    seed: int = 42,
    use_first_frame_conditioning: bool = False,
    save_previews: bool = False,
    preview_output_dir: str | Path | None = None,
    load_text_encoder_in_8bit: bool = False,
) -> None:
    """Generate synthetic negative latents directly into trainer-format latent files."""
    if with_audio and audio_output_dir is None:
        raise ValueError("audio_output_dir must be provided when with_audio=True")

    if not specs:
        return

    positive_latents_root = Path(positive_latents_dir)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    audio_output_root = None
    if with_audio and audio_output_dir is not None:
        audio_output_root = Path(audio_output_dir)
        audio_output_root.mkdir(parents=True, exist_ok=True)

    preview_root = None
    if save_previews:
        preview_root = Path(preview_output_dir) if preview_output_dir is not None else output_root.parent / "generated_negative_previews"
        preview_root.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating {len(specs):,} paired NSYNC negatives from prompts...")

    components = load_ltx_model(
        checkpoint_path=model_path,
        text_encoder_path=None,
        device=device,
        dtype=torch.bfloat16,
        with_video_vae_encoder=False,
        with_video_vae_decoder=save_previews,
        with_audio_vae_decoder=save_previews and with_audio,
        with_vocoder=save_previews and with_audio,
        with_text_encoder=False,
    )
    text_encoder = load_text_encoder(
        gemma_model_path=text_encoder_path,
        device=device,
        dtype=torch.bfloat16,
        load_in_8bit=load_text_encoder_in_8bit,
    )
    embeddings_processor = load_embeddings_processor(
        checkpoint_path=model_path,
        device=device,
        dtype=torch.bfloat16,
    )

    sampler = ValidationSampler(
        transformer=components.transformer,
        vae_decoder=components.video_vae_decoder,
        vae_encoder=None,
        text_encoder=text_encoder,
        audio_decoder=components.audio_vae_decoder,
        vocoder=components.vocoder,
        embeddings_processor=embeddings_processor,
    )

    audio_sample_rate = getattr(components.audio_vae_decoder, "sample_rate", None)

    for spec in specs:
        output_rel_path = Path(spec.output_rel_path)
        positive_latent_path = positive_latents_root / Path(spec.positive_media_path).with_suffix(".pt")
        if not positive_latent_path.is_file():
            raise FileNotFoundError(f"Positive latent file not found for negative generation: {positive_latent_path}")

        positive_latent_data = torch.load(positive_latent_path, map_location="cpu", weights_only=True)
        condition_latents = None
        if use_first_frame_conditioning:
            condition_latents = _normalize_video_latents(positive_latent_data)["latents"][:, :1].contiguous()

        generation_config = _build_generation_config(
            positive_latent_data,
            prompt=spec.prompt,
            negative_prompt=negative_prompt,
            inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            with_audio=with_audio,
            condition_latents=condition_latents,
        )

        generated = sampler.generate_latents(generation_config, device=device, decode_preview=save_previews)
        _save_generated_negative(output_root / output_rel_path, generated, fps=float(positive_latent_data["fps"]))

        if audio_output_root is not None and generated.audio_latents is not None:
            _save_generated_negative_audio(audio_output_root / output_rel_path, generated)

        if preview_root is not None and generated.preview_video is not None:
            preview_path = preview_root / output_rel_path.with_suffix(".mp4")
            save_video(
                generated.preview_video,
                preview_path,
                fps=float(positive_latent_data["fps"]),
                audio=generated.preview_audio if audio_sample_rate is not None else None,
                audio_sample_rate=audio_sample_rate if audio_sample_rate is not None else None,
            )


def _build_generation_config(
    positive_latent_data: dict[str, Any],
    *,
    prompt: str,
    negative_prompt: str,
    inference_steps: int,
    guidance_scale: float,
    seed: int,
    with_audio: bool,
    condition_latents: torch.Tensor | None = None,
) -> GenerationConfig:
    """Create a generation config aligned to the already-encoded positive sample shape."""
    latent_frames = int(positive_latent_data["num_frames"])
    latent_height = int(positive_latent_data["height"])
    latent_width = int(positive_latent_data["width"])
    fps = float(positive_latent_data["fps"])

    return GenerationConfig(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=latent_width * VIDEO_LATENT_SPATIAL_FACTOR,
        height=latent_height * VIDEO_LATENT_SPATIAL_FACTOR,
        num_frames=(latent_frames - 1) * VIDEO_LATENT_TEMPORAL_FACTOR + 1,
        frame_rate=fps,
        num_inference_steps=inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
        condition_latents=condition_latents,
        generate_audio=with_audio,
    )


def _save_generated_negative(output_path: Path, generated: GeneratedLatents, *, fps: float) -> None:
    """Persist generated negative video latents in trainer preprocessing format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "latents": generated.video_latents.cpu().contiguous(),
            "num_frames": generated.num_frames,
            "height": generated.height,
            "width": generated.width,
            "fps": fps,
        },
        output_path,
    )


def _save_generated_negative_audio(output_path: Path, generated: GeneratedLatents) -> None:
    """Persist generated negative audio latents in trainer preprocessing format."""
    if generated.audio_latents is None:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "latents": generated.audio_latents.cpu().contiguous(),
            "num_time_steps": generated.num_time_steps,
            "frequency_bins": generated.frequency_bins,
            "duration": generated.duration,
        },
        output_path,
    )


def _normalize_video_latents(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize saved video latents to non-patchified [C, F, H, W] format."""
    latents = data["latents"]
    if latents.dim() == 2:
        num_frames = int(data["num_frames"])
        height = int(data["height"])
        width = int(data["width"])
        latents = latents.reshape(num_frames, height, width, latents.shape[1]).permute(3, 0, 1, 2).contiguous()
        return {**data, "latents": latents}
    if latents.dim() != 4:
        raise ValueError(f"Unsupported video latent shape for negative generation: {tuple(latents.shape)}")
    return data


def _load_dataset_records(dataset_file: Path) -> list[dict[str, Any]]:
    """Load dataset metadata rows from CSV, JSON, or JSONL."""
    suffix = dataset_file.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(dataset_file).to_dict(orient="records")
    if suffix == ".json":
        with dataset_file.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, list):
            raise ValueError("JSON dataset file must contain a list of objects")
        return data
    if suffix == ".jsonl":
        records = []
        with dataset_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
        return records
    raise ValueError(f"Unsupported dataset file format: {dataset_file}")


def _is_missing_value(value: Any) -> bool:
    """Treat empty strings and NaN-like values as missing."""
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    return bool(pd.isna(value))
