#!/usr/bin/env python3

"""
Preprocess a video dataset by computing video clips latents and text captions embeddings.
This script provides a command-line interface for preprocessing video datasets by computing
latent representations of video clips and text embeddings of their captions. The preprocessed
data can be used to accelerate training of video generation models and to save GPU memory.
Basic usage:
    python scripts/process_dataset.py /path/to/dataset.json --resolution-buckets 768x768x49 \
        --model-path /path/to/ltx2.safetensors --text-encoder-path /path/to/gemma.safetensors
The dataset must be a CSV, JSON, or JSONL file with columns for captions and video paths.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import typer
from decode_latents import LatentsDecoder
from process_captions import CaptionsDataset, compute_captions_embeddings
from process_videos import _atomic_save, compute_latents, compute_scaled_resolution_buckets, parse_resolution_buckets
from rich.console import Console

from ltx_core.quantization import QuantizationPolicy
from ltx_trainer import logger
from ltx_trainer.gpu_utils import free_gpu_memory_context

if TYPE_CHECKING:
    import torch

console = Console()

app = typer.Typer(
    pretty_exceptions_enable=False,
    no_args_is_help=True,
    help="Preprocess a video dataset by computing video clips latents and text captions embeddings. "
    "The dataset must be a CSV, JSON, or JSONL file with columns for captions and video paths.",
)


def preprocess_dataset(  # noqa: PLR0912, PLR0913, PLR0915
    dataset_file: str,
    caption_column: str,
    video_column: str,
    resolution_buckets: list[tuple[int, int, int]],
    batch_size: int,
    output_dir: str | None,
    lora_trigger: str | None,
    vae_tiling: bool,
    decode: bool,
    model_path: str,
    text_encoder_path: str,
    device: str,
    remove_llm_prefixes: bool = False,
    reference_column: str | None = None,
    reference_downscale_factor: int = 1,
    with_audio: bool = False,
    mixed_audio: bool = False,
    audio_min_rms_db: float = -60.0,
    audio_min_active_ratio: float = 0.01,
    audio_activity_window_ms: float = 100.0,
    generate_negatives: bool = False,
    negative_videos_dir: str | None = None,
    negative_latents_dir: str = "negative_latents",
    negative_prompt: str = "",
    negative_distilled_lora: str | None = None,
    negative_distilled_lora_strength: float = 1.0,
    negative_inference_steps: int = 8,
    negative_guidance_scale: float = 1.0,
    negative_backend: str = "native",
    negative_quantization: str = "auto",
    comfy_server: str = "http://127.0.0.1:8188",
    comfy_checkpoint_name: str | None = None,
    comfy_text_encoder_name: str | None = None,
    comfy_distilled_lora_name: str | None = None,
    comfy_sampler_name: str = "euler_cfg_pp",
    comfy_timeout_seconds: float = 1800.0,
    negative_seed: int = 42,
    load_text_encoder_in_8bit: bool = False,
    overwrite: bool = False,
) -> None:
    """Run the preprocessing pipeline with the given arguments."""
    # Validate dataset file
    _validate_dataset_file(dataset_file)

    # Set up output directories
    output_base = Path(output_dir) if output_dir else Path(dataset_file).parent / ".precomputed"
    conditions_dir = output_base / "conditions"
    latents_dir = output_base / "latents"

    if lora_trigger:
        logger.info(f'LoRA trigger word "{lora_trigger}" will be prepended to all captions')

    with free_gpu_memory_context():
        # Process captions using the dedicated function
        compute_captions_embeddings(
            dataset_file=dataset_file,
            output_dir=str(conditions_dir),
            model_path=model_path,
            text_encoder_path=text_encoder_path,
            caption_column=caption_column,
            media_column=video_column,
            lora_trigger=lora_trigger,
            remove_llm_prefixes=remove_llm_prefixes,
            batch_size=batch_size,
            device=device,
            load_in_8bit=load_text_encoder_in_8bit,
            overwrite=overwrite,
        )

    # Process videos using the dedicated function
    audio_latents_dir = None
    if with_audio:
        logger.info("Audio preprocessing enabled - will extract and encode audio from videos")
        audio_latents_dir = output_base / "audio_latents"

    with free_gpu_memory_context():
        compute_latents(
            dataset_file=dataset_file,
            video_column=video_column,
            resolution_buckets=resolution_buckets,
            output_dir=str(latents_dir),
            model_path=model_path,
            batch_size=batch_size,
            device=device,
            vae_tiling=vae_tiling,
            with_audio=with_audio,
            audio_output_dir=str(audio_latents_dir) if audio_latents_dir else None,
            audio_min_rms_db=audio_min_rms_db,
            audio_min_active_ratio=audio_min_active_ratio,
            audio_activity_window_ms=audio_activity_window_ms,
            allow_missing_audio=mixed_audio,
            overwrite=overwrite,
        )

        if generate_negatives:
            logger.info("Generating synthetic negative latents with one-stage distilled-LoRA sampling...")
            if negative_videos_dir is not None:
                logger.warning(
                    "--negative-videos-dir is ignored by the fast direct-latent negative generator; "
                    "no intermediate videos are written."
                )

            if negative_backend == "comfy":
                generate_negative_latents_comfy(
                    dataset_file=dataset_file,
                    caption_column=caption_column,
                    media_column=video_column,
                    output_dir=output_base / negative_latents_dir,
                    model_path=model_path,
                    text_encoder_path=text_encoder_path,
                    resolution_buckets=resolution_buckets,
                    negative_prompt=negative_prompt,
                    distilled_lora=negative_distilled_lora,
                    distilled_lora_strength=negative_distilled_lora_strength,
                    num_inference_steps=negative_inference_steps,
                    guidance_scale=negative_guidance_scale,
                    seed=negative_seed,
                    lora_trigger=lora_trigger,
                    remove_llm_prefixes=remove_llm_prefixes,
                    comfy_server=comfy_server,
                    comfy_checkpoint_name=comfy_checkpoint_name,
                    comfy_text_encoder_name=comfy_text_encoder_name,
                    comfy_distilled_lora_name=comfy_distilled_lora_name,
                    comfy_sampler_name=comfy_sampler_name,
                    comfy_timeout_seconds=comfy_timeout_seconds,
                    overwrite=overwrite,
                )
            else:
                generate_negative_latents(
                    dataset_file=dataset_file,
                    caption_column=caption_column,
                    media_column=video_column,
                    conditions_dir=conditions_dir,
                    output_dir=output_base / negative_latents_dir,
                    model_path=model_path,
                    text_encoder_path=text_encoder_path,
                    resolution_buckets=resolution_buckets,
                    device=device,
                    negative_prompt=negative_prompt,
                    distilled_lora=negative_distilled_lora,
                    distilled_lora_strength=negative_distilled_lora_strength,
                    num_inference_steps=negative_inference_steps,
                    guidance_scale=negative_guidance_scale,
                    quantization=negative_quantization,
                    seed=negative_seed,
                    load_text_encoder_in_8bit=load_text_encoder_in_8bit,
                    overwrite=overwrite,
                )

        # Process reference videos if reference_column is provided
        if reference_column:
            # Validate: scaled references with multiple buckets can cause ambiguous bucket matching
            if reference_downscale_factor > 1 and len(resolution_buckets) > 1:
                raise ValueError(
                    "When using --reference-downscale-factor > 1, only a single resolution bucket is supported. "
                    "Using multiple buckets with scaled references can cause ambiguous bucket matching "
                    "(e.g., a 512x256 reference could match either the scaled-down 1024x512 bucket or the 512x256 "
                    "bucket). Please use a single resolution bucket or set --reference-downscale-factor to 1."
                )

            # Calculate and validate scaled resolution buckets for reference videos
            reference_buckets = compute_scaled_resolution_buckets(resolution_buckets, reference_downscale_factor)

            if reference_downscale_factor > 1:
                logger.info(
                    f"Processing reference videos for IC-LoRA training at 1/{reference_downscale_factor} resolution..."
                )
                logger.info(f"Reference resolution buckets: {reference_buckets}")
            else:
                logger.info("Processing reference videos for IC-LoRA training...")

            reference_latents_dir = output_base / "reference_latents"

            compute_latents(
                dataset_file=dataset_file,
                main_media_column=video_column,
                video_column=reference_column,
                resolution_buckets=reference_buckets,
                output_dir=str(reference_latents_dir),
                model_path=model_path,
                batch_size=batch_size,
                device=device,
                vae_tiling=vae_tiling,
                overwrite=overwrite,
            )

    # Handle decoding if requested (for verification)
    if decode:
        logger.info("Decoding latents for verification...")

        decoder = LatentsDecoder(
            model_path=model_path,
            device=device,
            vae_tiling=vae_tiling,
            with_audio=with_audio,
        )
        decoder.decode(latents_dir, output_base / "decoded_videos")

        # Also decode reference videos if they exist
        if reference_column:
            reference_latents_dir = output_base / "reference_latents"
            if reference_latents_dir.exists():
                logger.info("Decoding reference videos...")
                decoder.decode(reference_latents_dir, output_base / "decoded_reference_videos")

        # Decode audio latents if they exist
        if with_audio and audio_latents_dir and audio_latents_dir.exists():
            logger.info("Decoding audio latents...")
            decoder.decode_audio(audio_latents_dir, output_base / "decoded_audio")

    # Print summary
    logger.info(f"Dataset preprocessing complete! Results saved to {output_base}")
    if reference_column:
        logger.info("Reference videos processed and saved to reference_latents/ directory for IC-LoRA training")
    if with_audio:
        logger.info("Audio latents saved to audio_latents/ directory for audio-video training")
        if mixed_audio:
            logger.info("Mixed-audio training enabled: samples without usable audio latents are kept as video-only")
    if generate_negatives:
        logger.info(f"Synthetic negative latents saved to {negative_latents_dir}/ directory for NSYNC training")


def _validate_dataset_file(dataset_path: str) -> None:
    """Validate that the dataset file exists and has the correct format."""
    dataset_file = Path(dataset_path)

    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {dataset_file}")

    if not dataset_file.is_file():
        raise ValueError(f"Dataset path must be a file, not a directory: {dataset_file}")

    if dataset_file.suffix.lower() not in [".csv", ".json", ".jsonl"]:
        raise ValueError(f"Dataset file must be CSV, JSON, or JSONL format: {dataset_file}")


def generate_negative_latents(  # noqa: PLR0913, PLR0915
    dataset_file: str,
    caption_column: str,
    media_column: str,
    conditions_dir: Path,
    output_dir: Path,
    model_path: str,
    text_encoder_path: str,
    resolution_buckets: list[tuple[int, int, int]],
    device: str,
    negative_prompt: str,
    distilled_lora: str | None,
    distilled_lora_strength: float,
    num_inference_steps: int,
    guidance_scale: float,
    quantization: str,
    seed: int,
    load_text_encoder_in_8bit: bool,
    overwrite: bool,
) -> None:
    """Generate one synthetic negative latent per caption without decoding intermediate videos."""
    import torch  # noqa: PLC0415
    from accelerate import PartialState  # noqa: PLC0415

    from ltx_core.components.guiders import MultiModalGuiderParams  # noqa: PLC0415
    from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps  # noqa: PLC0415
    from ltx_pipelines.ti2vid_one_stage import TI2VidOneStagePipeline  # noqa: PLC0415
    from ltx_pipelines.utils.constants import DISTILLED_SIGMAS  # noqa: PLC0415
    from ltx_trainer.model_loader import load_embeddings_processor, load_text_encoder  # noqa: PLC0415

    if distilled_lora is None:
        raise ValueError("distilled_lora is required for synthetic negative generation")

    if len(resolution_buckets) > 1:
        logger.warning(
            "Negative generation uses the first resolution bucket. "
            "Use a single bucket when generated negatives must exactly mirror mixed-bucket training data."
        )

    frames, height, width = resolution_buckets[0]
    output_dir.mkdir(parents=True, exist_ok=True)
    conditions_dir.mkdir(parents=True, exist_ok=True)
    distilled_lora_path = Path(distilled_lora).expanduser().resolve()
    if not distilled_lora_path.is_file():
        raise FileNotFoundError(f"Distilled LoRA not found: {distilled_lora_path}")

    captions = CaptionsDataset(
        dataset_file=dataset_file,
        caption_column=caption_column,
        media_column=media_column,
        lora_trigger=None,
        remove_llm_prefixes=False,
    )

    state = PartialState()

    def _output_file_for_item(item: dict) -> Path:
        return output_dir / Path(item["output_path"])

    todo = [
        idx
        for idx in range(state.process_index, len(captions), state.num_processes)
        if overwrite or not _output_file_for_item(captions[idx]).is_file()
    ]
    if not todo:
        logger.info(f"Rank {state.process_index}/{state.num_processes}: no negative latents to generate")
        return
    logger.info(
        f"Rank {state.process_index}/{state.num_processes}: generating {len(todo):,} "
        f"of {len(captions):,} negative latents"
    )

    torch_device = torch.device(device)
    quantization_policy = _build_negative_quantization_policy(quantization, model_path)

    pipeline = TI2VidOneStagePipeline(
        checkpoint_path=model_path,
        gemma_root=text_encoder_path,
        loras=[
            LoraPathStrengthAndSDOps(
                str(distilled_lora_path),
                distilled_lora_strength,
                LTXV_LORA_COMFY_RENAMING_MAP,
            )
        ],
        device=torch_device,
        quantization=quantization_policy,
    )
    sigmas = None
    if num_inference_steps == len(DISTILLED_SIGMAS) - 1:
        sigmas = DISTILLED_SIGMAS
    else:
        logger.warning(
            "Using %s custom negative inference steps with the generic scheduler instead of the default "
            "distilled sigma schedule.",
            num_inference_steps,
        )

    embeddings_processor = load_embeddings_processor(model_path, device=torch_device, dtype=torch.bfloat16).eval()
    negative_video_context = None
    negative_audio_context = None
    if not math.isclose(guidance_scale, 1.0):
        text_encoder = load_text_encoder(
            text_encoder_path,
            device=torch_device,
            dtype=torch.bfloat16,
            load_in_8bit=load_text_encoder_in_8bit,
        ).eval()
        with torch.inference_mode():
            neg_hidden_states, neg_attention_mask = text_encoder.encode(negative_prompt, padding_side="left")
            neg_out = embeddings_processor.process_hidden_states(neg_hidden_states, neg_attention_mask, "left")
            negative_video_context = neg_out.video_encoding
            negative_audio_context = neg_out.audio_encoding
        del text_encoder

    embeddings_processor.feature_extractor = None

    with torch.inference_mode(), pipeline.stage.model_context() as transformer:
        for idx in todo:
            item = captions[idx]
            output_file = _output_file_for_item(item)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            condition_file = conditions_dir / Path(item["output_path"])
            video_context, audio_context = _load_condition_context(
                condition_file=condition_file,
                embeddings_processor=embeddings_processor,
                device=torch_device,
            )

            video_latent, _, _ = pipeline.generate_latents_from_context(
                v_context_p=video_context,
                a_context_p=audio_context,
                v_context_n=negative_video_context,
                a_context_n=negative_audio_context,
                seed=seed + idx,
                height=height,
                width=width,
                num_frames=frames,
                frame_rate=24.0,
                num_inference_steps=num_inference_steps,
                video_guider_params=MultiModalGuiderParams(cfg_scale=guidance_scale),
                audio_guider_params=MultiModalGuiderParams(cfg_scale=1.0),
                images=[],
                sigmas=sigmas,
                generate_audio=False,
                transformer=transformer,
            )

            latent = video_latent.squeeze(0).detach().cpu().contiguous()
            _atomic_save(
                {
                    "latents": latent,
                    "num_frames": latent.shape[1],
                    "height": latent.shape[2],
                    "width": latent.shape[3],
                    "fps": 24.0,
                },
                output_file,
            )


def generate_negative_latents_comfy(  # noqa: PLR0913, PLR0915
    dataset_file: str,
    caption_column: str,
    media_column: str,
    output_dir: Path,
    model_path: str,
    text_encoder_path: str,
    resolution_buckets: list[tuple[int, int, int]],
    negative_prompt: str,
    distilled_lora: str | None,
    distilled_lora_strength: float,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
    lora_trigger: str | None,
    remove_llm_prefixes: bool,
    comfy_server: str,
    comfy_checkpoint_name: str | None,
    comfy_text_encoder_name: str | None,
    comfy_distilled_lora_name: str | None,
    comfy_sampler_name: str,
    comfy_timeout_seconds: float,
    overwrite: bool,
) -> None:
    """Generate one synthetic negative latent per caption through a running ComfyUI server."""
    from accelerate import PartialState  # noqa: PLC0415

    from ltx_pipelines.utils.constants import DISTILLED_SIGMA_VALUES  # noqa: PLC0415
    from ltx_trainer.comfy_negative_backend import (  # noqa: PLC0415
        build_ltxv_negative_workflow,
        comfy_latent_bytes_to_trainer_payload,
        run_comfy_workflow_and_fetch_latent,
    )

    if distilled_lora is None:
        raise ValueError("distilled_lora is required for synthetic negative generation")

    if len(resolution_buckets) > 1:
        logger.warning(
            "Comfy negative generation uses the first resolution bucket. "
            "Use a single bucket when generated negatives must exactly mirror mixed-bucket training data."
        )

    frames, height, width = resolution_buckets[0]
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_name = comfy_checkpoint_name or Path(model_path).name
    text_encoder_name = comfy_text_encoder_name or Path(text_encoder_path).name
    lora_name = comfy_distilled_lora_name or Path(distilled_lora).name

    captions = CaptionsDataset(
        dataset_file=dataset_file,
        caption_column=caption_column,
        media_column=media_column,
        lora_trigger=lora_trigger,
        remove_llm_prefixes=remove_llm_prefixes,
    )

    state = PartialState()

    def _output_file_for_item(item: dict) -> Path:
        return output_dir / Path(item["output_path"])

    todo = [
        idx
        for idx in range(state.process_index, len(captions), state.num_processes)
        if overwrite or not _output_file_for_item(captions[idx]).is_file()
    ]
    if not todo:
        logger.info(f"Rank {state.process_index}/{state.num_processes}: no Comfy negative latents to generate")
        return
    logger.info(
        f"Rank {state.process_index}/{state.num_processes}: generating {len(todo):,} "
        f"of {len(captions):,} negative latents through ComfyUI at {comfy_server}"
    )
    logger.info(
        "Using ComfyUI model names: checkpoint=%s, text_encoder=%s, distilled_lora=%s",
        checkpoint_name,
        text_encoder_name,
        lora_name,
    )

    sigmas = DISTILLED_SIGMA_VALUES if num_inference_steps == len(DISTILLED_SIGMA_VALUES) - 1 else None
    if sigmas is None:
        logger.warning(
            "Using %s custom negative inference steps with Comfy's LTXV scheduler instead of the default "
            "distilled sigma schedule.",
            num_inference_steps,
        )

    for idx in todo:
        item = captions[idx]
        output_file = _output_file_for_item(item)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        workflow = build_ltxv_negative_workflow(
            checkpoint_name=checkpoint_name,
            text_encoder_name=text_encoder_name,
            prompt=item["prompt"],
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            frames=frames,
            frame_rate=24.0,
            seed=seed + idx,
            guidance_scale=guidance_scale,
            distilled_lora_name=lora_name,
            distilled_lora_strength=distilled_lora_strength,
            sampler_name=comfy_sampler_name,
            filename_prefix=f"ltx_trainer_negatives/{state.process_index:03d}_{idx:08d}",
            sigmas=sigmas,
            num_inference_steps=num_inference_steps,
        )
        latent_bytes = run_comfy_workflow_and_fetch_latent(
            workflow,
            server_url=comfy_server,
            timeout_seconds=comfy_timeout_seconds,
        )
        _atomic_save(
            comfy_latent_bytes_to_trainer_payload(latent_bytes, frame_rate=24.0),
            output_file,
        )


def _load_condition_context(
    condition_file: Path,
    embeddings_processor: object,
    device: "torch.device",
) -> tuple["torch.Tensor", "torch.Tensor | None"]:
    import torch  # noqa: PLC0415

    from ltx_core.text_encoders.gemma import convert_to_additive_mask  # noqa: PLC0415

    if not condition_file.is_file():
        raise FileNotFoundError(f"Caption embedding file not found: {condition_file}")

    data = torch.load(condition_file, map_location="cpu")
    video_features = data["video_prompt_embeds"]
    if video_features.dim() == 2:
        video_features = video_features.unsqueeze(0)
    video_features = video_features.to(device=device, dtype=torch.bfloat16)

    audio_features = data.get("audio_prompt_embeds")
    if audio_features is not None:
        if audio_features.dim() == 2:
            audio_features = audio_features.unsqueeze(0)
        audio_features = audio_features.to(device=device, dtype=torch.bfloat16)
    elif getattr(embeddings_processor, "audio_connector", None) is not None:
        audio_features = video_features

    attention_mask = data["prompt_attention_mask"]
    if attention_mask.dim() == 1:
        attention_mask = attention_mask.unsqueeze(0)
    attention_mask = attention_mask.to(device=device)
    additive_mask = convert_to_additive_mask(attention_mask, video_features.dtype)

    video_context, audio_context, _ = embeddings_processor.create_embeddings(
        video_features,
        audio_features,
        additive_mask,
    )
    return video_context, audio_context


def _build_negative_quantization_policy(kind: str, checkpoint_path: str) -> QuantizationPolicy | None:
    from safetensors import safe_open  # noqa: PLC0415

    from ltx_pipelines.utils.quantization_factory import QuantizationKind  # noqa: PLC0415

    normalized = kind.lower()
    if normalized == "none":
        return None
    if normalized == "auto":
        normalized = "none"
        try:
            with safe_open(checkpoint_path, framework="pt") as f:
                keys = list(f.keys())
            if any(key.endswith(".weight_scale") for key in keys):
                normalized = QuantizationKind.FP8_SCALED_MM.value
            elif "fp8" in Path(checkpoint_path).name.lower():
                normalized = QuantizationKind.FP8_CAST.value
        except Exception as e:
            logger.warning("Could not inspect checkpoint for automatic negative quantization: %s", e)
        if normalized == "none":
            logger.info("Negative generation quantization auto-detect: no fp8 checkpoint markers found")
            return None

    quantization_kind = QuantizationKind(normalized)
    logger.info("Using %s quantization for negative generation", quantization_kind.value)
    return quantization_kind.to_policy(checkpoint_path)


@app.command()
def main(  # noqa: PLR0913
    dataset_path: str = typer.Argument(
        ...,
        help="Path to metadata file (CSV/JSON/JSONL) containing captions and video paths",
    ),
    resolution_buckets: str = typer.Option(
        ...,
        help='Resolution buckets in format "WxHxF;WxHxF;..." (e.g. "768x768x25;512x512x49")',
    ),
    model_path: str = typer.Option(
        ...,
        help="Path to LTX-2 checkpoint (.safetensors file)",
    ),
    text_encoder_path: str = typer.Option(
        ...,
        help="Path to a single Gemma .safetensors file or model directory",
    ),
    caption_column: str = typer.Option(
        default="caption",
        help="Column name containing captions in the dataset JSON/JSONL/CSV file",
    ),
    video_column: str = typer.Option(
        default="media_path",
        help="Column name containing video paths in the dataset JSON/JSONL/CSV file",
    ),
    batch_size: int = typer.Option(
        default=1,
        help="Batch size for preprocessing",
    ),
    device: str = typer.Option(
        default="cuda",
        help="Device to use for computation",
    ),
    vae_tiling: bool = typer.Option(
        default=False,
        help="Enable VAE tiling for larger video resolutions",
    ),
    output_dir: str | None = typer.Option(
        default=None,
        help="Output directory (defaults to .precomputed in dataset directory)",
    ),
    lora_trigger: str | None = typer.Option(
        default=None,
        help="Optional trigger word to prepend to each caption (activates the LoRA during inference)",
    ),
    decode: bool = typer.Option(
        default=False,
        help="Decode and save latents after encoding (videos and audio) for verification",
    ),
    remove_llm_prefixes: bool = typer.Option(
        default=False,
        help="Remove LLM prefixes from captions",
    ),
    reference_column: str | None = typer.Option(
        default=None,
        help="Column name containing reference video paths (for video-to-video training)",
    ),
    with_audio: bool = typer.Option(
        default=False,
        help="Extract and encode audio from video files",
    ),
    mixed_audio: bool = typer.Option(
        default=False,
        help="Keep videos whose audio is missing or silent as video-only samples for mixed-audio training",
    ),
    audio_min_rms_db: float = typer.Option(
        default=-60.0,
        help="Minimum RMS level in dBFS for an audio track to count as non-silent",
    ),
    audio_min_active_ratio: float = typer.Option(
        default=0.01,
        help="Minimum ratio of audio windows above --audio-min-rms-db required to save audio latents",
    ),
    audio_activity_window_ms: float = typer.Option(
        default=100.0,
        help="Audio activity analysis window size in milliseconds",
    ),
    generate_negatives: bool = typer.Option(
        default=False,
        help="Generate one synthetic negative latent per caption with one-stage distilled-LoRA sampling for NSYNC",
    ),
    negative_videos_dir: str | None = typer.Option(
        default=None,
        help="Deprecated: ignored by the fast direct-latent negative generator",
    ),
    negative_latents_dir: str = typer.Option(
        default="negative_latents",
        help="Directory name under the precomputed root for generated negative latents",
    ),
    negative_prompt: str = typer.Option(
        default="",
        help="Negative prompt used only when --negative-guidance-scale is not 1.0",
    ),
    negative_distilled_lora: str | None = typer.Option(
        default=None,
        help="Distilled LoRA path used for fast one-stage synthetic negative generation",
    ),
    negative_distilled_lora_strength: float = typer.Option(
        default=1.0,
        help="Strength for --negative-distilled-lora",
    ),
    negative_inference_steps: int = typer.Option(
        default=8,
        help="Number of inference steps for synthetic negative latent generation",
    ),
    negative_guidance_scale: float = typer.Option(
        default=1.0,
        help="CFG scale for synthetic negative generation. Default 1.0 disables CFG for speed",
    ),
    negative_backend: str = typer.Option(
        default="native",
        help="Backend for synthetic negative generation: native or comfy",
    ),
    negative_quantization: str = typer.Option(
        default="auto",
        help=(
            "Quantization for synthetic negative generation: auto, none, fp8-cast, or fp8-scaled-mm. "
            "Use auto for fp8 checkpoint files. Ignored when --negative-backend comfy."
        ),
    ),
    comfy_server: str = typer.Option(
        default="http://127.0.0.1:8188",
        help="ComfyUI server URL used when --negative-backend comfy",
    ),
    comfy_checkpoint_name: str | None = typer.Option(
        default=None,
        help="ComfyUI checkpoints model name for the LTX checkpoint. Defaults to basename of --model-path.",
    ),
    comfy_text_encoder_name: str | None = typer.Option(
        default=None,
        help="ComfyUI text_encoders model name for Gemma. Defaults to basename of --text-encoder-path.",
    ),
    comfy_distilled_lora_name: str | None = typer.Option(
        default=None,
        help="ComfyUI loras model name for the distilled LoRA. Defaults to basename of --negative-distilled-lora.",
    ),
    comfy_sampler_name: str = typer.Option(
        default="euler_cfg_pp",
        help="ComfyUI sampler name used when --negative-backend comfy",
    ),
    comfy_timeout_seconds: float = typer.Option(
        default=1800.0,
        help="Seconds to wait for each ComfyUI negative-latent prompt before timing out",
    ),
    negative_seed: int = typer.Option(
        default=42,
        help="Base random seed for synthetic negative latent generation",
    ),
    load_text_encoder_in_8bit: bool = typer.Option(
        default=False,
        help="Load the Gemma text encoder in 8-bit precision to save GPU memory (requires bitsandbytes)",
    ),
    reference_downscale_factor: int = typer.Option(
        default=1,
        help="Downscale factor for reference video resolution. When > 1, reference videos are processed at "
        "1/n resolution (e.g., 2 means half resolution). Used for efficient IC-LoRA training.",
    ),
    overwrite: bool = typer.Option(
        default=False,
        help="Re-compute every item even if its output exists. Use when rerunning with "
        "changed parameters (different model, resolution, etc.) so stale outputs are replaced.",
    ),
) -> None:
    """Preprocess a video dataset by computing and saving latents and text embeddings.
    For multi-GPU preprocessing, invoke under ``accelerate launch`` - each process
    will handle an interleaved shard of the dataset.
    The dataset must be a CSV, JSON, or JSONL file with columns for captions and video paths.
    This script is designed for LTX-2 models which use the Gemma text encoder.
    Examples:
        # Process a dataset with LTX-2 model
        python scripts/process_dataset.py dataset.json --resolution-buckets 768x768x25 \\
            --model-path /path/to/ltx2.safetensors --text-encoder-path /path/to/gemma.safetensors
        # Process dataset with custom column names
        python scripts/process_dataset.py dataset.json --resolution-buckets 768x768x25 \\
            --model-path /path/to/ltx2.safetensors --text-encoder-path /path/to/gemma.safetensors \\
            --caption-column "text" --video-column "video_path"
        # Process dataset with reference videos for IC-LoRA training
        python scripts/process_dataset.py dataset.json --resolution-buckets 768x768x25 \\
            --model-path /path/to/ltx2.safetensors --text-encoder-path /path/to/gemma.safetensors \\
            --reference-column "reference_path"
        # Process dataset with scaled reference videos (half resolution) for efficient IC-LoRA
        python scripts/process_dataset.py dataset.json --resolution-buckets 768x768x25 \\
            --model-path /path/to/ltx2.safetensors --text-encoder-path /path/to/gemma.safetensors \\
            --reference-column "reference_path" --reference-downscale-factor 2
        # Process dataset with audio for audio-video training
        python scripts/process_dataset.py dataset.json --resolution-buckets 768x512x97 \\
            --model-path /path/to/ltx2.safetensors --text-encoder-path /path/to/gemma.safetensors \\
            --with-audio
    """
    parsed_resolution_buckets = parse_resolution_buckets(resolution_buckets)

    if len(parsed_resolution_buckets) > 1:
        logger.warning(
            "Using multiple resolution buckets. "
            "When training with multiple resolution buckets, you must use a batch size of 1."
        )

    # Validate reference_downscale_factor
    if reference_downscale_factor < 1:
        raise typer.BadParameter("--reference-downscale-factor must be >= 1")

    if reference_downscale_factor > 1 and not reference_column:
        logger.warning("--reference-downscale-factor specified but no --reference-column provided. Ignoring.")
    if mixed_audio and not with_audio:
        raise typer.BadParameter("--mixed-audio requires --with-audio")
    if generate_negatives and negative_distilled_lora is None:
        raise typer.BadParameter("--negative-distilled-lora is required when --generate-negatives is set")
    if negative_inference_steps < 1:
        raise typer.BadParameter("--negative-inference-steps must be >= 1")
    if negative_backend not in {"native", "comfy"}:
        raise typer.BadParameter("--negative-backend must be one of: native, comfy")
    if negative_quantization not in {"auto", "none", "fp8-cast", "fp8-scaled-mm"}:
        raise typer.BadParameter("--negative-quantization must be one of: auto, none, fp8-cast, fp8-scaled-mm")
    if comfy_timeout_seconds <= 0:
        raise typer.BadParameter("--comfy-timeout-seconds must be > 0")

    preprocess_dataset(
        dataset_file=dataset_path,
        caption_column=caption_column,
        video_column=video_column,
        resolution_buckets=parsed_resolution_buckets,
        batch_size=batch_size,
        output_dir=output_dir,
        lora_trigger=lora_trigger,
        vae_tiling=vae_tiling,
        decode=decode,
        model_path=model_path,
        text_encoder_path=text_encoder_path,
        device=device,
        remove_llm_prefixes=remove_llm_prefixes,
        reference_column=reference_column,
        reference_downscale_factor=reference_downscale_factor,
        with_audio=with_audio,
        mixed_audio=mixed_audio,
        audio_min_rms_db=audio_min_rms_db,
        audio_min_active_ratio=audio_min_active_ratio,
        audio_activity_window_ms=audio_activity_window_ms,
        generate_negatives=generate_negatives,
        negative_videos_dir=negative_videos_dir,
        negative_latents_dir=negative_latents_dir,
        negative_prompt=negative_prompt,
        negative_distilled_lora=negative_distilled_lora,
        negative_distilled_lora_strength=negative_distilled_lora_strength,
        negative_inference_steps=negative_inference_steps,
        negative_guidance_scale=negative_guidance_scale,
        negative_backend=negative_backend,
        negative_quantization=negative_quantization,
        comfy_server=comfy_server,
        comfy_checkpoint_name=comfy_checkpoint_name,
        comfy_text_encoder_name=comfy_text_encoder_name,
        comfy_distilled_lora_name=comfy_distilled_lora_name,
        comfy_sampler_name=comfy_sampler_name,
        comfy_timeout_seconds=comfy_timeout_seconds,
        negative_seed=negative_seed,
        load_text_encoder_in_8bit=load_text_encoder_in_8bit,
        overwrite=overwrite,
    )


if __name__ == "__main__":
    app()
