#!/usr/bin/env python3

"""
Preprocess a video dataset by computing video clips latents and text captions embeddings.
This script provides a command-line interface for preprocessing video datasets by computing
latent representations of video clips and text embeddings of their captions. The preprocessed
data can be used to accelerate training of video generation models and to save GPU memory.
Basic usage:
    python scripts/process_dataset.py /path/to/dataset.json --resolution-buckets 768x768x49 \
        --model-path /path/to/ltx2.safetensors --text-encoder-path /path/to/gemma
The dataset must be a CSV, JSON, or JSONL file with columns for captions and video paths.
"""

import tempfile
from pathlib import Path

import typer
from decode_latents import LatentsDecoder
from process_captions import CaptionEmbeddingTask, compute_caption_embeddings_from_tasks, compute_captions_embeddings
from process_videos import compute_latents, compute_scaled_resolution_buckets, parse_resolution_buckets
from rich.console import Console

from ltx_trainer import logger
from ltx_trainer.gpu_utils import free_gpu_memory_context
from ltx_trainer.negative_generation import (
    NegativeLatentGenerationSpec,
    generate_negative_latents,
    generate_missing_negative_latents,
    load_negative_sample_specs,
    split_negative_sample_specs,
    write_negative_media_subset,
)
from ltx_trainer.nsync_manifest import (
    NSYNC_MANIFEST_FILENAME,
    build_nsync_manifest,
    filter_advanced_nsync_specs,
    load_advanced_nsync_specs,
    write_nsync_manifest,
)

console = Console()

DEFAULT_NEGATIVE_PROMPT = "worst quality, inconsistent motion, blurry, jittery, distorted"

app = typer.Typer(
    pretty_exceptions_enable=False,
    no_args_is_help=True,
    help="Preprocess a video dataset by computing video clips latents and text captions embeddings. "
    "The dataset must be a CSV, JSON, or JSONL file with columns for captions and video paths.",
)


def preprocess_dataset(  # noqa: PLR0913
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
    load_text_encoder_in_8bit: bool = False,
    negative_caption_column: str = "negative_caption",
    negative_media_column: str = "negative_media_path",
    negative_inference_steps: int = 30,
    negative_guidance_scale: float = 4.0,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    negative_seed: int = 42,
    save_generated_negatives: bool = False,
) -> None:
    """Run the preprocessing pipeline with the given arguments."""
    # Validate dataset file
    _validate_dataset_file(dataset_file)

    # Set up output directories
    output_base = Path(output_dir) if output_dir else Path(dataset_file).parent / ".precomputed"
    conditions_dir = output_base / "conditions"
    latents_dir = output_base / "latents"
    negative_conditions_dir = output_base / "negative_conditions"
    negative_latents_dir = output_base / "negative_latents"
    negative_audio_latents_dir = output_base / "negative_audio_latents"

    advanced_nsync_specs = load_advanced_nsync_specs(
        dataset_file,
        media_column=video_column,
        legacy_negative_caption_column=negative_caption_column,
        legacy_negative_media_column=negative_media_column,
    )
    negative_specs = []
    manual_negative_specs = []
    generated_negative_specs = []
    if advanced_nsync_specs is None:
        negative_specs = load_negative_sample_specs(
            dataset_file,
            media_column=video_column,
            negative_caption_column=negative_caption_column,
            negative_media_column=negative_media_column,
        )
        manual_negative_specs, generated_negative_specs = split_negative_sample_specs(negative_specs)

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
        )
        if negative_specs:
            compute_captions_embeddings(
                dataset_file=dataset_file,
                output_dir=str(negative_conditions_dir),
                model_path=model_path,
                text_encoder_path=text_encoder_path,
                caption_column=negative_caption_column,
                media_column=video_column,
                lora_trigger=lora_trigger,
                remove_llm_prefixes=remove_llm_prefixes,
                batch_size=batch_size,
                device=device,
                load_in_8bit=load_text_encoder_in_8bit,
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
            )

        if manual_negative_specs:
            logger.info(f"Processing {len(manual_negative_specs):,} user-supplied NSYNC negative media files...")
            with tempfile.TemporaryDirectory(prefix="ltx-negative-media-") as temp_dir:
                subset_path = write_negative_media_subset(
                    manual_negative_specs,
                    output_path=Path(temp_dir) / "negative_media_subset.jsonl",
                    media_column=video_column,
                    negative_media_column=negative_media_column,
                )
                compute_latents(
                    dataset_file=subset_path,
                    main_media_column=video_column,
                    video_column=negative_media_column,
                    resolution_buckets=resolution_buckets,
                    output_dir=str(negative_latents_dir),
                    model_path=model_path,
                    batch_size=batch_size,
                    device=device,
                    vae_tiling=vae_tiling,
                    with_audio=with_audio,
                    audio_output_dir=str(negative_audio_latents_dir) if with_audio else None,
                )

        if generated_negative_specs:
            generate_missing_negative_latents(
                generated_negative_specs,
                positive_latents_dir=latents_dir,
                output_dir=negative_latents_dir,
                model_path=model_path,
                text_encoder_path=text_encoder_path,
                device=device,
                with_audio=with_audio,
                audio_output_dir=negative_audio_latents_dir if with_audio else None,
                inference_steps=negative_inference_steps,
                guidance_scale=negative_guidance_scale,
                negative_prompt=negative_prompt,
                seed=negative_seed,
                save_previews=save_generated_negatives,
                preview_output_dir=output_base / "generated_negative_previews",
                load_text_encoder_in_8bit=load_text_encoder_in_8bit,
            )

        if advanced_nsync_specs is not None:
            available_sample_rel_paths = {
                str(path.relative_to(latents_dir))
                for path in sorted(latents_dir.glob("**/*.pt"))
            }
            filtered_advanced_specs = filter_advanced_nsync_specs(
                advanced_nsync_specs,
                available_sample_rel_paths=available_sample_rel_paths,
            )
            filtered_count = len(advanced_nsync_specs) - len(filtered_advanced_specs)
            if filtered_count > 0:
                logger.warning(
                    f"Skipping {filtered_count} advanced NSYNC sample(s) without matching positive latents after "
                    "preprocessing"
                )

            manifest = build_nsync_manifest(
                filtered_advanced_specs,
                with_audio=with_audio,
            )

            negative_caption_tasks: list[CaptionEmbeddingTask] = []
            synthetic_generation_specs: list[NegativeLatentGenerationSpec] = []
            for sample_spec, manifest_sample in zip(filtered_advanced_specs, manifest.samples, strict=True):
                for negative_spec, manifest_negative in zip(sample_spec.negatives, manifest_sample.negatives, strict=True):
                    negative_caption_tasks.append(
                        CaptionEmbeddingTask(
                            prompt=negative_spec.caption,
                            output_path=manifest_negative.condition_rel_path,
                        )
                    )
                    if negative_spec.media == "synthetic":
                        if manifest_negative.latent_rel_path is None or negative_spec.prompt is None:
                            raise ValueError("Synthetic advanced NSYNC negatives must provide both output path and prompt")
                        synthetic_generation_specs.append(
                            NegativeLatentGenerationSpec(
                                positive_media_path=sample_spec.media_path,
                                output_rel_path=manifest_negative.latent_rel_path,
                                prompt=negative_spec.prompt,
                            )
                        )

            with free_gpu_memory_context():
                compute_caption_embeddings_from_tasks(
                    tasks=negative_caption_tasks,
                    output_dir=str(negative_conditions_dir),
                    model_path=model_path,
                    text_encoder_path=text_encoder_path,
                    lora_trigger=lora_trigger,
                    remove_llm_prefixes=remove_llm_prefixes,
                    batch_size=batch_size,
                    device=device,
                    load_in_8bit=load_text_encoder_in_8bit,
                )

            if synthetic_generation_specs:
                generate_negative_latents(
                    synthetic_generation_specs,
                    positive_latents_dir=latents_dir,
                    output_dir=negative_latents_dir,
                    model_path=model_path,
                    text_encoder_path=text_encoder_path,
                    device=device,
                    with_audio=with_audio,
                    audio_output_dir=negative_audio_latents_dir if with_audio else None,
                    inference_steps=negative_inference_steps,
                    guidance_scale=negative_guidance_scale,
                    negative_prompt=negative_prompt,
                    seed=negative_seed,
                    save_previews=save_generated_negatives,
                    preview_output_dir=output_base / "generated_negative_previews",
                    load_text_encoder_in_8bit=load_text_encoder_in_8bit,
                )

            manifest_path = write_nsync_manifest(manifest, output_base / NSYNC_MANIFEST_FILENAME)
            logger.info(f"Advanced NSYNC manifest saved to {manifest_path}")

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
    if negative_specs:
        logger.info("NSYNC negatives saved to negative_conditions/ and negative_latents/ directories")
        if with_audio:
            logger.info("NSYNC negative audio latents saved to negative_audio_latents/ directory")
    if advanced_nsync_specs is not None:
        logger.info("Advanced NSYNC metadata saved to nsync_manifest.json")


def _validate_dataset_file(dataset_path: str) -> None:
    """Validate that the dataset file exists and has the correct format."""
    dataset_file = Path(dataset_path)

    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {dataset_file}")

    if not dataset_file.is_file():
        raise ValueError(f"Dataset path must be a file, not a directory: {dataset_file}")

    if dataset_file.suffix.lower() not in [".csv", ".json", ".jsonl"]:
        raise ValueError(f"Dataset file must be CSV, JSON, or JSONL format: {dataset_file}")


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
        help="Path to Gemma text encoder directory",
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
    load_text_encoder_in_8bit: bool = typer.Option(
        default=False,
        help="Load the Gemma text encoder in 8-bit precision to save GPU memory (requires bitsandbytes)",
    ),
    negative_caption_column: str = typer.Option(
        default="negative_caption",
        help="Column name containing paired negative captions for NSYNC preprocessing",
    ),
    negative_media_column: str = typer.Option(
        default="negative_media_path",
        help="Optional column name containing paired user-supplied negative media paths for NSYNC preprocessing",
    ),
    negative_inference_steps: int = typer.Option(
        default=30,
        help="Number of denoising steps used when auto-generating missing NSYNC negative media",
    ),
    negative_guidance_scale: float = typer.Option(
        default=4.0,
        help="CFG guidance scale used when auto-generating missing NSYNC negative media",
    ),
    negative_prompt: str = typer.Option(
        default=DEFAULT_NEGATIVE_PROMPT,
        help="Artifact-suppression prompt used as the CFG negative prompt during missing-negative generation",
    ),
    negative_seed: int = typer.Option(
        default=42,
        help="Seed used when auto-generating missing NSYNC negative media",
    ),
    save_generated_negatives: bool = typer.Option(
        default=False,
        help="Decode and save preview videos for auto-generated NSYNC negatives",
    ),
    reference_downscale_factor: int = typer.Option(
        default=1,
        help="Downscale factor for reference video resolution. When > 1, reference videos are processed at "
        "1/n resolution (e.g., 2 means half resolution). Used for efficient IC-LoRA training.",
    ),
) -> None:
    """Preprocess a video dataset by computing and saving latents and text embeddings.
    The dataset must be a CSV, JSON, or JSONL file with columns for captions and video paths.
    This script is designed for LTX-2 models which use the Gemma text encoder.
    Examples:
        # Process a dataset with LTX-2 model
        python scripts/process_dataset.py dataset.json --resolution-buckets 768x768x25 \\
            --model-path /path/to/ltx2.safetensors --text-encoder-path /path/to/gemma
        # Process dataset with custom column names
        python scripts/process_dataset.py dataset.json --resolution-buckets 768x768x25 \\
            --model-path /path/to/ltx2.safetensors --text-encoder-path /path/to/gemma \\
            --caption-column "text" --video-column "video_path"
        # Process dataset with reference videos for IC-LoRA training
        python scripts/process_dataset.py dataset.json --resolution-buckets 768x768x25 \\
            --model-path /path/to/ltx2.safetensors --text-encoder-path /path/to/gemma \\
            --reference-column "reference_path"
        # Process dataset with scaled reference videos (half resolution) for efficient IC-LoRA
        python scripts/process_dataset.py dataset.json --resolution-buckets 768x768x25 \\
            --model-path /path/to/ltx2.safetensors --text-encoder-path /path/to/gemma \\
            --reference-column "reference_path" --reference-downscale-factor 2
        # Process dataset with audio for audio-video training
        python scripts/process_dataset.py dataset.json --resolution-buckets 768x512x97 \\
            --model-path /path/to/ltx2.safetensors --text-encoder-path /path/to/gemma \\
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
        load_text_encoder_in_8bit=load_text_encoder_in_8bit,
        negative_caption_column=negative_caption_column,
        negative_media_column=negative_media_column,
        negative_inference_steps=negative_inference_steps,
        negative_guidance_scale=negative_guidance_scale,
        negative_prompt=negative_prompt,
        negative_seed=negative_seed,
        save_generated_negatives=save_generated_negatives,
    )


if __name__ == "__main__":
    app()
