# ruff: noqa: PLC0415

"""
Model loader for LTX-2 trainer using the new ltx-core package.
This module provides a unified interface for loading LTX-2 model components
for training, using SingleGPUModelBuilder from ltx-core.
Example usage:
    # Load individual components
    vae_encoder = load_video_vae_encoder("/path/to/checkpoint.safetensors", device="cuda")
    vae_decoder = load_video_vae_decoder("/path/to/checkpoint.safetensors", device="cuda")
    text_encoder = load_text_encoder("/path/to/gemma.safetensors", device="cuda")
    # Load all components at once
    components = load_model("/path/to/checkpoint.safetensors", text_encoder_path="/path/to/gemma.safetensors")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from ltx_trainer import logger

# Type alias for device specification
Device = str | torch.device

# Type checking imports (not loaded at runtime)
if TYPE_CHECKING:
    from ltx_core.components.schedulers import LTX2Scheduler
    from ltx_core.model.audio_vae import AudioDecoder, AudioEncoder, Vocoder
    from ltx_core.model.transformer import LTXModel
    from ltx_core.model.video_vae import VideoDecoder, VideoEncoder
    from ltx_core.text_encoders.gemma import GemmaTextEncoder
    from ltx_core.text_encoders.gemma.embeddings_processor import EmbeddingsProcessor


def _to_torch_device(device: Device) -> torch.device:
    """Convert device specification to torch.device."""
    return torch.device(device) if isinstance(device, str) else device


# =============================================================================
# Individual Component Loaders
# =============================================================================


def load_transformer(
    checkpoint_path: str | Path,
    device: Device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> "LTXModel":
    """Load the LTX transformer model.
    Args:
        checkpoint_path: Path to the safetensors checkpoint file
        device: Device to load model on
        dtype: Data type for model weights
    Returns:
        Loaded LTXModel transformer
    """
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from ltx_core.model.transformer.model_configurator import (
        LTXV_MODEL_COMFY_RENAMING_MAP,
        LTXModelConfigurator,
    )

    return SingleGPUModelBuilder(
        model_path=str(checkpoint_path),
        model_class_configurator=LTXModelConfigurator,
        model_sd_ops=LTXV_MODEL_COMFY_RENAMING_MAP,
    ).build(device=_to_torch_device(device), dtype=dtype)


def load_video_vae_encoder(
    checkpoint_path: str | Path,
    device: Device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> "VideoEncoder":
    """Load the video VAE encoder (for preprocessing).
    Args:
        checkpoint_path: Path to the safetensors checkpoint file
        device: Device to load model on
        dtype: Data type for model weights
    Returns:
        Loaded VideoEncoder
    """
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from ltx_core.model.video_vae import VAE_ENCODER_COMFY_KEYS_FILTER, VideoEncoderConfigurator

    return SingleGPUModelBuilder(
        model_path=str(checkpoint_path),
        model_class_configurator=VideoEncoderConfigurator,
        model_sd_ops=VAE_ENCODER_COMFY_KEYS_FILTER,
    ).build(device=_to_torch_device(device), dtype=dtype)


def load_video_vae_decoder(
    checkpoint_path: str | Path,
    device: Device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> "VideoDecoder":
    """Load the video VAE decoder (for inference/validation).
    Args:
        checkpoint_path: Path to the safetensors checkpoint file
        device: Device to load model on
        dtype: Data type for model weights
    Returns:
        Loaded VideoDecoder
    """
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from ltx_core.model.video_vae import VAE_DECODER_COMFY_KEYS_FILTER, VideoDecoderConfigurator

    return SingleGPUModelBuilder(
        model_path=str(checkpoint_path),
        model_class_configurator=VideoDecoderConfigurator,
        model_sd_ops=VAE_DECODER_COMFY_KEYS_FILTER,
    ).build(device=_to_torch_device(device), dtype=dtype)


def load_audio_vae_encoder(
    checkpoint_path: str | Path,
    device: Device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> "AudioEncoder":
    """Load the audio VAE encoder (for preprocessing).
    Args:
        checkpoint_path: Path to the safetensors checkpoint file
        device: Device to load model on
        dtype: Data type for model weights (default bfloat16, but float32 recommended for quality)
    Returns:
        Loaded AudioEncoder
    """
    from ltx_core.loader import SingleGPUModelBuilder
    from ltx_core.model.audio_vae import AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER, AudioEncoderConfigurator

    return SingleGPUModelBuilder(
        model_path=str(checkpoint_path),
        model_class_configurator=AudioEncoderConfigurator,
        model_sd_ops=AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER,
    ).build(device=_to_torch_device(device), dtype=dtype)


def load_audio_vae_decoder(
    checkpoint_path: str | Path,
    device: Device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> "AudioDecoder":
    """Load the audio VAE decoder.
    Args:
        checkpoint_path: Path to the safetensors checkpoint file
        device: Device to load model on
        dtype: Data type for model weights
    Returns:
        Loaded AudioDecoder
    """
    from ltx_core.loader import SingleGPUModelBuilder
    from ltx_core.model.audio_vae import AUDIO_VAE_DECODER_COMFY_KEYS_FILTER, AudioDecoderConfigurator

    return SingleGPUModelBuilder(
        model_path=str(checkpoint_path),
        model_class_configurator=AudioDecoderConfigurator,
        model_sd_ops=AUDIO_VAE_DECODER_COMFY_KEYS_FILTER,
    ).build(device=_to_torch_device(device), dtype=dtype)


def load_vocoder(
    checkpoint_path: str | Path,
    device: Device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> "Vocoder":
    """Load the vocoder (for audio waveform generation).
    Args:
        checkpoint_path: Path to the safetensors checkpoint file
        device: Device to load model on
        dtype: Data type for model weights
    Returns:
        Loaded Vocoder
    """
    from ltx_core.loader import SingleGPUModelBuilder
    from ltx_core.model.audio_vae import VOCODER_COMFY_KEYS_FILTER, VocoderConfigurator

    return SingleGPUModelBuilder(
        model_path=str(checkpoint_path),
        model_class_configurator=VocoderConfigurator,
        model_sd_ops=VOCODER_COMFY_KEYS_FILTER,
    ).build(device=_to_torch_device(device), dtype=dtype)


def load_text_encoder(
    gemma_model_path: str | Path,
    device: Device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    load_in_8bit: bool = False,
) -> "GemmaTextEncoder":
    """Load the Gemma text encoder.
    Args:
        gemma_model_path: Path to a single Gemma .safetensors file or a Gemma model directory
        device: Device to load model on
        dtype: Data type for model weights
        load_in_8bit: Whether to load the Gemma model in 8-bit precision using bitsandbytes.
    Returns:
        Loaded GemmaTextEncoder
    """
    # Use 8-bit loading path if requested
    if load_in_8bit:
        from ltx_trainer.gemma_8bit import load_8bit_gemma

        return load_8bit_gemma(gemma_model_path, dtype, device=device)

    # Standard loading path
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from ltx_core.text_encoders.gemma import (
        GEMMA_LLM_KEY_OPS,
        GEMMA_MODEL_OPS,
        GemmaTextEncoderConfigurator,
        gemma_weight_paths_from_source,
        module_ops_from_gemma_source,
    )

    torch_device = _to_torch_device(device)
    gemma_weight_paths = gemma_weight_paths_from_source(gemma_model_path)

    text_encoder = SingleGPUModelBuilder(
        model_path=tuple(gemma_weight_paths),
        model_class_configurator=GemmaTextEncoderConfigurator,
        model_sd_ops=GEMMA_LLM_KEY_OPS,
        module_ops=(GEMMA_MODEL_OPS, *module_ops_from_gemma_source(gemma_model_path, include_processor=False)),
    ).build(device=torch_device, dtype=dtype)

    _move_gemma_language_runtime_to_device(text_encoder, torch_device)

    return text_encoder


def _move_gemma_language_runtime_to_device(text_encoder: "GemmaTextEncoder", device: torch.device) -> None:
    language_model = text_encoder.model.model.language_model
    uninitialized = _meta_tensor_names(language_model)
    if uninitialized:
        preview = ", ".join(uninitialized[:8])
        suffix = "" if len(uninitialized) <= 8 else f", ... ({len(uninitialized)} total)"
        raise RuntimeError(
            "Gemma language weights were not fully loaded from text_encoder_path. "
            f"First uninitialized language tensors: {preview}{suffix}. "
            "Check that the Gemma safetensors file is an LTX/Comfy Gemma3-12B text encoder "
            "or a Hugging Face Gemma3-12B checkpoint."
        )

    language_model.to(device)

    lm_head = getattr(text_encoder.model, "lm_head", None)
    if lm_head is not None and not _meta_tensor_names(lm_head):
        lm_head.to(device)


def _meta_tensor_names(module: torch.nn.Module) -> list[str]:
    names = []
    for name, param in module.named_parameters():
        if param.device.type == "meta":
            names.append(name)
    for name, buf in module.named_buffers():
        if buf.device.type == "meta":
            names.append(name)
    return names


def load_embeddings_processor(
    checkpoint_path: str | Path,
    device: Device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> "EmbeddingsProcessor":
    """Load the embeddings processor (feature extractor + video/audio connectors).
    Args:
        checkpoint_path: Path to the LTX-2 safetensors checkpoint file
        device: Device to load model on
        dtype: Data type for model weights
    Returns:
        Loaded EmbeddingsProcessor with feature extractor and connectors
    """
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from ltx_core.text_encoders.gemma import (
        EMBEDDINGS_PROCESSOR_KEY_OPS,
        EmbeddingsProcessorConfigurator,
    )

    torch_device = _to_torch_device(device)

    return SingleGPUModelBuilder(
        model_path=str(checkpoint_path),
        model_class_configurator=EmbeddingsProcessorConfigurator,
        model_sd_ops=EMBEDDINGS_PROCESSOR_KEY_OPS,
    ).build(device=torch_device, dtype=dtype)


# =============================================================================
# Combined Component Loader
# =============================================================================


@dataclass
class LtxModelComponents:
    """Container for all LTX-2 model components."""

    transformer: "LTXModel"
    video_vae_encoder: "VideoEncoder | None" = None
    video_vae_decoder: "VideoDecoder | None" = None
    audio_vae_decoder: "AudioDecoder | None" = None
    vocoder: "Vocoder | None" = None
    text_encoder: "GemmaTextEncoder | None" = None
    scheduler: "LTX2Scheduler | None" = None


def load_model(  # noqa: PLR0913
    checkpoint_path: str | Path | None,
    text_encoder_path: str | Path | None = None,
    device: Device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    with_video_vae_encoder: bool = False,
    with_video_vae_decoder: bool = True,
    with_audio_vae_decoder: bool = True,
    with_vocoder: bool = True,
    with_text_encoder: bool = True,
    transformer_path: str | Path | None = None,
    video_vae_encoder_path: str | Path | None = None,
    video_vae_decoder_path: str | Path | None = None,
    audio_vae_decoder_path: str | Path | None = None,
    vocoder_path: str | Path | None = None,
) -> LtxModelComponents:
    """
    Load LTX-2 model components from a safetensors checkpoint.
    This is a convenience function that loads multiple components at once.
    For loading individual components, use the dedicated functions:
    - load_transformer()
    - load_video_vae_encoder()
    - load_video_vae_decoder()
    - load_audio_vae_decoder()
    - load_vocoder()
    - load_text_encoder()
    Args:
        checkpoint_path: Path to the monolithic safetensors checkpoint file. Optional when all requested component
            paths are supplied separately.
        text_encoder_path: Path to a single Gemma .safetensors file or directory (required if with_text_encoder=True)
        device: Device to load models on ("cuda", "cpu", etc.)
        dtype: Data type for model weights
        with_video_vae_encoder: Whether to load the video VAE encoder (for preprocessing)
        with_video_vae_decoder: Whether to load the video VAE decoder (for inference/validation)
        with_audio_vae_decoder: Whether to load the audio VAE decoder
        with_vocoder: Whether to load the vocoder
        with_text_encoder: Whether to load the text encoder
        transformer_path: Optional checkpoint override for transformer weights
        video_vae_encoder_path: Optional checkpoint override for video VAE encoder weights
        video_vae_decoder_path: Optional checkpoint override for video VAE decoder weights
        audio_vae_decoder_path: Optional checkpoint override for audio VAE decoder weights
        vocoder_path: Optional checkpoint override for vocoder weights
    Returns:
        LtxModelComponents containing all loaded model components
    """
    from ltx_core.components.schedulers import LTX2Scheduler

    checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None

    # Validate checkpoint exists when used as a component fallback.
    if checkpoint_path is not None:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        logger.info(f"Loading LTX-2 model from {checkpoint_path}")
    else:
        logger.info("Loading LTX-2 model from per-component checkpoint paths")

    torch_device = _to_torch_device(device)

    def _component_path(component_name: str, override_path: str | Path | None) -> str | Path:
        if override_path is not None:
            return override_path
        if checkpoint_path is not None:
            return checkpoint_path
        raise ValueError(f"{component_name} path must be provided when checkpoint_path is not set")

    # Load transformer
    logger.debug("Loading transformer...")
    transformer = load_transformer(_component_path("transformer", transformer_path), torch_device, dtype)

    # Load video VAE encoder
    video_vae_encoder = None
    if with_video_vae_encoder:
        logger.debug("Loading video VAE encoder...")
        video_vae_encoder = load_video_vae_encoder(
            _component_path("video_vae_encoder", video_vae_encoder_path), torch_device, dtype
        )

    # Load video VAE decoder
    video_vae_decoder = None
    if with_video_vae_decoder:
        logger.debug("Loading video VAE decoder...")
        video_vae_decoder = load_video_vae_decoder(
            _component_path("video_vae_decoder", video_vae_decoder_path), torch_device, dtype
        )

    # Load audio VAE decoder
    audio_vae_decoder = None
    if with_audio_vae_decoder:
        logger.debug("Loading audio VAE decoder...")
        audio_vae_decoder = load_audio_vae_decoder(
            _component_path("audio_vae_decoder", audio_vae_decoder_path), torch_device, dtype
        )

    # Load vocoder
    vocoder = None
    if with_vocoder:
        logger.debug("Loading vocoder...")
        vocoder = load_vocoder(_component_path("vocoder", vocoder_path), torch_device, dtype)

    # Load text encoder
    text_encoder = None
    if with_text_encoder:
        if text_encoder_path is None:
            raise ValueError("text_encoder_path must be provided when with_text_encoder=True")
        logger.debug("Loading Gemma text encoder...")
        text_encoder = load_text_encoder(text_encoder_path, torch_device, dtype)

    # Create scheduler (stateless, no loading needed)
    scheduler = LTX2Scheduler()

    return LtxModelComponents(
        transformer=transformer,
        video_vae_encoder=video_vae_encoder,
        video_vae_decoder=video_vae_decoder,
        audio_vae_decoder=audio_vae_decoder,
        vocoder=vocoder,
        text_encoder=text_encoder,
        scheduler=scheduler,
    )
