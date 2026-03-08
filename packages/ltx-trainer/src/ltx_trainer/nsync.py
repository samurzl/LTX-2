from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor

from ltx_trainer.training_strategies.base_strategy import BatchSourceKeys


def extend_data_sources_for_nsync(
    base_sources: dict[str, str],
    *,
    enabled: bool,
    with_audio: bool,
    negative_latents_dir: str,
    negative_conditions_dir: str,
    negative_audio_latents_dir: str,
) -> dict[str, str]:
    """Add NSYNC-specific paired sources to the dataset source mapping."""
    sources = base_sources.copy()
    if not enabled:
        return sources

    sources[negative_latents_dir] = "negative_latents"
    sources[negative_conditions_dir] = "negative_conditions"
    if with_audio:
        sources[negative_audio_latents_dir] = "negative_audio_latents"
    return sources


def positive_branch_source_keys() -> BatchSourceKeys:
    """Canonical source keys for the standard positive branch."""
    return BatchSourceKeys()


def negative_branch_source_keys() -> BatchSourceKeys:
    """Canonical source keys for the paired negative branch.

    The negative branch always reuses the positive reference latents for IC-LoRA.
    """
    return BatchSourceKeys(
        latents="negative_latents",
        conditions="negative_conditions",
        audio_latents="negative_audio_latents",
        ref_latents="ref_latents",
    )


def tensor_projection(source: Tensor, target: Tensor, eps: float = 1e-12) -> Tensor:
    """Project ``source`` onto ``target`` with zero-norm protection."""
    dot_product = torch.dot(source.reshape(-1), target.reshape(-1))
    target_norm_sq = torch.dot(target.reshape(-1), target.reshape(-1)).clamp_min(eps)
    return target * (dot_product / target_norm_sq)


def combine_tensor_gradients(
    positive: Tensor,
    negative: Tensor,
    *,
    anchor: Tensor | None = None,
    eps: float = 1e-12,
) -> Tensor:
    """Apply the NSYNC update rule to standalone tensors."""
    updated = positive - tensor_projection(positive, negative, eps=eps)
    if anchor is not None:
        updated = updated + tensor_projection(positive, anchor, eps=eps)
    return updated


def gradient_statistics(
    source_grads: Iterable[Tensor | None],
    target_grads: Iterable[Tensor | None],
    *,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Compute local dot-product and squared norm for gradient projection."""
    dot_product = torch.zeros((), device=device, dtype=torch.float32)
    target_norm_sq = torch.zeros((), device=device, dtype=torch.float32)

    for source_grad, target_grad in zip(source_grads, target_grads, strict=True):
        if source_grad is None or target_grad is None:
            continue
        dot_product = dot_product + torch.sum(source_grad.float() * target_grad.float())
        target_norm_sq = target_norm_sq + torch.sum(target_grad.float().pow(2))

    return dot_product, target_norm_sq
