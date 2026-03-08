from __future__ import annotations

from collections.abc import Iterable, Sequence

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


def weighted_tensor_projection_average(
    source: Tensor,
    targets: Sequence[Tensor],
    *,
    weights: Sequence[float] | Tensor | None = None,
    eps: float = 1e-12,
) -> Tensor:
    """Return a weighted average of projections of ``source`` onto each target."""
    if not targets:
        return torch.zeros_like(source)

    normalized_weights = _normalize_projection_weights(targets, weights, device=source.device, dtype=source.dtype)
    projection_terms = [
        weight * tensor_projection(source, target, eps=eps)
        for weight, target in zip(normalized_weights, targets, strict=True)
    ]
    return torch.stack(projection_terms).sum(dim=0)


def weighted_reverse_tensor_projection_average(
    sources: Sequence[Tensor],
    target: Tensor,
    *,
    weights: Sequence[float] | Tensor | None = None,
    eps: float = 1e-12,
) -> Tensor:
    """Return a weighted average of projections of each source onto ``target``."""
    if not sources:
        return torch.zeros_like(target)

    normalized_weights = _normalize_projection_weights(sources, weights, device=target.device, dtype=target.dtype)
    projection_terms = [
        weight * tensor_projection(source, target, eps=eps)
        for weight, source in zip(normalized_weights, sources, strict=True)
    ]
    return torch.stack(projection_terms).sum(dim=0)


def combine_advanced_tensor_gradients(
    positive: Tensor,
    negatives: Sequence[Tensor],
    *,
    anchors: Sequence[Tensor] | None = None,
    negative_weights: Sequence[float] | Tensor | None = None,
    anchor_weights: Sequence[float] | Tensor | None = None,
    eps: float = 1e-12,
) -> Tensor:
    """Apply the advanced NSYNC update rule to standalone tensors."""
    anchor_list = list(anchors or [])
    negative_projection = weighted_tensor_projection_average(
        positive,
        negatives,
        weights=negative_weights,
        eps=eps,
    )
    positive_anchor_projection = weighted_tensor_projection_average(
        positive,
        anchor_list,
        weights=anchor_weights,
        eps=eps,
    )
    anchor_positive_projection = weighted_reverse_tensor_projection_average(
        anchor_list,
        positive,
        weights=anchor_weights,
        eps=eps,
    )
    agree_projection = 0.5 * (positive_anchor_projection + anchor_positive_projection)
    return positive - negative_projection + positive_anchor_projection + agree_projection


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


def _normalize_projection_weights(
    items: Sequence[Tensor],
    weights: Sequence[float] | Tensor | None,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    if weights is None:
        normalized = torch.ones(len(items), device=device, dtype=dtype)
    else:
        normalized = torch.as_tensor(weights, device=device, dtype=dtype)
        if normalized.shape != (len(items),):
            raise ValueError("Projection weights must match the number of tensors")

    return normalized / normalized.sum().clamp_min(torch.finfo(dtype).eps)
