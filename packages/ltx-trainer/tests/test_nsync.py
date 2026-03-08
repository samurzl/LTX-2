import torch

from ltx_trainer.nsync import (
    combine_advanced_tensor_gradients,
    combine_tensor_gradients,
    extend_data_sources_for_nsync,
    negative_branch_source_keys,
    tensor_projection,
)


def test_tensor_projection_matches_manual_solution() -> None:
    source = torch.tensor([3.0, 4.0])
    target = torch.tensor([1.0, 2.0])

    projection = tensor_projection(source, target)

    expected = (11.0 / 5.0) * target
    assert torch.allclose(projection, expected)


def test_combine_tensor_gradients_uses_positive_negative_and_anchor_terms() -> None:
    positive = torch.tensor([2.0, 1.0])
    negative = torch.tensor([1.0, 0.0])
    anchor = torch.tensor([0.0, 2.0])

    combined = combine_tensor_gradients(positive, negative, anchor=anchor)

    expected = torch.tensor([0.0, 2.0])
    assert torch.allclose(combined, expected)


def test_tensor_projection_zero_norm_is_stable() -> None:
    source = torch.tensor([1.0, 2.0])
    target = torch.zeros(2)

    projection = tensor_projection(source, target, eps=1e-12)

    assert torch.equal(projection, torch.zeros_like(target))


def test_extend_data_sources_for_nsync_adds_negative_audio_sources_when_requested() -> None:
    extended = extend_data_sources_for_nsync(
        {"latents": "latents", "conditions": "conditions"},
        enabled=True,
        with_audio=True,
        negative_latents_dir="negative_latents",
        negative_conditions_dir="negative_conditions",
        negative_audio_latents_dir="negative_audio_latents",
    )

    assert extended == {
        "latents": "latents",
        "conditions": "conditions",
        "negative_latents": "negative_latents",
        "negative_conditions": "negative_conditions",
        "negative_audio_latents": "negative_audio_latents",
    }


def test_negative_branch_source_keys_reuse_positive_reference_latents() -> None:
    source_keys = negative_branch_source_keys()

    assert source_keys.latents == "negative_latents"
    assert source_keys.conditions == "negative_conditions"
    assert source_keys.audio_latents == "negative_audio_latents"
    assert source_keys.ref_latents == "ref_latents"


def test_combine_advanced_tensor_gradients_applies_negative_anchor_and_agreement_terms() -> None:
    positive = torch.tensor([2.0, 2.0])
    negatives = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 2.0])]
    anchors = [torch.tensor([2.0, 0.0]), torch.tensor([0.0, 2.0])]

    combined = combine_advanced_tensor_gradients(
        positive,
        negatives,
        anchors=anchors,
    )

    expected = torch.tensor([3.0, 3.0])
    assert torch.allclose(combined, expected)
