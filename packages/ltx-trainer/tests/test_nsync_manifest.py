import json
from pathlib import Path

import pytest
import torch

from ltx_trainer.datasets import ManifestNsyncDataset, PrecomputedDataset, collate_manifest_nsync_batch
from ltx_trainer.nsync_manifest import (
    NSYNC_MANIFEST_FILENAME,
    build_nsync_manifest,
    load_advanced_nsync_specs,
    write_nsync_manifest,
)


def _write_dataset(tmp_path: Path, rows: list[dict]) -> Path:
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps(rows), encoding="utf-8")
    return dataset_path


def _write_positive_sample(root: Path, rel_path: str, *, value: float) -> None:
    latent_path = root / "latents" / rel_path
    latent_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "latents": torch.full((1, 1, 1, 1), value),
            "num_frames": 1,
            "height": 1,
            "width": 1,
            "fps": 24.0,
        },
        latent_path,
    )

    condition_path = root / "conditions" / rel_path
    condition_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "video_prompt_embeds": torch.full((1, 2), value),
            "prompt_attention_mask": torch.ones(1, dtype=torch.bool),
        },
        condition_path,
    )


def _write_negative_condition(root: Path, rel_path: str, *, value: float) -> None:
    condition_path = root / "negative_conditions" / rel_path
    condition_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "video_prompt_embeds": torch.full((1, 2), value),
            "prompt_attention_mask": torch.ones(1, dtype=torch.bool),
        },
        condition_path,
    )


def _write_negative_latent(root: Path, rel_path: str, *, value: float) -> None:
    latent_path = root / "negative_latents" / rel_path
    latent_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "latents": torch.full((1, 1, 1, 1), value),
            "num_frames": 1,
            "height": 1,
            "width": 1,
            "fps": 24.0,
        },
        latent_path,
    )


def test_load_advanced_nsync_specs_rejects_mixed_legacy_columns(tmp_path: Path) -> None:
    dataset_path = _write_dataset(
        tmp_path,
        [
            {
                "caption": "positive",
                "media_path": "videos/one.mp4",
                "negative_caption": "legacy",
                "nsync": {
                    "categories": ["cat"],
                    "negatives": [{"media": "positive", "caption": "neg"}],
                },
            }
        ],
    )

    with pytest.raises(ValueError, match="cannot be mixed"):
        load_advanced_nsync_specs(
            dataset_path,
            media_column="media_path",
            legacy_negative_caption_column="negative_caption",
            legacy_negative_media_column="negative_media_path",
        )


def test_load_advanced_nsync_specs_requires_non_empty_categories(tmp_path: Path) -> None:
    dataset_path = _write_dataset(
        tmp_path,
        [
            {
                "caption": "positive",
                "media_path": "videos/one.mp4",
                "nsync": {
                    "categories": [],
                    "negatives": [{"media": "positive", "caption": "neg"}],
                },
            }
        ],
    )

    with pytest.raises(ValueError, match="categories"):
        load_advanced_nsync_specs(
            dataset_path,
            media_column="media_path",
            legacy_negative_caption_column="negative_caption",
            legacy_negative_media_column="negative_media_path",
        )


def test_load_advanced_nsync_specs_requires_valid_negative_prompt_rules(tmp_path: Path) -> None:
    dataset_path = _write_dataset(
        tmp_path,
        [
            {
                "caption": "positive",
                "media_path": "videos/one.mp4",
                "nsync": {
                    "categories": ["cat"],
                    "negatives": [{"media": "synthetic", "caption": "neg"}],
                },
            }
        ],
    )

    with pytest.raises(ValueError, match="prompt"):
        load_advanced_nsync_specs(
            dataset_path,
            media_column="media_path",
            legacy_negative_caption_column="negative_caption",
            legacy_negative_media_column="negative_media_path",
        )


def test_load_advanced_nsync_specs_rejects_prompt_for_positive_negative_media(tmp_path: Path) -> None:
    dataset_path = _write_dataset(
        tmp_path,
        [
            {
                "caption": "positive",
                "media_path": "videos/one.mp4",
                "nsync": {
                    "categories": ["cat"],
                    "negatives": [{"media": "positive", "caption": "neg", "prompt": "should not exist"}],
                },
            }
        ],
    )

    with pytest.raises(ValueError, match="only allowed when media='synthetic'"):
        load_advanced_nsync_specs(
            dataset_path,
            media_column="media_path",
            legacy_negative_caption_column="negative_caption",
            legacy_negative_media_column="negative_media_path",
        )


def test_build_nsync_manifest_tracks_feasible_extra_random_categories(tmp_path: Path) -> None:
    dataset_path = _write_dataset(
        tmp_path,
        [
            {
                "caption": "one",
                "media_path": "videos/one.mp4",
                "nsync": {
                    "categories": ["cat", "cinematic"],
                    "negatives": [{"media": "positive", "caption": "neg-one"}],
                    "anchors": [{"required_categories": ["cat"], "extra_random_category": True}],
                },
            },
            {
                "caption": "two",
                "media_path": "videos/two.mp4",
                "nsync": {
                    "categories": ["cat", "studio"],
                    "negatives": [{"media": "positive", "caption": "neg-two"}],
                },
            },
        ],
    )

    specs = load_advanced_nsync_specs(
        dataset_path,
        media_column="media_path",
        legacy_negative_caption_column="negative_caption",
        legacy_negative_media_column="negative_media_path",
    )
    manifest = build_nsync_manifest(specs, with_audio=False)

    assert manifest.categories == ["cat", "cinematic", "studio"]
    assert manifest.samples[0].anchors[0].extra_category_candidates == ["studio"]


def test_build_nsync_manifest_rejects_unknown_anchor_categories(tmp_path: Path) -> None:
    dataset_path = _write_dataset(
        tmp_path,
        [
            {
                "caption": "one",
                "media_path": "videos/one.mp4",
                "nsync": {
                    "categories": ["cat"],
                    "negatives": [{"media": "positive", "caption": "neg-one"}],
                    "anchors": [{"required_categories": ["dog"]}],
                },
            },
            {
                "caption": "two",
                "media_path": "videos/two.mp4",
                "nsync": {
                    "categories": ["cat"],
                    "negatives": [{"media": "positive", "caption": "neg-two"}],
                },
            },
        ],
    )

    specs = load_advanced_nsync_specs(
        dataset_path,
        media_column="media_path",
        legacy_negative_caption_column="negative_caption",
        legacy_negative_media_column="negative_media_path",
    )

    with pytest.raises(ValueError, match="unknown categories"):
        build_nsync_manifest(specs, with_audio=False)


def test_manifest_nsync_dataset_collates_variable_negative_and_anchor_slots(tmp_path: Path) -> None:
    precomputed_root = tmp_path / ".precomputed"
    dataset_path = _write_dataset(
        tmp_path,
        [
            {
                "caption": "one",
                "media_path": "videos/one.mp4",
                "nsync": {
                    "categories": ["cat", "cinematic"],
                    "negatives": [
                        {"media": "positive", "caption": "neg-one-a"},
                        {"media": "synthetic", "prompt": "neg-one-b prompt", "caption": "neg-one-b"},
                    ],
                    "anchors": [{"required_categories": ["cat"], "extra_random_category": True}],
                },
            },
            {
                "caption": "two",
                "media_path": "videos/two.mp4",
                "nsync": {
                    "categories": ["cat", "studio"],
                    "negatives": [{"media": "positive", "caption": "neg-two-a"}],
                    "anchors": [],
                },
            },
        ],
    )
    specs = load_advanced_nsync_specs(
        dataset_path,
        media_column="media_path",
        legacy_negative_caption_column="negative_caption",
        legacy_negative_media_column="negative_media_path",
    )
    manifest = build_nsync_manifest(specs, with_audio=False)
    write_nsync_manifest(manifest, precomputed_root / NSYNC_MANIFEST_FILENAME)

    _write_positive_sample(precomputed_root, "videos/one.pt", value=1.0)
    _write_positive_sample(precomputed_root, "videos/two.pt", value=2.0)

    _write_negative_condition(precomputed_root, manifest.samples[0].negatives[0].condition_rel_path, value=10.0)
    _write_negative_condition(precomputed_root, manifest.samples[0].negatives[1].condition_rel_path, value=11.0)
    _write_negative_condition(precomputed_root, manifest.samples[1].negatives[0].condition_rel_path, value=20.0)
    _write_negative_latent(precomputed_root, manifest.samples[0].negatives[1].latent_rel_path, value=99.0)

    base_dataset = PrecomputedDataset(str(precomputed_root), data_sources={"latents": "latents", "conditions": "conditions"})
    nsync_dataset = ManifestNsyncDataset(
        base_dataset,
        manifest_path=precomputed_root / NSYNC_MANIFEST_FILENAME,
        negative_conditions_dir="negative_conditions",
        negative_latents_dir="negative_latents",
        negative_audio_latents_dir="negative_audio_latents",
        with_audio=False,
        use_anchor=True,
    )

    first_item = nsync_dataset[0]
    second_item = nsync_dataset[1]

    assert len(first_item["negative_branches"]) == 2
    assert torch.equal(
        first_item["negative_branches"][0]["latents"]["latents"],
        first_item["positive"]["latents"]["latents"],
    )
    assert first_item["anchor_branches"][0]["idx"] == 1

    collated = collate_manifest_nsync_batch([first_item, second_item])

    assert len(collated["negative_slots"]) == 2
    assert torch.allclose(collated["negative_slot_weights"], torch.tensor([1.0, 0.5]))
    assert len(collated["anchor_slots"]) == 1
    assert torch.allclose(collated["anchor_slot_weights"], torch.tensor([0.5]))
