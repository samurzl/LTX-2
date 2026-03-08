from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd

NSYNC_MANIFEST_FILENAME = "nsync_manifest.json"
NSYNC_NEGATIVE_FILE_PREFIX = "__nsync_negative_"


@dataclass(frozen=True)
class AdvancedNsyncNegativeSpec:
    media: Literal["positive", "synthetic"]
    caption: str
    prompt: str | None = None


@dataclass(frozen=True)
class AdvancedNsyncAnchorRuleSpec:
    required_categories: list[str]
    extra_random_category: bool = False


@dataclass(frozen=True)
class AdvancedNsyncSampleSpec:
    media_path: str
    sample_rel_path: str
    categories: list[str]
    negatives: list[AdvancedNsyncNegativeSpec]
    anchors: list[AdvancedNsyncAnchorRuleSpec]


@dataclass(frozen=True)
class NsyncManifestNegative:
    media: Literal["positive", "synthetic"]
    condition_rel_path: str
    latent_rel_path: str | None
    audio_latent_rel_path: str | None


@dataclass(frozen=True)
class NsyncManifestAnchorRule:
    required_categories: list[str]
    extra_random_category: bool
    extra_category_candidates: list[str]


@dataclass(frozen=True)
class NsyncManifestSample:
    media_path: str
    sample_rel_path: str
    categories: list[str]
    negatives: list[NsyncManifestNegative]
    anchors: list[NsyncManifestAnchorRule]


@dataclass(frozen=True)
class NsyncManifest:
    version: int
    categories: list[str]
    samples: list[NsyncManifestSample]


def resolve_nsync_manifest_path(data_root: str | Path) -> Path:
    root = Path(data_root).expanduser().resolve()
    if (root / ".precomputed").exists():
        root = root / ".precomputed"
    return root / NSYNC_MANIFEST_FILENAME


def build_nsync_negative_output_rel_path(sample_rel_path: str | Path, negative_index: int) -> Path:
    rel_path = Path(sample_rel_path)
    return rel_path.with_name(f"{rel_path.stem}{NSYNC_NEGATIVE_FILE_PREFIX}{negative_index:03d}.pt")


def pt_path_to_virtual_media_path(relative_path: str | Path) -> Path:
    return Path(relative_path).with_suffix(".mp4")


def load_advanced_nsync_specs(
    dataset_file: str | Path,
    *,
    media_column: str,
    legacy_negative_caption_column: str,
    legacy_negative_media_column: str,
) -> list[AdvancedNsyncSampleSpec] | None:
    dataset_path = Path(dataset_file)
    records = _load_dataset_records(dataset_path)
    if not records:
        return None

    has_structured_nsync = any("nsync" in record for record in records)
    if not has_structured_nsync:
        return None

    if dataset_path.suffix.lower() == ".csv":
        raise ValueError("Advanced NSYNC metadata requires JSON or JSONL input. CSV remains legacy-only.")

    has_legacy_columns = any(
        legacy_negative_caption_column in record or legacy_negative_media_column in record for record in records
    )
    if has_legacy_columns:
        raise ValueError(
            "Advanced NSYNC metadata cannot be mixed with legacy negative caption/media columns in the same dataset."
        )

    specs: list[AdvancedNsyncSampleSpec] = []
    for row_index, record in enumerate(records):
        if "nsync" not in record:
            raise ValueError(f"Missing 'nsync' object for dataset row {row_index}")
        if media_column not in record:
            raise ValueError(f"Key '{media_column}' not found in dataset row {row_index}")

        nsync = record["nsync"]
        if not isinstance(nsync, dict):
            raise ValueError(f"'nsync' must be an object in dataset row {row_index}")

        media_path = _require_non_empty_string(record[media_column], f"{media_column} (row {row_index})")
        categories = _normalize_string_list(nsync.get("categories"), field_name=f"nsync.categories (row {row_index})")
        negatives = _parse_negative_specs(nsync.get("negatives"), row_index=row_index)
        anchors = _parse_anchor_specs(nsync.get("anchors"), row_index=row_index)

        specs.append(
            AdvancedNsyncSampleSpec(
                media_path=media_path,
                sample_rel_path=str(Path(media_path).with_suffix(".pt")),
                categories=categories,
                negatives=negatives,
                anchors=anchors,
            )
        )

    return specs


def filter_advanced_nsync_specs(
    specs: list[AdvancedNsyncSampleSpec],
    *,
    available_sample_rel_paths: set[str],
) -> list[AdvancedNsyncSampleSpec]:
    return [spec for spec in specs if spec.sample_rel_path in available_sample_rel_paths]


def build_nsync_manifest(
    specs: list[AdvancedNsyncSampleSpec],
    *,
    with_audio: bool,
) -> NsyncManifest:
    if not specs:
        raise ValueError("Advanced NSYNC manifest requires at least one valid sample")

    sample_rel_paths = [spec.sample_rel_path for spec in specs]
    if len(sample_rel_paths) != len(set(sample_rel_paths)):
        raise ValueError("Advanced NSYNC dataset contains duplicate media paths after preprocessing")

    categories = sorted({category for spec in specs for category in spec.categories})
    category_to_indices = {category: set[int]() for category in categories}
    for index, spec in enumerate(specs):
        for category in spec.categories:
            category_to_indices[category].add(index)

    manifest_samples: list[NsyncManifestSample] = []
    for sample_index, spec in enumerate(specs):
        manifest_negatives = [
            NsyncManifestNegative(
                media=negative.media,
                condition_rel_path=str(build_nsync_negative_output_rel_path(spec.sample_rel_path, negative_index)),
                latent_rel_path=(
                    None
                    if negative.media == "positive"
                    else str(build_nsync_negative_output_rel_path(spec.sample_rel_path, negative_index))
                ),
                audio_latent_rel_path=(
                    None
                    if negative.media == "positive" or not with_audio
                    else str(build_nsync_negative_output_rel_path(spec.sample_rel_path, negative_index))
                ),
            )
            for negative_index, negative in enumerate(spec.negatives)
        ]

        manifest_anchors: list[NsyncManifestAnchorRule] = []
        for rule_index, anchor in enumerate(spec.anchors):
            missing_categories = [category for category in anchor.required_categories if category not in category_to_indices]
            if missing_categories:
                raise ValueError(
                    f"Sample '{spec.media_path}' anchor rule {rule_index} references unknown categories: "
                    f"{', '.join(missing_categories)}"
                )

            base_candidates = _candidate_indices_for_required_categories(
                category_to_indices,
                anchor.required_categories,
            )
            base_candidates.discard(sample_index)

            if anchor.extra_random_category:
                extra_categories = [
                    category
                    for category in categories
                    if category not in anchor.required_categories
                    and _candidate_indices_for_required_categories(
                        category_to_indices,
                        [*anchor.required_categories, category],
                    )
                    - {sample_index}
                ]
                if not extra_categories:
                    raise ValueError(
                        f"Sample '{spec.media_path}' anchor rule {rule_index} has no feasible extra random categories"
                    )
            else:
                extra_categories = []
                if not base_candidates:
                    raise ValueError(
                        f"Sample '{spec.media_path}' anchor rule {rule_index} has no eligible non-self anchor candidates"
                    )

            manifest_anchors.append(
                NsyncManifestAnchorRule(
                    required_categories=anchor.required_categories,
                    extra_random_category=anchor.extra_random_category,
                    extra_category_candidates=extra_categories,
                )
            )

        manifest_samples.append(
            NsyncManifestSample(
                media_path=spec.media_path,
                sample_rel_path=spec.sample_rel_path,
                categories=spec.categories,
                negatives=manifest_negatives,
                anchors=manifest_anchors,
            )
        )

    return NsyncManifest(
        version=1,
        categories=categories,
        samples=manifest_samples,
    )


def write_nsync_manifest(manifest: NsyncManifest, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_nsync_manifest(manifest_path: str | Path) -> NsyncManifest:
    data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    samples = [
        NsyncManifestSample(
            media_path=sample["media_path"],
            sample_rel_path=sample["sample_rel_path"],
            categories=list(sample["categories"]),
            negatives=[
                NsyncManifestNegative(
                    media=negative["media"],
                    condition_rel_path=negative["condition_rel_path"],
                    latent_rel_path=negative["latent_rel_path"],
                    audio_latent_rel_path=negative["audio_latent_rel_path"],
                )
                for negative in sample["negatives"]
            ],
            anchors=[
                NsyncManifestAnchorRule(
                    required_categories=list(anchor["required_categories"]),
                    extra_random_category=bool(anchor["extra_random_category"]),
                    extra_category_candidates=list(anchor.get("extra_category_candidates", [])),
                )
                for anchor in sample["anchors"]
            ],
        )
        for sample in data["samples"]
    ]
    return NsyncManifest(
        version=int(data["version"]),
        categories=list(data["categories"]),
        samples=samples,
    )


def _parse_negative_specs(value: Any, *, row_index: int) -> list[AdvancedNsyncNegativeSpec]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"nsync.negatives must be a non-empty list in dataset row {row_index}")

    negatives: list[AdvancedNsyncNegativeSpec] = []
    for negative_index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"nsync.negatives[{negative_index}] must be an object in dataset row {row_index}")

        media = item.get("media")
        if media not in {"positive", "synthetic"}:
            raise ValueError(
                f"nsync.negatives[{negative_index}].media must be 'positive' or 'synthetic' in dataset row {row_index}"
            )

        caption = _require_non_empty_string(
            item.get("caption"),
            f"nsync.negatives[{negative_index}].caption (row {row_index})",
        )
        prompt = item.get("prompt")

        if media == "synthetic":
            prompt = _require_non_empty_string(
                prompt,
                f"nsync.negatives[{negative_index}].prompt (row {row_index})",
            )
        elif prompt is not None:
            raise ValueError(
                f"nsync.negatives[{negative_index}].prompt is only allowed when media='synthetic' in dataset row "
                f"{row_index}"
            )

        negatives.append(
            AdvancedNsyncNegativeSpec(
                media=media,
                caption=caption,
                prompt=prompt,
            )
        )

    return negatives


def _parse_anchor_specs(value: Any, *, row_index: int) -> list[AdvancedNsyncAnchorRuleSpec]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"nsync.anchors must be a list in dataset row {row_index}")

    anchors: list[AdvancedNsyncAnchorRuleSpec] = []
    for anchor_index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"nsync.anchors[{anchor_index}] must be an object in dataset row {row_index}")

        anchors.append(
            AdvancedNsyncAnchorRuleSpec(
                required_categories=_normalize_string_list(
                    item.get("required_categories"),
                    field_name=f"nsync.anchors[{anchor_index}].required_categories (row {row_index})",
                ),
                extra_random_category=bool(item.get("extra_random_category", False)),
            )
        )

    return anchors


def _normalize_string_list(value: Any, *, field_name: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field_name} must be a non-empty list of strings")

    normalized: list[str] = []
    for item in value:
        normalized.append(_require_non_empty_string(item, field_name))

    unique: list[str] = []
    seen: set[str] = set()
    for item in normalized:
        if item not in seen:
            unique.append(item)
            seen.add(item)
    return unique


def _require_non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _candidate_indices_for_required_categories(
    category_to_indices: dict[str, set[int]],
    required_categories: list[str],
) -> set[int]:
    required_sets = [category_to_indices[category] for category in required_categories]
    return set.intersection(*required_sets) if required_sets else set()


def _load_dataset_records(dataset_file: Path) -> list[dict[str, Any]]:
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
