import random
from pathlib import Path
from typing import Any

import torch
from einops import rearrange
from torch import Tensor
from torch.utils.data import Dataset, default_collate

from ltx_trainer import logger
from ltx_trainer.nsync_manifest import (
    NsyncManifestAnchorRule,
    NsyncManifestNegative,
    NsyncManifestSample,
    load_nsync_manifest,
)

# Constants for precomputed data directories
PRECOMPUTED_DIR_NAME = ".precomputed"


class DummyDataset(Dataset):
    """Produce random latents and prompt embeddings. For minimal demonstration and benchmarking purposes"""

    def __init__(
        self,
        width: int = 1024,
        height: int = 1024,
        num_frames: int = 25,
        fps: int = 24,
        dataset_length: int = 200,
        latent_dim: int = 128,
        latent_spatial_compression_ratio: int = 32,
        latent_temporal_compression_ratio: int = 8,
        prompt_embed_dim: int = 4096,
        prompt_sequence_length: int = 256,
    ) -> None:
        if width % 32 != 0:
            raise ValueError(f"Width must be divisible by 32, got {width=}")

        if height % 32 != 0:
            raise ValueError(f"Height must be divisible by 32, got {height=}")

        if num_frames % 8 != 1:
            raise ValueError(f"Number of frames must have a remainder of 1 when divided by 8, got {num_frames=}")

        self.width = width
        self.height = height
        self.num_frames = num_frames
        self.fps = fps
        self.dataset_length = dataset_length
        self.latent_dim = latent_dim
        self.num_latent_frames = (num_frames - 1) // latent_temporal_compression_ratio + 1
        self.latent_height = height // latent_spatial_compression_ratio
        self.latent_width = width // latent_spatial_compression_ratio
        self.latent_sequence_length = self.num_latent_frames * self.latent_height * self.latent_width
        self.prompt_embed_dim = prompt_embed_dim
        self.prompt_sequence_length = prompt_sequence_length

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> dict[str, dict[str, Tensor]]:
        return {
            "latent_conditions": {
                "latents": torch.randn(
                    self.latent_dim,
                    self.num_latent_frames,
                    self.latent_height,
                    self.latent_width,
                ),
                "num_frames": self.num_latent_frames,
                "height": self.latent_height,
                "width": self.latent_width,
                "fps": self.fps,
            },
            "text_conditions": {
                "video_prompt_embeds": torch.randn(
                    self.prompt_sequence_length,
                    self.prompt_embed_dim,
                ),
                "audio_prompt_embeds": torch.randn(
                    self.prompt_sequence_length,
                    self.prompt_embed_dim,
                ),
                "prompt_attention_mask": torch.ones(
                    self.prompt_sequence_length,
                    dtype=torch.bool,
                ),
            },
        }


class PrecomputedDataset(Dataset):
    def __init__(self, data_root: str, data_sources: dict[str, str] | list[str] | None = None) -> None:
        """
        Generic dataset for loading precomputed data from multiple sources.
        Args:
            data_root: Root directory containing preprocessed data
            data_sources: Either:
              - Dict mapping directory names to output keys
              - List of directory names (keys will equal values)
              - None (defaults to ["latents", "conditions"])
        Example:
            # Standard mode (list)
            dataset = PrecomputedDataset("data/", ["latents", "conditions"])
            # Standard mode (dict)
            dataset = PrecomputedDataset("data/", {"latents": "latent_conditions", "conditions": "text_conditions"})
            # IC-LoRA mode
            dataset = PrecomputedDataset("data/", ["latents", "conditions", "reference_latents"])
        Note:
            Latents are always returned in non-patchified format [C, F, H, W].
            Legacy patchified format [seq_len, C] is automatically converted.
        """
        super().__init__()

        self.data_root = self._setup_data_root(data_root)
        self.data_sources = self._normalize_data_sources(data_sources)
        self.source_paths = self._setup_source_paths()
        self.sample_files = self._discover_samples()
        self._validate_setup()

    @staticmethod
    def _setup_data_root(data_root: str) -> Path:
        """Setup and validate the data root directory."""
        data_root = Path(data_root).expanduser().resolve()

        if not data_root.exists():
            raise FileNotFoundError(f"Data root directory does not exist: {data_root}")

        # If the given path is the dataset root, use the precomputed subdirectory
        if (data_root / PRECOMPUTED_DIR_NAME).exists():
            data_root = data_root / PRECOMPUTED_DIR_NAME

        return data_root

    @staticmethod
    def _normalize_data_sources(data_sources: dict[str, str] | list[str] | None) -> dict[str, str]:
        """Normalize data_sources input to a consistent dict format."""
        if data_sources is None:
            # Default sources
            return {"latents": "latent_conditions", "conditions": "text_conditions"}
        elif isinstance(data_sources, list):
            # Convert list to dict where keys equal values
            return {source: source for source in data_sources}
        elif isinstance(data_sources, dict):
            return data_sources.copy()
        else:
            raise TypeError(f"data_sources must be dict, list, or None, got {type(data_sources)}")

    def _setup_source_paths(self) -> dict[str, Path]:
        """Map data source names to their actual directory paths."""
        source_paths = {}

        for dir_name in self.data_sources:
            source_path = self.data_root / dir_name
            source_paths[dir_name] = source_path

            # Check that all sources exist.
            if not source_path.exists():
                raise FileNotFoundError(f"Required {dir_name} directory does not exist: {source_path}")

        return source_paths

    def _discover_samples(self) -> dict[str, list[Path]]:
        """Discover all valid sample files across all data sources."""
        # Use first data source as the reference to discover samples
        data_key = "latents" if "latents" in self.data_sources else next(iter(self.data_sources.keys()))
        data_path = self.source_paths[data_key]
        data_files = sorted(data_path.glob("**/*.pt"))

        if not data_files:
            raise ValueError(f"No data files found in {data_path}")

        # Initialize sample files dict
        sample_files = {output_key: [] for output_key in self.data_sources.values()}

        # For each data file, find corresponding files in other sources
        for data_file in data_files:
            rel_path = data_file.relative_to(data_path)

            # Check if corresponding files exist in ALL sources
            if self._all_source_files_exist(data_file, rel_path):
                self._fill_sample_data_files(data_file, rel_path, sample_files)

        return sample_files

    def _all_source_files_exist(self, data_file: Path, rel_path: Path) -> bool:
        """Check if corresponding files exist in all data sources."""
        for dir_name in self.data_sources:
            expected_path = self._get_expected_file_path(dir_name, data_file, rel_path)
            if not expected_path.exists():
                logger.warning(
                    f"No matching {dir_name} file found for: {data_file.name} (expected in: {expected_path})"
                )
                return False

        return True

    def _get_expected_file_path(self, dir_name: str, data_file: Path, rel_path: Path) -> Path:
        """Get the expected file path for a given data source."""
        source_path = self.source_paths[dir_name]

        # For conditions, handle legacy naming where latent_X.pt maps to condition_X.pt
        if dir_name == "conditions" and data_file.name.startswith("latent_"):
            return source_path / f"condition_{data_file.stem[7:]}.pt"

        return source_path / rel_path

    def _fill_sample_data_files(self, data_file: Path, rel_path: Path, sample_files: dict[str, list[Path]]) -> None:
        """Add a valid sample to the sample_files tracking."""
        for dir_name, output_key in self.data_sources.items():
            expected_path = self._get_expected_file_path(dir_name, data_file, rel_path)
            sample_files[output_key].append(expected_path.relative_to(self.source_paths[dir_name]))

    def _validate_setup(self) -> None:
        """Validate that the dataset setup is correct."""
        if not self.sample_files:
            raise ValueError("No valid samples found - all data sources must have matching files")

        # Verify all output keys have the same number of samples
        sample_counts = {key: len(files) for key, files in self.sample_files.items()}
        if len(set(sample_counts.values())) > 1:
            raise ValueError(f"Mismatched sample counts across sources: {sample_counts}")

    def __len__(self) -> int:
        # Use the first output key as reference count
        first_key = next(iter(self.sample_files.keys()))
        return len(self.sample_files[first_key])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        result = {}

        for dir_name, output_key in self.data_sources.items():
            source_path = self.source_paths[dir_name]
            file_rel_path = self.sample_files[output_key][index]
            file_path = source_path / file_rel_path

            try:
                data = torch.load(file_path, map_location="cpu", weights_only=True)

                # Normalize video latent format if this is a latent source
                if "latent" in dir_name.lower():
                    data = self._normalize_video_latents(data)

                result[output_key] = data
            except Exception as e:
                raise RuntimeError(f"Failed to load {output_key} from {file_path}: {e}") from e

        # Add index for debugging
        result["idx"] = index
        return result

    @staticmethod
    def _normalize_video_latents(data: dict) -> dict:
        """
        Normalize video latents to non-patchified format [C, F, H, W].
        Used for keeping backward compatibility with legacy datasets.
        """
        latents = data["latents"]

        # Check if latents are in legacy patchified format [seq_len, C]
        if latents.dim() == 2:
            # Legacy format: [seq_len, C] where seq_len = F * H * W
            num_frames = data["num_frames"]
            height = data["height"]
            width = data["width"]

            # Unpatchify: [seq_len, C] -> [C, F, H, W]
            latents = rearrange(
                latents,
                "(f h w) c -> c f h w",
                f=num_frames,
                h=height,
                w=width,
            )

            # Update the data dict with unpatchified latents
            data = data.copy()
            data["latents"] = latents

        return data


class ManifestNsyncDataset(Dataset):
    """Wrap a precomputed positive dataset with manifest-driven NSYNC branches."""

    def __init__(
        self,
        base_dataset: PrecomputedDataset,
        *,
        manifest_path: str | Path,
        negative_conditions_dir: str,
        negative_latents_dir: str,
        negative_audio_latents_dir: str,
        with_audio: bool,
        use_anchor: bool,
        anchor_retry_count: int = 10,
    ) -> None:
        super().__init__()
        self._base_dataset = base_dataset
        self._manifest = load_nsync_manifest(manifest_path)
        self._data_root = base_dataset.data_root
        self._negative_conditions_root = self._data_root / negative_conditions_dir
        self._negative_latents_root = self._data_root / negative_latents_dir
        self._negative_audio_latents_root = self._data_root / negative_audio_latents_dir
        self._with_audio = with_audio
        self._use_anchor = use_anchor
        self._anchor_retry_count = anchor_retry_count
        self._sample_path_to_dataset_index = self._build_dataset_index_map()
        self._category_to_manifest_indices = self._build_category_index()

        if not self._negative_conditions_root.exists():
            raise FileNotFoundError(f"Advanced NSYNC negative conditions directory not found: {self._negative_conditions_root}")

    def __len__(self) -> int:
        return len(self._manifest.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample_entry = self._manifest.samples[index]
        positive_sample = self._base_dataset[self._sample_path_to_dataset_index[sample_entry.sample_rel_path]]

        negative_branches = [
            self._build_negative_branch_sample(positive_sample, negative_entry)
            for negative_entry in sample_entry.negatives
        ]

        if self._use_anchor:
            anchor_indices = self._resolve_anchor_indices(index, sample_entry)
            anchor_branches = [
                self._base_dataset[self._sample_path_to_dataset_index[self._manifest.samples[anchor_index].sample_rel_path]]
                for anchor_index in anchor_indices
            ]
        else:
            anchor_branches = []

        return {
            "positive": positive_sample,
            "negative_branches": negative_branches,
            "anchor_branches": anchor_branches,
        }

    def _build_dataset_index_map(self) -> dict[str, int]:
        if "latents" not in self._base_dataset.sample_files:
            raise ValueError("Advanced NSYNC requires the base dataset to expose 'latents' sample files")

        mapping = {
            str(relative_path): index
            for index, relative_path in enumerate(self._base_dataset.sample_files["latents"])
        }

        missing_paths = [
            sample.sample_rel_path
            for sample in self._manifest.samples
            if sample.sample_rel_path not in mapping
        ]
        if missing_paths:
            raise ValueError(
                "Advanced NSYNC manifest references samples that are missing from the positive dataset: "
                f"{missing_paths[:5]}"
            )
        return mapping

    def _build_category_index(self) -> dict[str, set[int]]:
        category_to_indices = {category: set[int]() for category in self._manifest.categories}
        for sample_index, sample in enumerate(self._manifest.samples):
            for category in sample.categories:
                category_to_indices[category].add(sample_index)
        return category_to_indices

    def _build_negative_branch_sample(
        self,
        positive_sample: dict[str, Any],
        negative_entry: NsyncManifestNegative,
    ) -> dict[str, Any]:
        branch = {
            "latents": positive_sample["latents"]
            if negative_entry.media == "positive"
            else self._load_data_file(self._negative_latents_root / negative_entry.latent_rel_path, normalize_video=True),
            "conditions": self._load_data_file(self._negative_conditions_root / negative_entry.condition_rel_path),
            "idx": positive_sample["idx"],
        }

        if "audio_latents" in positive_sample:
            if negative_entry.media == "positive":
                branch["audio_latents"] = positive_sample["audio_latents"]
            elif self._with_audio and negative_entry.audio_latent_rel_path is not None:
                branch["audio_latents"] = self._load_data_file(
                    self._negative_audio_latents_root / negative_entry.audio_latent_rel_path
                )

        if "ref_latents" in positive_sample:
            branch["ref_latents"] = positive_sample["ref_latents"]

        return branch

    def _resolve_anchor_indices(self, sample_index: int, sample_entry: NsyncManifestSample) -> list[int]:
        if not sample_entry.anchors:
            return []

        failed_rules: list[int] = []
        for _attempt in range(self._anchor_retry_count):
            used_indices = {sample_index}
            selected_indices: list[int] = []
            failed_rules.clear()

            for rule_index, rule in enumerate(sample_entry.anchors):
                candidates = self._candidate_anchor_indices(sample_index, rule) - used_indices
                if not candidates:
                    failed_rules.append(rule_index)
                    break

                chosen_index = random.choice(sorted(candidates))
                used_indices.add(chosen_index)
                selected_indices.append(chosen_index)

            if not failed_rules and len(selected_indices) == len(sample_entry.anchors):
                return selected_indices

        raise ValueError(
            f"Failed to resolve unique anchors for sample '{sample_entry.media_path}' after {self._anchor_retry_count} "
            f"attempt(s). Failing rule indices: {failed_rules}"
        )

    def _candidate_anchor_indices(self, sample_index: int, rule: NsyncManifestAnchorRule) -> set[int]:
        required_candidates = self._indices_for_categories(rule.required_categories)
        required_candidates.discard(sample_index)
        if not rule.extra_random_category:
            return required_candidates

        extra_category = random.choice(rule.extra_category_candidates)
        candidate_categories = [*rule.required_categories, extra_category]
        candidates = self._indices_for_categories(candidate_categories)
        candidates.discard(sample_index)
        return candidates

    def _indices_for_categories(self, categories: list[str]) -> set[int]:
        category_sets = [self._category_to_manifest_indices[category] for category in categories]
        return set.intersection(*category_sets) if category_sets else set()

    @staticmethod
    def _load_data_file(path: Path, *, normalize_video: bool = False) -> dict[str, Any]:
        if not path.is_file():
            raise FileNotFoundError(f"Advanced NSYNC artifact not found: {path}")

        data = torch.load(path, map_location="cpu", weights_only=True)
        if normalize_video:
            data = PrecomputedDataset._normalize_video_latents(data)
        return data


def collate_manifest_nsync_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate positive samples plus variable negative/anchor branch slots."""
    if not batch:
        raise ValueError("Cannot collate an empty advanced NSYNC batch")

    collated = default_collate([item["positive"] for item in batch])
    batch_size = len(batch)

    negative_slots, negative_slot_weights = _collate_branch_slots(batch, branch_key="negative_branches", batch_size=batch_size)
    anchor_slots, anchor_slot_weights = _collate_branch_slots(batch, branch_key="anchor_branches", batch_size=batch_size)

    collated["negative_slots"] = negative_slots
    collated["negative_slot_weights"] = negative_slot_weights
    collated["anchor_slots"] = anchor_slots
    collated["anchor_slot_weights"] = anchor_slot_weights
    return collated


def _collate_branch_slots(
    batch: list[dict[str, Any]],
    *,
    branch_key: str,
    batch_size: int,
) -> tuple[list[dict[str, Any]], Tensor]:
    max_slots = max(len(item[branch_key]) for item in batch)
    collated_slots: list[dict[str, Any]] = []
    slot_weights: list[float] = []

    for slot_index in range(max_slots):
        slot_items = [item[branch_key][slot_index] for item in batch if len(item[branch_key]) > slot_index]
        if not slot_items:
            continue

        collated_slots.append(default_collate(slot_items))
        slot_weights.append(len(slot_items) / batch_size)

    return collated_slots, torch.tensor(slot_weights, dtype=torch.float32)
