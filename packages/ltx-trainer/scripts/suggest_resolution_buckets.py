#!/usr/bin/env python3

"""
Suggest preprocessing resolution buckets for an LTX trainer dataset.

The output is the exact string accepted by ``process_dataset.py`` and
``process_videos.py``:

    uv run python scripts/suggest_resolution_buckets.py dataset.json
    960x544x49
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import av
import numpy as np
import pandas as pd
import typer
from pillow_heif import register_heif_opener
from process_videos import VAE_SPATIAL_FACTOR, VAE_TEMPORAL_FACTOR, parse_resolution_buckets
from rich.console import Console

from ltx_trainer.utils import open_image_as_srgb

register_heif_opener()

DEFAULT_TARGET_PIXELS = 960 * 544
DEFAULT_MAX_BUCKETS = 3
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

app = typer.Typer(
    pretty_exceptions_enable=False,
    no_args_is_help=True,
    help="Inspect a dataset and print a sensible --resolution-buckets string for preprocessing.",
)
err_console = Console(stderr=True)


@dataclass(frozen=True)
class MediaInfo:
    path: Path
    width: int
    height: int
    frames: int
    is_image: bool

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height

    @property
    def area(self) -> int:
        return self.width * self.height


@dataclass(frozen=True)
class BucketSuggestion:
    width: int
    height: int
    frames: int
    count: int
    median_aspect_ratio: float

    @property
    def area(self) -> int:
        return self.width * self.height

    def as_preprocess_value(self) -> str:
        return f"{self.width}x{self.height}x{self.frames}"


def _load_media_paths(dataset_file: Path, media_column: str) -> list[Path]:
    data_root = dataset_file.parent

    if dataset_file.suffix == ".csv":
        df = pd.read_csv(dataset_file)
        if media_column not in df.columns:
            raise ValueError(f"Column '{media_column}' not found in CSV file")
        values = df[media_column].tolist()
    elif dataset_file.suffix == ".json":
        with open(dataset_file, "r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, list):
            raise ValueError("JSON file must contain a list of objects")
        values = _values_from_records(data, media_column)
    elif dataset_file.suffix == ".jsonl":
        values = []
        with open(dataset_file, "r", encoding="utf-8") as file:
            for line in file:
                values.extend(_values_from_records([json.loads(line)], media_column))
    else:
        raise ValueError("Expected `dataset_file` to be a CSV, JSON, or JSONL file.")

    paths = []
    for value in values:
        media_path = Path(str(value).strip())
        paths.append(media_path if media_path.is_absolute() else data_root / media_path)
    return paths


def _values_from_records(records: list[dict[str, Any]], media_column: str) -> list[Any]:
    values = []
    for entry in records:
        if media_column not in entry:
            raise ValueError(f"Key '{media_column}' not found in dataset entry")
        values.append(entry[media_column])
    return values


def _inspect_media(path: Path) -> MediaInfo:
    if not path.is_file():
        raise FileNotFoundError(path)

    if path.suffix.lower() in IMAGE_EXTENSIONS:
        image = open_image_as_srgb(path)
        width, height = image.size
        return MediaInfo(path=path, width=width, height=height, frames=1, is_image=True)

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        width = stream.width
        height = stream.height
        frames = _video_frame_count(container, stream)
        return MediaInfo(path=path, width=width, height=height, frames=frames, is_image=False)


def _video_frame_count(container: av.container.InputContainer, stream: av.video.stream.VideoStream) -> int:
    if stream.frames > 0:
        return stream.frames

    rate = stream.average_rate or stream.base_rate
    if stream.duration and stream.time_base and rate:
        return round(float(stream.duration * stream.time_base * rate))

    return sum(1 for _ in container.decode(video=0))


def _kmeans_1d(values: list[float], k: int) -> list[int]:
    values_array = np.asarray(values, dtype=np.float64)
    if k == 1:
        return [0] * len(values)

    centers = np.quantile(values_array, np.linspace(0, 1, k + 2)[1:-1])
    for _ in range(25):
        distances = np.abs(values_array[:, None] - centers[None, :])
        assignments = np.argmin(distances, axis=1)
        next_centers = np.array(
            [
                values_array[assignments == cluster_idx].mean()
                if np.any(assignments == cluster_idx)
                else centers[cluster_idx]
                for cluster_idx in range(k)
            ]
        )
        if np.allclose(centers, next_centers):
            break
        centers = next_centers

    distances = np.abs(values_array[:, None] - centers[None, :])
    return np.argmin(distances, axis=1).tolist()


def _cluster_items_by_aspect(
    items: list[MediaInfo],
    max_buckets: int,
    min_bucket_share: float,
    max_aspect_spread: float,
) -> list[list[MediaInfo]]:
    if max_buckets < 1:
        raise ValueError("max_buckets must be >= 1")
    if len(items) <= 1 or max_buckets == 1:
        return [items]

    log_aspects = [math.log(item.aspect_ratio) for item in items]
    min_cluster_size = max(1, math.ceil(len(items) * min_bucket_share))
    best_clusters = [items]

    for k in range(1, min(max_buckets, len(items)) + 1):
        assignments = _kmeans_1d(log_aspects, k)
        clusters = [
            [item for item, assignment in zip(items, assignments, strict=True) if assignment == i]
            for i in range(k)
        ]
        if any(len(cluster) < min_cluster_size for cluster in clusters):
            continue

        clusters.sort(key=_median_aspect_ratio)
        best_clusters = clusters
        if _clusters_fit_aspect_spread(clusters, max_aspect_spread):
            return clusters

    return best_clusters


def _clusters_fit_aspect_spread(clusters: list[list[MediaInfo]], max_aspect_spread: float) -> bool:
    for cluster in clusters:
        aspects = [item.aspect_ratio for item in cluster]
        if max(aspects) / min(aspects) > max_aspect_spread:
            return False
    return True


def _median_aspect_ratio(items: list[MediaInfo]) -> float:
    return float(np.median([item.aspect_ratio for item in items]))


def _choose_frames(items: list[MediaInfo], min_coverage: float) -> int:
    frame_counts = np.asarray([item.frames for item in items], dtype=np.float64)
    if np.all(frame_counts == 1):
        return 1

    covered_count = int(np.floor(np.quantile(frame_counts, 1.0 - min_coverage)))
    covered_count = max(1, covered_count)
    if covered_count == 1:
        return 1

    return 1 + ((covered_count - 1) // VAE_TEMPORAL_FACTOR) * VAE_TEMPORAL_FACTOR


def _choose_resolution(
    items: list[MediaInfo],
    target_pixels: int,
    pixel_quantile: float,
) -> tuple[int, int]:
    median_log_aspect = float(np.median([math.log(item.aspect_ratio) for item in items]))
    aspect_ratio = math.exp(median_log_aspect)
    native_area = int(np.floor(np.quantile([item.area for item in items], pixel_quantile)))
    target_area = max(VAE_SPATIAL_FACTOR * VAE_SPATIAL_FACTOR, min(target_pixels, native_area))

    width = _snap_dimension(math.sqrt(target_area * aspect_ratio))
    height = _snap_dimension(math.sqrt(target_area / aspect_ratio))
    return width, height


def _snap_dimension(value: float) -> int:
    return max(VAE_SPATIAL_FACTOR, round(value / VAE_SPATIAL_FACTOR) * VAE_SPATIAL_FACTOR)


def _bucket_for_items(
    items: list[MediaInfo],
    *,
    target_pixels: int,
    pixel_quantile: float,
    min_coverage: float,
    force_frames: int | None = None,
) -> BucketSuggestion:
    width, height = _choose_resolution(items, target_pixels=target_pixels, pixel_quantile=pixel_quantile)
    frames = force_frames if force_frames is not None else _choose_frames(items, min_coverage=min_coverage)
    return BucketSuggestion(
        width=width,
        height=height,
        frames=frames,
        count=len(items),
        median_aspect_ratio=_median_aspect_ratio(items),
    )


def suggest_resolution_buckets(  # noqa: PLR0913
    media: list[MediaInfo],
    *,
    max_buckets: int,
    target_pixels: int,
    pixel_quantile: float,
    min_coverage: float,
    min_bucket_share: float,
    max_aspect_spread: float,
) -> list[BucketSuggestion]:
    images = [item for item in media if item.is_image]
    videos = [item for item in media if not item.is_image]
    buckets = []

    if images:
        image_clusters = _cluster_items_by_aspect(
            images,
            max_buckets=1 if videos else max_buckets,
            min_bucket_share=min_bucket_share,
            max_aspect_spread=max_aspect_spread,
        )
        buckets.extend(
            _bucket_for_items(
                cluster,
                target_pixels=target_pixels,
                pixel_quantile=pixel_quantile,
                min_coverage=min_coverage,
                force_frames=1,
            )
            for cluster in image_clusters
        )

    if videos:
        remaining_bucket_count = max(1, max_buckets - len(buckets))
        video_clusters = _cluster_items_by_aspect(
            videos,
            max_buckets=remaining_bucket_count,
            min_bucket_share=min_bucket_share,
            max_aspect_spread=max_aspect_spread,
        )
        buckets.extend(
            _bucket_for_items(
                cluster,
                target_pixels=target_pixels,
                pixel_quantile=pixel_quantile,
                min_coverage=min_coverage,
            )
            for cluster in video_clusters
        )

    return _dedupe_and_sort_buckets(buckets)


def _dedupe_and_sort_buckets(buckets: list[BucketSuggestion]) -> list[BucketSuggestion]:
    merged: dict[tuple[int, int, int], BucketSuggestion] = {}
    for bucket in buckets:
        key = (bucket.width, bucket.height, bucket.frames)
        existing = merged.get(key)
        if existing is None:
            merged[key] = bucket
        else:
            merged[key] = BucketSuggestion(
                width=bucket.width,
                height=bucket.height,
                frames=bucket.frames,
                count=existing.count + bucket.count,
                median_aspect_ratio=bucket.median_aspect_ratio,
            )

    return sorted(
        merged.values(),
        key=lambda bucket: (-bucket.count, -bucket.frames, -bucket.area, bucket.width),
    )


def _format_buckets(buckets: list[BucketSuggestion]) -> str:
    bucket_string = ";".join(bucket.as_preprocess_value() for bucket in buckets)
    parse_resolution_buckets(bucket_string)
    return bucket_string


def _validate_options(  # noqa: PLR0913
    max_buckets: int,
    target_pixels: int,
    pixel_quantile: float,
    min_coverage: float,
    min_bucket_share: float,
    max_aspect_spread: float,
) -> None:
    if max_buckets < 1:
        raise typer.BadParameter("--max-buckets must be >= 1")
    if target_pixels < VAE_SPATIAL_FACTOR * VAE_SPATIAL_FACTOR:
        raise typer.BadParameter(f"--target-pixels must be at least {VAE_SPATIAL_FACTOR * VAE_SPATIAL_FACTOR}")
    if not 0.0 < pixel_quantile <= 1.0:
        raise typer.BadParameter("--pixel-quantile must be in (0, 1]")
    if not 0.0 < min_coverage <= 1.0:
        raise typer.BadParameter("--min-coverage must be in (0, 1]")
    if not 0.0 < min_bucket_share <= 1.0:
        raise typer.BadParameter("--min-bucket-share must be in (0, 1]")
    if max_aspect_spread < 1.0:
        raise typer.BadParameter("--max-aspect-spread must be >= 1")


def _print_details(buckets: list[BucketSuggestion], inspected_count: int, skipped_count: int) -> None:
    err_console.print(f"Inspected {inspected_count} media files; skipped {skipped_count}.")
    for bucket in buckets:
        err_console.print(
            f"- {bucket.as_preprocess_value()} "
            f"({bucket.count} items, median aspect {bucket.median_aspect_ratio:.3f})"
        )


@app.command()
def main(  # noqa: PLR0913
    dataset_file: str = typer.Argument(
        ...,
        help="Path to metadata file (CSV/JSON/JSONL) containing media paths",
    ),
    media_column: str = typer.Option(
        default="media_path",
        help="Column/key containing video or image paths",
    ),
    max_buckets: int = typer.Option(
        default=DEFAULT_MAX_BUCKETS,
        help="Maximum aspect-ratio buckets to suggest",
    ),
    target_pixels: int = typer.Option(
        default=DEFAULT_TARGET_PIXELS,
        help="Maximum target pixel area for each bucket before snapping to multiples of 32",
    ),
    pixel_quantile: float = typer.Option(
        default=0.75,
        help="Native resolution quantile used before applying --target-pixels",
    ),
    min_coverage: float = typer.Option(
        default=0.95,
        help="Approximate share of items that should have at least the suggested frame count",
    ),
    min_bucket_share: float = typer.Option(
        default=0.15,
        help="Smallest dataset share allowed for an aspect-ratio bucket",
    ),
    max_aspect_spread: float = typer.Option(
        default=1.35,
        help="Split aspect buckets until max/min aspect inside a bucket is at most this ratio",
    ),
    strict: bool = typer.Option(
        default=False,
        help="Fail on the first unreadable media file instead of skipping it",
    ),
    details: bool = typer.Option(
        default=False,
        help="Print inspected counts and bucket details to stderr",
    ),
) -> None:
    """Inspect media metadata and print a copy-pasteable bucket string."""
    _validate_options(
        max_buckets=max_buckets,
        target_pixels=target_pixels,
        pixel_quantile=pixel_quantile,
        min_coverage=min_coverage,
        min_bucket_share=min_bucket_share,
        max_aspect_spread=max_aspect_spread,
    )

    dataset_path = Path(dataset_file)
    if not dataset_path.is_file():
        raise typer.BadParameter(f"Dataset file not found: {dataset_file}")

    media_paths = _load_media_paths(dataset_path, media_column)
    inspected = []
    skipped_count = 0

    with err_console.status(f"Inspecting {len(media_paths)} media files...", spinner="dots"):
        for path in media_paths:
            try:
                inspected.append(_inspect_media(path))
            except Exception as exc:
                if strict:
                    raise
                skipped_count += 1
                if details:
                    err_console.print(f"Skipping {path}: {exc}")

    if not inspected:
        raise typer.BadParameter("No readable media files found in dataset")

    buckets = suggest_resolution_buckets(
        inspected,
        max_buckets=max_buckets,
        target_pixels=target_pixels,
        pixel_quantile=pixel_quantile,
        min_coverage=min_coverage,
        min_bucket_share=min_bucket_share,
        max_aspect_spread=max_aspect_spread,
    )
    bucket_string = _format_buckets(buckets)

    if details:
        _print_details(buckets, inspected_count=len(inspected), skipped_count=skipped_count)

    typer.echo(bucket_string)


if __name__ == "__main__":
    app()
