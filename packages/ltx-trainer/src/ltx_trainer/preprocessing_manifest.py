from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ltx_trainer import logger

MANIFEST_FILENAME = "preprocessing_manifest.json"
MANIFEST_SCHEMA_VERSION = 1


def stable_hash(payload: Any) -> str:
    """Create a stable hash for JSON-serializable preprocessing metadata."""

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def resolved_path_string(path: str | Path) -> str:
    """Resolve a path for config signatures."""

    return str(Path(path).expanduser().resolve())


def file_stat_payload(path: str | Path) -> dict[str, int]:
    """Capture the file metadata used for incremental invalidation."""

    stat_result = Path(path).expanduser().resolve().stat()
    return {
        "mtime_ns": stat_result.st_mtime_ns,
        "size": stat_result.st_size,
    }


def infer_manifest_root(output_dir: str | Path, source_name: str) -> Path:
    """Infer the manifest root from a preprocessing output directory."""

    output_path = Path(output_dir).expanduser().resolve()
    if output_path.name == source_name:
        return output_path.parent
    return output_path


def source_dir_value(root: str | Path, source_dir: str | Path) -> str:
    """Store source directories relative to the manifest root when possible."""

    root_path = Path(root).expanduser().resolve()
    source_path = Path(source_dir).expanduser().resolve()

    try:
        relative = source_path.relative_to(root_path)
    except ValueError:
        return source_path.as_posix()

    if relative == Path("."):
        return "."
    return relative.as_posix()


def resolve_source_dir(root: str | Path, source_dir: str) -> Path:
    """Resolve a source directory from manifest storage."""

    root_path = Path(root).expanduser().resolve()
    dir_path = Path(source_dir)
    if dir_path.is_absolute():
        return dir_path
    return (root_path / dir_path).resolve()


def normalize_rel_path(path: str | Path) -> str:
    """Normalize manifest sample paths to a portable POSIX form."""

    path_obj = Path(path)
    if path_obj == Path("."):
        return "."
    return path_obj.as_posix()


def dedupe_paths(paths: Iterable[str | Path]) -> list[str]:
    """Deduplicate sample paths while preserving order."""

    deduped: list[str] = []
    seen: set[str] = set()

    for path in paths:
        normalized = normalize_rel_path(path)
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)

    return deduped


class PreprocessingManifest:
    """Read and write the manifest used for incremental preprocessing."""

    def __init__(self, root: str | Path, data: dict[str, Any] | None = None) -> None:
        self.root = Path(root).expanduser().resolve()
        self.path = self.root / MANIFEST_FILENAME
        self.data = data or {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "sources": {},
        }
        self.data.setdefault("schema_version", MANIFEST_SCHEMA_VERSION)
        self.data.setdefault("sources", {})

    @classmethod
    def load(cls, root: str | Path) -> PreprocessingManifest:
        """Load an existing manifest or initialize a new one."""

        root_path = Path(root).expanduser().resolve()
        manifest_path = root_path / MANIFEST_FILENAME

        if not manifest_path.exists():
            return cls(root_path)

        with manifest_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if data.get("schema_version") != MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported preprocessing manifest version in {manifest_path}: "
                f"{data.get('schema_version')!r}"
            )

        if not isinstance(data.get("sources"), dict):
            raise ValueError(f"Invalid preprocessing manifest structure in {manifest_path}")

        return cls(root_path, data=data)

    def save(self) -> None:
        """Atomically persist the manifest."""

        self.root.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(".json.tmp")

        with tmp_path.open("w", encoding="utf-8") as file:
            json.dump(self.data, file, indent=2, sort_keys=True)
            file.write("\n")

        tmp_path.replace(self.path)

    def get_source(self, source_name: str) -> dict[str, Any] | None:
        """Return a source entry if it exists."""

        source = self.data["sources"].get(source_name)
        if source is None:
            return None

        source.setdefault("active", True)
        source.setdefault("source_dir", source_name)
        source.setdefault("config_signature", "")
        source.setdefault("active_samples", [])
        source.setdefault("sample_fingerprints", {})
        return source

    def update_source(
        self,
        source_name: str,
        source_dir: str | Path,
        config_signature: str,
        active_samples: Iterable[str | Path],
        sample_fingerprints: dict[str | Path, str],
        *,
        active: bool = True,
    ) -> None:
        """Replace a source entry with the latest preprocessing state."""

        normalized_samples = dedupe_paths(active_samples)
        normalized_fingerprints = {
            normalize_rel_path(sample_path): fingerprint
            for sample_path, fingerprint in sample_fingerprints.items()
        }

        self.data["sources"][source_name] = {
            "active": active,
            "source_dir": source_dir_value(self.root, source_dir),
            "config_signature": config_signature,
            "active_samples": normalized_samples,
            "sample_fingerprints": normalized_fingerprints,
        }

    def deactivate_source(self, source_name: str, source_dir: str | Path | None = None) -> None:
        """Mark a source inactive without deleting any files from disk."""

        existing_source = self.get_source(source_name)
        if existing_source is not None:
            dir_value = existing_source["source_dir"]
        elif source_dir is not None:
            dir_value = source_dir_value(self.root, source_dir)
        else:
            dir_value = source_name

        self.data["sources"][source_name] = {
            "active": False,
            "source_dir": dir_value,
            "config_signature": "",
            "active_samples": [],
            "sample_fingerprints": {},
        }

    def is_sample_current(
        self,
        source_name: str,
        sample_path: str | Path,
        sample_fingerprint: str,
        config_signature: str,
        output_file: str | Path,
    ) -> bool:
        """Check whether a cached output can be reused."""

        source = self.get_source(source_name)
        if source is None or not source["active"]:
            return False

        normalized_sample = normalize_rel_path(sample_path)
        if source["config_signature"] != config_signature:
            return False

        if normalized_sample not in source["active_samples"]:
            return False

        if source["sample_fingerprints"].get(normalized_sample) != sample_fingerprint:
            return False

        return Path(output_file).expanduser().resolve().exists()

    def has_source(self, source_name: str) -> bool:
        """Return whether the manifest contains a source entry."""

        return source_name in self.data["sources"]

    def require_active_source(self, source_name: str) -> dict[str, Any]:
        """Return a source entry or raise when it is unavailable for training."""

        source = self.get_source(source_name)
        if source is None:
            raise FileNotFoundError(
                f"Required {source_name} source is missing from preprocessing manifest: {self.path}"
            )

        if not source["active"]:
            raise FileNotFoundError(
                f"Required {source_name} source is inactive in preprocessing manifest: {self.path}"
            )

        return source

    def log_skip_summary(self, source_name: str, total: int, processed: int) -> None:
        """Log a short incremental-processing summary."""

        skipped = total - processed
        if skipped > 0:
            logger.info(f"Skipping {skipped:,} unchanged {source_name}")
