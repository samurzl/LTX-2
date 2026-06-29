"""Long-lived model cache used by warm preprocessing and training sessions."""

from __future__ import annotations

import gc
import hashlib
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TypeVar

import torch

from ltx_trainer import logger

ModelT = TypeVar("ModelT")
ModelLoader = Callable[[torch.device], ModelT]


def _source_fingerprint(source: str | Path) -> tuple[str, int, int]:
    """Return a cheap identity that invalidates a cache entry when its source changes."""
    path = Path(source).expanduser().resolve()
    stat = path.stat()
    if path.is_file():
        return str(path), stat.st_size, stat.st_mtime_ns

    digest = hashlib.blake2b(digest_size=8)
    total_size = 0
    for child in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        child_stat = child.stat()
        total_size += child_stat.st_size
        digest.update(str(child.relative_to(path)).encode())
        digest.update(child_stat.st_size.to_bytes(8, "little", signed=False))
        digest.update(child_stat.st_mtime_ns.to_bytes(8, "little", signed=False))
    return str(path), total_size, int.from_bytes(digest.digest(), "little")


@dataclass(frozen=True)
class ModelCacheKey:
    """Identity of one reusable model representation."""

    component: str
    source: tuple[str, int, int]
    dtype: str
    options: tuple[tuple[str, str], ...] = ()

    @classmethod
    def create(
        cls,
        component: str,
        source: str | Path,
        dtype: torch.dtype,
        **options: object,
    ) -> ModelCacheKey:
        normalized_options = tuple(sorted((name, str(value)) for name, value in options.items()))
        return cls(
            component=component,
            source=_source_fingerprint(source),
            dtype=str(dtype),
            options=normalized_options,
        )

    @property
    def label(self) -> str:
        return f"{self.component} ({Path(self.source[0]).name})"


@dataclass
class _CacheEntry:
    model: Any
    offloadable: bool


class WarmModelPool:
    """Cache model objects across multiple jobs in one Python process.

    Models remain on their current device until another job needs the memory. Frozen
    models are moved to CPU rather than destroyed. Components that cannot be moved
    after construction (currently 8-bit bitsandbytes models) are evicted when a
    different component group needs the device.
    """

    def __init__(self) -> None:
        self._entries: dict[ModelCacheKey, _CacheEntry] = {}
        self._lock = threading.RLock()

    def get_or_load(
        self,
        key: ModelCacheKey,
        loader: ModelLoader[ModelT],
        device: torch.device | str,
        *,
        offloadable: bool = True,
        move_cached: bool = True,
    ) -> ModelT:
        """Return a cached model or load it once on ``device``."""
        target = torch.device(device)
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                logger.info("Warm model cache miss: loading %s", key.label)
                model = loader(target)
                self._entries[key] = _CacheEntry(model=model, offloadable=offloadable)
                return model

            logger.info("Warm model cache hit: reusing %s", key.label)
            if move_cached:
                self._move(entry.model, target)
            return entry.model

    def replace(self, key: ModelCacheKey, model: object, *, offloadable: bool = True) -> None:
        """Replace an entry after a temporary wrapper (such as PEFT) is removed."""
        with self._lock:
            self._entries[key] = _CacheEntry(model=model, offloadable=offloadable)

    def offload_all(self, *, exclude: set[ModelCacheKey] | None = None) -> None:
        """Move cached models to CPU, retaining the excluded entries on their device."""
        excluded = exclude or set()
        with self._lock:
            evicted: list[ModelCacheKey] = []
            for key, entry in self._entries.items():
                if key in excluded:
                    continue
                if entry.offloadable:
                    self._move(entry.model, torch.device("cpu"))
                else:
                    logger.info("Evicting non-offloadable warm model: %s", key.label)
                    evicted.append(key)

            for key in evicted:
                del self._entries[key]

        self._free_device_cache()

    def offload(self, key: ModelCacheKey) -> None:
        """Move one cached model to CPU, or evict it when it is not movable."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return
            if entry.offloadable:
                self._move(entry.model, torch.device("cpu"))
            else:
                del self._entries[key]
        self._free_device_cache()

    def clear(self) -> None:
        """Destroy all cached models and release allocator caches."""
        with self._lock:
            self._entries.clear()
        self._free_device_cache()

    def contains(self, key: ModelCacheKey) -> bool:
        with self._lock:
            return key in self._entries

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._entries)

    @staticmethod
    def _move(model: object, device: torch.device) -> None:
        move = getattr(model, "to", None)
        if move is None:
            return
        move(device)

    @staticmethod
    def _free_device_cache() -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __enter__(self) -> WarmModelPool:
        return self

    def __exit__(self, *_args: object) -> None:
        self.clear()
