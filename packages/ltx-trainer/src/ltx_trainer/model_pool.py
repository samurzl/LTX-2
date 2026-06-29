"""Long-lived model cache used by warm preprocessing and training sessions."""

from __future__ import annotations

import gc
import hashlib
import threading
from dataclasses import dataclass, replace
from itertools import chain
from pathlib import Path
from typing import Any, Callable, Hashable, TypeVar

import torch

from ltx_trainer import logger

ModelT = TypeVar("ModelT")
ModelLoader = Callable[[torch.device], ModelT]
ModelStatusListener = Callable[["ModelStatus"], None]


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


@dataclass(frozen=True)
class ModelStatus:
    """A display-friendly snapshot of one model cache entry."""

    key: ModelCacheKey
    status: str
    device: str | None = None
    detail: str | None = None
    error: str | None = None
    offloadable: bool = True
    memory_bytes: tuple[tuple[str, int], ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "id": repr(self.key),
            "component": self.key.component,
            "name": Path(self.key.source[0]).name,
            "path": self.key.source[0],
            "size_bytes": self.key.source[1],
            "modified_ns": self.key.source[2],
            "dtype": self.key.dtype,
            "options": dict(self.key.options),
            "status": self.status,
            "device": self.device,
            "detail": self.detail,
            "error": self.error,
            "offloadable": self.offloadable,
            "memory_bytes": dict(self.memory_bytes),
        }


class WarmModelPool:
    """Cache model objects across multiple jobs in one Python process.

    Models remain on their current device until another job needs the memory. Frozen
    models are moved to CPU rather than destroyed. Components that cannot be moved
    after construction (currently 8-bit bitsandbytes models) are evicted when a
    different component group needs the device.
    """

    def __init__(self, status_listener: ModelStatusListener | None = None) -> None:
        self._entries: dict[ModelCacheKey, _CacheEntry] = {}
        self._artifacts: dict[Hashable, object] = {}
        self._statuses: dict[ModelCacheKey, ModelStatus] = {}
        self._status_snapshot: tuple[ModelStatus, ...] = ()
        self._status_listener = status_listener
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
                self._set_status(key, "loading", detail=f"Loading on {target}", offloadable=offloadable)
                try:
                    model = loader(target)
                except Exception as error:
                    self._set_status(
                        key,
                        "unloaded",
                        detail="Load failed",
                        error=str(error),
                        offloadable=offloadable,
                    )
                    raise
                self._entries[key] = _CacheEntry(model=model, offloadable=offloadable)
                self._set_status(
                    key,
                    "loaded",
                    device=self._model_device(model),
                    detail="Ready",
                    offloadable=offloadable,
                    memory_bytes=self._model_memory(model),
                )
                return model

            logger.info("Warm model cache hit: reusing %s", key.label)
            if move_cached:
                self._set_status(
                    key,
                    "loading",
                    device=self._model_device(entry.model),
                    detail=f"Moving to {target}",
                    offloadable=entry.offloadable,
                    memory_bytes=self._model_memory(entry.model),
                )
                try:
                    self.move_model(entry.model, target)
                except Exception as error:
                    self._set_status(
                        key,
                        "unloaded",
                        detail="Device move failed",
                        error=str(error),
                        offloadable=entry.offloadable,
                    )
                    raise
            self._set_status(
                key,
                "loaded",
                device=self._model_device(entry.model),
                detail="Ready",
                offloadable=entry.offloadable,
                memory_bytes=self._model_memory(entry.model),
            )
            return entry.model

    def replace(self, key: ModelCacheKey, model: object, *, offloadable: bool = True) -> None:
        """Replace an entry after a temporary wrapper (such as PEFT) is removed."""
        with self._lock:
            self._entries[key] = _CacheEntry(model=model, offloadable=offloadable)
            self._set_status(
                key,
                "loaded",
                device=self._model_device(model),
                detail="Ready",
                offloadable=offloadable,
                memory_bytes=self._model_memory(model),
            )

    def offload_all(self, *, exclude: set[ModelCacheKey] | None = None) -> None:
        """Move cached models to CPU, retaining the excluded entries on their device."""
        excluded = exclude or set()
        with self._lock:
            evicted: list[ModelCacheKey] = []
            for key, entry in self._entries.items():
                if key in excluded:
                    continue
                if entry.offloadable:
                    self._set_status(
                        key,
                        "loading",
                        device=self._model_device(entry.model),
                        detail="Moving to CPU",
                        offloadable=True,
                        memory_bytes=self._model_memory(entry.model),
                    )
                    self.move_model(entry.model, torch.device("cpu"))
                    self._set_status(
                        key,
                        "loaded",
                        device=self._model_device(entry.model),
                        detail="Ready",
                        offloadable=True,
                        memory_bytes=self._model_memory(entry.model),
                    )
                else:
                    logger.info("Evicting non-offloadable warm model: %s", key.label)
                    evicted.append(key)

            for key in evicted:
                del self._entries[key]
                self._set_status(key, "unloaded", detail="Evicted", offloadable=False)

        self._free_device_cache()

    def offload(self, key: ModelCacheKey) -> None:
        """Move one cached model to CPU, or evict it when it is not movable."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return
            if entry.offloadable:
                self._set_status(
                    key,
                    "loading",
                    device=self._model_device(entry.model),
                    detail="Moving to CPU",
                    offloadable=True,
                    memory_bytes=self._model_memory(entry.model),
                )
                self.move_model(entry.model, torch.device("cpu"))
                self._set_status(
                    key,
                    "loaded",
                    device=self._model_device(entry.model),
                    detail="Ready",
                    offloadable=True,
                    memory_bytes=self._model_memory(entry.model),
                )
            else:
                del self._entries[key]
                self._set_status(key, "unloaded", detail="Evicted", offloadable=False)
        self._free_device_cache()

    def clear(self) -> None:
        """Destroy all cached models and release allocator caches."""
        with self._lock:
            self._entries.clear()
            self._artifacts.clear()
            for key, status in tuple(self._statuses.items()):
                self._set_status(
                    key,
                    "unloaded",
                    detail="Released",
                    offloadable=status.offloadable,
                )
        self._free_device_cache()

    def get_or_create_artifact(self, key: Hashable, factory: Callable[[], ModelT]) -> ModelT:
        """Reuse a CPU-side immutable artifact such as a discovered dataset index."""
        with self._lock:
            if key not in self._artifacts:
                self._artifacts[key] = factory()
            return self._artifacts[key]

    def get_artifact(self, key: Hashable) -> object | None:
        with self._lock:
            return self._artifacts.get(key)

    def put_artifact(self, key: Hashable, artifact: object) -> None:
        with self._lock:
            self._artifacts[key] = artifact

    def clear_artifacts(self, prefix: str | None = None) -> None:
        """Invalidate cached artifacts, optionally restricting tuple keys by prefix."""
        with self._lock:
            if prefix is None:
                self._artifacts.clear()
                return
            stale = [key for key in self._artifacts if isinstance(key, tuple) and key and key[0] == prefix]
            for key in stale:
                del self._artifacts[key]

    def contains(self, key: ModelCacheKey) -> bool:
        with self._lock:
            return key in self._entries

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._entries)

    @property
    def statuses(self) -> tuple[ModelStatus, ...]:
        """Return the latest state of every model encountered by this pool."""
        if not self._lock.acquire(blocking=False):
            return self._status_snapshot
        try:
            current: list[ModelStatus] = []
            for key, status in self._statuses.items():
                entry = self._entries.get(key)
                if entry is None:
                    current.append(status)
                    continue
                current.append(
                    replace(
                        status,
                        device=self._model_device(entry.model),
                        memory_bytes=self._model_memory(entry.model),
                    )
                )
            self._status_snapshot = tuple(current)
            return self._status_snapshot
        finally:
            self._lock.release()

    def _set_status(
        self,
        key: ModelCacheKey,
        status: str,
        *,
        device: str | None = None,
        detail: str | None = None,
        error: str | None = None,
        offloadable: bool = True,
        memory_bytes: tuple[tuple[str, int], ...] = (),
    ) -> None:
        snapshot = ModelStatus(
            key=key,
            status=status,
            device=device,
            detail=detail,
            error=error,
            offloadable=offloadable,
            memory_bytes=memory_bytes,
        )
        self._statuses[key] = snapshot
        self._status_snapshot = tuple(self._statuses.values())
        if self._status_listener is not None:
            self._status_listener(snapshot)

    @staticmethod
    def _model_device(model: object) -> str | None:
        if isinstance(model, torch.nn.Module):
            devices = {str(tensor.device) for tensor in chain(model.parameters(), model.buffers())}
            if devices:
                return " + ".join(sorted(devices, key=lambda value: value == "meta"))

        device = getattr(model, "device", None)
        if device is not None:
            return str(device)
        return None

    @staticmethod
    def _model_memory(model: object) -> tuple[tuple[str, int], ...]:
        if not isinstance(model, torch.nn.Module):
            return ()
        totals: dict[str, int] = {}
        seen: set[int] = set()
        for tensor in chain(model.parameters(), model.buffers()):
            if tensor.device.type == "meta" or id(tensor) in seen:
                continue
            seen.add(id(tensor))
            device = str(tensor.device)
            totals[device] = totals.get(device, 0) + tensor.numel() * tensor.element_size()
        return tuple(sorted(totals.items()))

    @staticmethod
    def move_model(model: object, device: torch.device | str) -> None:
        """Move real tensors while retaining any intentionally empty meta placeholders."""
        target = torch.device(device)
        if isinstance(model, torch.nn.Module) and any(
            tensor.device.type == "meta" for tensor in chain(model.parameters(), model.buffers())
        ):
            # Component loaders may deliberately leave unused submodules as meta
            # placeholders. ``Module.to`` attempts to copy those nonexistent values
            # and raises. Keep placeholders on meta while moving every real tensor.
            model._apply(lambda tensor: tensor if tensor.device.type == "meta" else tensor.to(target))
            return
        move = getattr(model, "to", None)
        if move is None:
            return
        move(target)

    @staticmethod
    def _free_device_cache() -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __enter__(self) -> WarmModelPool:
        return self

    def __exit__(self, *_args: object) -> None:
        self.clear()
