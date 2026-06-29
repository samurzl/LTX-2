"""Live terminal dashboard for the persistent warm-model server."""

from __future__ import annotations

import resource
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable, ClassVar

import psutil
import torch
from rich import box
from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

from ltx_trainer.model_pool import ModelStatus


@dataclass(frozen=True)
class MemorySegment:
    label: str
    value: int
    style: str


@dataclass(frozen=True)
class MemoryUsage:
    name: str
    total: int
    used: int
    segments: tuple[MemorySegment, ...]
    detail: str
    peak_hint: int = 0


ResourceProvider = Callable[[list[dict[str, object]]], tuple[MemoryUsage, ...]]
ModelStatusProvider = Callable[[], tuple[ModelStatus, ...]]


class WarmServerState:
    """Thread-safe state shared by the model pool and terminal dashboard."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._started_at = time.time()
        self._models: dict[str, dict[str, object]] = {}
        self._active_job: str | None = None
        self._job_started_at: float | None = None
        self._completed_jobs = 0
        self._last_error: str | None = None

    def update_model(self, status: ModelStatus) -> None:
        model = status.as_dict()
        with self._lock:
            self._models[str(model["id"])] = model

    def begin_job(self, command: str) -> None:
        with self._lock:
            self._active_job = command
            self._job_started_at = time.time()
            self._last_error = None

    def finish_job(self, error: str | None = None) -> None:
        with self._lock:
            self._active_job = None
            self._job_started_at = None
            self._completed_jobs += 1
            self._last_error = error

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            models = sorted(
                (dict(model) for model in self._models.values()),
                key=lambda model: (
                    {"loading": 0, "loaded": 1, "unloaded": 2}.get(str(model["status"]), 3),
                    str(model["component"]),
                    str(model["name"]),
                ),
            )
            counts = {
                state: sum(model["status"] == state for model in models) for state in ("loaded", "loading", "unloaded")
            }
            return {
                "started_at": self._started_at,
                "active_job": self._active_job,
                "job_started_at": self._job_started_at,
                "completed_jobs": self._completed_jobs,
                "last_error": self._last_error,
                "counts": counts,
                "models": models,
            }


class WarmConsoleDashboard:
    """Render warm-model state as a live Rich display in the server terminal."""

    _STATUS_STYLES: ClassVar[dict[str, str]] = {
        "loaded": "bold bright_green",
        "loading": "bold orange1",
        "unloaded": "bold bright_red",
    }
    _STATUS_COLORS: ClassVar[dict[str, str]] = {
        "loaded": "green",
        "loading": "orange1",
        "unloaded": "red",
    }

    def __init__(
        self,
        state: WarmServerState,
        console: Console,
        resource_provider: ResourceProvider | None = None,
        model_status_provider: ModelStatusProvider | None = None,
    ) -> None:
        self._state = state
        self._console = console
        self._resource_provider = resource_provider or _collect_memory_usage
        self._model_status_provider = model_status_provider
        self._resource_snapshot: tuple[MemoryUsage, ...] = ()
        self._resource_sampled_at = 0.0
        self._model_status_sampled_at = 0.0
        self._resource_peaks: dict[str, int] = {}
        self._live: Live | None = None

    def start(self) -> None:
        self._live = Live(
            console=self._console,
            get_renderable=self.render,
            refresh_per_second=8,
            redirect_stdout=False,
            redirect_stderr=False,
            vertical_overflow="visible",
        )
        self._live.start(refresh=True)

    def stop(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None

    def update_model(self, status: ModelStatus) -> None:
        self._state.update_model(status)
        self._resource_sampled_at = 0.0
        self._model_status_sampled_at = 0.0
        self.refresh()

    def begin_job(self, command: str) -> None:
        self._state.begin_job(command)
        self.refresh()

    def finish_job(self, error: str | None = None) -> None:
        self._state.finish_job(error)
        self.refresh()

    def refresh(self) -> None:
        if self._live is not None:
            self._live.update(self.render(), refresh=True)

    def render(self) -> RenderableType:
        now = time.monotonic()
        if self._model_status_provider is not None and now - self._model_status_sampled_at >= 1.0:
            for status in self._model_status_provider():
                self._state.update_model(status)
            self._model_status_sampled_at = now
        snapshot = self._state.snapshot()
        header = self._render_header(snapshot)
        resources = self._render_resources(snapshot["models"])
        models = self._render_models(snapshot["models"])
        contents: list[RenderableType] = [header, resources, models]
        if snapshot["last_error"]:
            contents.append(
                Panel(
                    Text(f"Last job failed: {snapshot['last_error']}", style="bright_red"),
                    border_style="red",
                    padding=(0, 1),
                )
            )
        return Panel(
            Group(*contents),
            title="[bold bright_green]LTX WARM MODEL SERVER[/]",
            subtitle="[dim]client output → invoking terminal[/]",
            border_style="green4",
            box=box.ROUNDED,
            padding=(1, 2),
        )

    def _render_header(self, snapshot: dict[str, object]) -> Table:
        counts = snapshot["counts"]
        assert isinstance(counts, dict)
        active_job = snapshot["active_job"]
        job_started_at = snapshot["job_started_at"]
        if active_job and isinstance(job_started_at, float):
            job_age = _format_duration(time.time() - job_started_at)
            job = Text.assemble(("● ", "bright_green"), (f"RUNNING {str(active_job).upper()}", "bold"), f"  {job_age}")
        else:
            job = Text.assemble(("● ", "green"), ("READY", "bold"))

        summary = Text.assemble(
            (f"{counts['loaded']} loaded", "bold bright_green"),
            "   ",
            (f"{counts['loading']} loading", "bold orange1"),
            "   ",
            (f"{counts['unloaded']} unloaded", "bold bright_red"),
            "   ",
            (f"{snapshot['completed_jobs']} jobs", "dim"),
        )
        uptime = Text(f"up {_format_duration(time.time() - float(snapshot['started_at']))}", style="dim")

        table = Table.grid(expand=True, padding=(0, 1))
        if self._console.width < 110:
            table.add_column(ratio=3)
            table.add_column(justify="right", ratio=1)
            table.add_row(job, uptime)
            table.add_row(summary, "")
        else:
            table.add_column(ratio=2)
            table.add_column(justify="center", ratio=3)
            table.add_column(justify="right", ratio=1)
            table.add_row(job, summary, uptime)
        return table

    def _render_resources(self, models_value: object) -> Panel:
        models = [model for model in models_value if isinstance(model, dict)] if isinstance(models_value, list) else []
        now = time.monotonic()
        if now - self._resource_sampled_at >= 1.0 or not self._resource_snapshot:
            self._resource_snapshot = self._resource_provider(models)
            self._resource_sampled_at = now

        meters: list[RenderableType] = []
        for usage in self._resource_snapshot:
            peak = max(self._resource_peaks.get(usage.name, 0), usage.used, usage.peak_hint)
            self._resource_peaks[usage.name] = peak
            meters.append(self._render_memory_meter(usage, peak))
        return Panel(
            Group(*meters),
            title="[bold]MEMORY[/]",
            border_style="grey37",
            padding=(0, 1),
        )

    def _render_memory_meter(self, usage: MemoryUsage, peak: int) -> RenderableType:
        if usage.total <= 0:
            return Group(Text(usage.name, style="bold"), Text(usage.detail, style="dim"))

        percent = usage.used / usage.total * 100
        heading = Text.assemble(
            (usage.name, "bold"),
            f"  {_format_bytes(usage.used)} / {_format_bytes(usage.total)}  ",
            (f"{percent:.0f}%", "bold"),
            "   ",
            (f"peak {_format_bytes(peak)}", "bold bright_red"),
        )
        bar_width = min(48, max(20, self._console.width - 30))
        bar = _segmented_bar(usage.segments, usage.total, bar_width, peak)
        legend = Text()
        for index, segment in enumerate(usage.segments):
            if index:
                legend.append("   ")
            legend.append("■ ", style=segment.style)
            legend.append(f"{segment.label} {_format_bytes(segment.value)}", style="dim")
        legend.append("   ")
        legend.append("│ ", style="bold bright_red")
        legend.append(f"peak {_format_bytes(peak)}", style="dim")
        return Group(heading, bar, legend, Text(usage.detail, style="dim"))

    def _render_models(self, models_value: object) -> RenderableType:
        if not isinstance(models_value, list) or not models_value:
            return Panel(
                "[dim]No models requested yet. Start preprocessing or training and they will appear here.[/]",
                title="[bold]MODEL INVENTORY[/]",
                border_style="grey37",
                padding=(1, 2),
            )
        models = [model for model in models_value if isinstance(model, dict)]
        if self._console.width < 110:
            cards = [Text("MODEL INVENTORY", style="bold")]
            cards.extend(self._render_model_card(model) for model in models)
            return Group(*cards)

        table = Table(
            box=box.SIMPLE_HEAD,
            expand=True,
            show_edge=False,
            border_style="grey23",
            header_style="bold grey70",
            padding=(0, 1),
        )
        table.add_column("STATE", width=16, no_wrap=True)
        table.add_column("MODEL", ratio=4)
        table.add_column("INFORMATION", ratio=3)
        table.add_column("DEVICE", ratio=1, no_wrap=True)

        for model in models:
            status = str(model["status"])
            table.add_row(
                self._render_status(status, str(model.get("detail") or model.get("error") or "")),
                self._render_identity(model, status),
                self._render_information(model),
                Text(str(model.get("device") or "—"), style="cyan" if model.get("device") else "dim"),
            )
        return Panel(table, title="[bold]MODEL INVENTORY[/]", border_style="grey37", padding=(0, 1))

    def _render_model_card(self, model: dict[str, object]) -> Panel:
        status = str(model["status"])
        detail = str(model.get("detail") or model.get("error") or "")
        device = Text.assemble(("device  ", "dim"), (str(model.get("device") or "—"), "cyan"))
        content = Table.grid(expand=True, padding=(0, 1))
        content.add_column(width=16, no_wrap=True)
        content.add_column(ratio=1)
        content.add_row(
            self._render_status(status, ""),
            Group(
                self._render_identity(model, status),
                self._render_information(model),
                device,
                Text(detail, style="dim"),
            ),
        )
        component = str(model["component"]).replace("_", " ").upper()
        return Panel(
            content,
            title=f"[{self._STATUS_STYLES.get(status, 'bold')}]{component}[/]",
            border_style=self._STATUS_COLORS.get(status, "white"),
            padding=(0, 1),
        )

    def _render_status(self, status: str, detail: str) -> RenderableType:
        style = self._STATUS_STYLES.get(status, "bold white")
        label = Text(status.upper(), style=style)
        if status == "loading":
            bar: RenderableType = ProgressBar(
                width=12,
                pulse=True,
                style="grey23",
                complete_style="bright_green",
                finished_style="bright_green",
                pulse_style="bright_green",
            )
        else:
            bar = Text("━" * 12, style=style)
        return Group(label, bar, Text(detail, style="dim", overflow="ellipsis", no_wrap=True))

    def _render_identity(self, model: dict[str, object], status: str) -> Text:
        style = self._STATUS_STYLES.get(status, "bold white")
        component = str(model["component"]).replace("_", " ").upper()
        identity = Text(component, style=style)
        identity.append(f"\n{model['name']}", style="bold")
        identity.append(f"\n{model['path']}", style="dim")
        return identity

    @staticmethod
    def _render_information(model: dict[str, object]) -> Text:
        info = Text(f"checkpoint {_format_bytes(int(model['size_bytes']))}", style="bold")
        info.append(f"  ·  {str(model['dtype']).removeprefix('torch.')}", style="cyan")
        memory_bytes = model.get("memory_bytes")
        if isinstance(memory_bytes, dict) and memory_bytes:
            resident = sum(int(value) for value in memory_bytes.values())
            locations = ", ".join(f"{device} {_format_bytes(int(value))}" for device, value in memory_bytes.items())
            info.append(f"\nresident {_format_bytes(resident)}", style="bright_green")
            info.append(f"  ({locations})", style="dim")
        options = model.get("options")
        if isinstance(options, dict):
            for key, value in options.items():
                info.append(f"\n{key}: ", style="dim")
                info.append(str(value), style="white")
        info.append("\noffloadable" if model.get("offloadable") else "\nevicted when inactive", style="dim")
        return info


def _collect_memory_usage(models: list[dict[str, object]]) -> tuple[MemoryUsage, ...]:
    return _collect_vram_usage(models), _collect_ram_usage(models)


def _collect_vram_usage(models: list[dict[str, object]]) -> MemoryUsage:
    if not torch.cuda.is_available():
        return MemoryUsage("VRAM", 0, 0, (), "CUDA is not available")

    try:
        device = torch.cuda.current_device()
        free, total = torch.cuda.mem_get_info(device)
        used = max(0, total - free)
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
        cached_models = min(_model_memory_on(models, "cuda"), used)
        server_total = min(used, max(reserved, cached_models))
        server_other = max(0, server_total - cached_models)
        other_gpu = max(0, used - server_total)
        available = max(0, total - used)
        peak_hint = min(total, max(used, other_gpu + max(peak_reserved, cached_models)))
        name = torch.cuda.get_device_name(device)
    except (RuntimeError, AssertionError) as error:
        return MemoryUsage("VRAM", 0, 0, (), f"CUDA memory unavailable: {error}")

    return MemoryUsage(
        name="VRAM",
        total=total,
        used=used,
        segments=(
            MemorySegment("cached models", cached_models, "bright_green"),
            MemorySegment("server other", server_other, "bright_cyan"),
            MemorySegment("other GPU use", other_gpu, "orange3"),
            MemorySegment("free", available, "grey23"),
        ),
        detail=(
            f"GPU {device} · {name} · warm server allocated {_format_bytes(allocated)}, "
            f"reserved {_format_bytes(reserved)}"
        ),
        peak_hint=peak_hint,
    )


def _collect_ram_usage(models: list[dict[str, object]]) -> MemoryUsage:
    memory = psutil.virtual_memory()
    total = int(memory.total)
    available = int(memory.available)
    used = max(0, total - available)
    process_rss = int(psutil.Process().memory_info().rss)
    cached_models = min(_model_memory_on(models, "cpu"), used, process_rss)
    server_total = min(used, max(process_rss, cached_models))
    server_other = max(0, server_total - cached_models)
    system_other = max(0, used - server_total)
    peak_rss = _peak_process_rss()
    peak_hint = min(total, max(used, system_other + peak_rss))
    return MemoryUsage(
        name="RAM",
        total=total,
        used=used,
        segments=(
            MemorySegment("cached models", cached_models, "bright_green"),
            MemorySegment("server other", server_other, "bright_cyan"),
            MemorySegment("system other", system_other, "orange3"),
            MemorySegment("available", available, "grey23"),
        ),
        detail=f"warm server RSS {_format_bytes(process_rss)} · process peak {_format_bytes(peak_rss)}",
        peak_hint=peak_hint,
    )


def _model_memory_on(models: list[dict[str, object]], device_prefix: str) -> int:
    total = 0
    for model in models:
        memory_bytes = model.get("memory_bytes")
        if not isinstance(memory_bytes, dict):
            continue
        total += sum(int(value) for device, value in memory_bytes.items() if str(device).startswith(device_prefix))
    return total


def _peak_process_rss() -> int:
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak if sys.platform == "darwin" else peak * 1024


def _segmented_bar(
    segments: tuple[MemorySegment, ...],
    total: int,
    width: int,
    peak: int,
) -> Text:
    cumulative: list[tuple[int, str]] = []
    boundary = 0
    for segment in segments:
        boundary += max(0, segment.value)
        cumulative.append((boundary, segment.style))

    peak_column = min(width - 1, max(0, round(min(peak, total) / total * (width - 1))))
    bar = Text()
    for column in range(width):
        if column == peak_column:
            bar.append("│", style="bold bright_red")
            continue
        position = (column + 0.5) / width * total
        style = cumulative[-1][1] if cumulative else "grey23"
        for segment_boundary, segment_style in cumulative:
            if position <= segment_boundary:
                style = segment_style
                break
        bar.append("█", style=style)
    return bar


def _format_bytes(value: int) -> str:
    size = float(value)
    units = ("B", "KB", "MB", "GB", "TB")
    for unit in units:
        if size < 1024 or unit == units[-1]:
            precision = 1 if unit in {"GB", "TB"} else 0
            return f"{size:.{precision}f} {unit}"
        size /= 1024
    return f"{value} B"


def _format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"
