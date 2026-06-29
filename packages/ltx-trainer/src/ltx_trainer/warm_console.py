"""Live terminal dashboard for the persistent warm-model server."""

from __future__ import annotations

import threading
import time
from typing import ClassVar

from rich import box
from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

from ltx_trainer.model_pool import ModelStatus


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

    def __init__(self, state: WarmServerState, console: Console) -> None:
        self._state = state
        self._console = console
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
        snapshot = self._state.snapshot()
        header = self._render_header(snapshot)
        models = self._render_models(snapshot["models"])
        contents: list[RenderableType] = [header, models]
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
        info = Text(_format_bytes(int(model["size_bytes"])), style="bold")
        info.append(f"  ·  {str(model['dtype']).removeprefix('torch.')}", style="cyan")
        options = model.get("options")
        if isinstance(options, dict):
            for key, value in options.items():
                info.append(f"\n{key}: ", style="dim")
                info.append(str(value), style="white")
        info.append("\noffloadable" if model.get("offloadable") else "\nevicted when inactive", style="dim")
        return info


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
