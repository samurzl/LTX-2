#!/usr/bin/env python3

"""Persistent local server that owns reusable LTX model components."""

from __future__ import annotations

import contextlib
import importlib
import io
import json
import os
import socket
import sys
import threading
import traceback
from pathlib import Path
from typing import Annotated, Any

# Set cache identity before warm_console imports model_pool (and therefore torch).
os.environ.setdefault(
    "TORCHINDUCTOR_CACHE_DIR",
    str(Path.home() / ".cache" / "ltx-trainer" / "torchinductor"),
)
os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")

import typer
from rich.console import Console

from ltx_trainer import logger
from ltx_trainer.warm_client import PROTOCOL_VERSION, SOCKET_ENV, default_socket_path, request
from ltx_trainer.warm_console import WarmConsoleDashboard, WarmServerState

# Pin this console to the server terminal. Job execution temporarily redirects
# ``sys.stdout``/``sys.stderr`` to the client socket, but the dashboard must stay here.
console = Console(file=sys.stdout)
app = typer.Typer(pretty_exceptions_enable=False, no_args_is_help=True)


class _ClientChannel:
    """Serialize protocol frames sent to one client connection."""

    def __init__(self, connection: socket.socket) -> None:
        self._connection = connection
        self._lock = threading.Lock()

    def send(self, payload: dict[str, Any]) -> None:
        encoded = (json.dumps(payload, default=str) + "\n").encode()
        with self._lock, contextlib.suppress(BrokenPipeError, ConnectionResetError):
            self._connection.sendall(encoded)


class _ClientTextStream(io.TextIOBase):
    """A text stream that emits stdout/stderr frames over the client socket."""

    def __init__(self, channel: _ClientChannel, stream: str, *, isatty: bool) -> None:
        self._channel = channel
        self._stream = stream
        self._isatty = isatty

    @property
    def encoding(self) -> str:
        return "utf-8"

    def writable(self) -> bool:
        return True

    def isatty(self) -> bool:
        return self._isatty

    def write(self, text: str) -> int:
        if text:
            self._channel.send({"status": "output", "stream": self._stream, "text": text})
        return len(text)

    def flush(self) -> None:
        return


class WarmModelServer:
    def __init__(self, socket_path: Path) -> None:
        from ltx_trainer.model_pool import WarmModelPool  # noqa: PLC0415

        self._socket_path = socket_path
        self._state = WarmServerState()
        self._dashboard = WarmConsoleDashboard(
            self._state,
            console,
            model_status_provider=lambda: self._model_pool.statuses,
        )
        self._model_pool = WarmModelPool(status_listener=self._dashboard.update_model)
        self._mixed_precision_mode: str | None = None
        self._stop_requested = False

    def serve(self) -> None:
        self._prepare_socket_path()
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            server.bind(str(self._socket_path))
            self._socket_path.chmod(0o600)
            server.listen(8)
            console.print(f"[green]Warm model server listening on {self._socket_path}[/]")
            console.print("Existing process_dataset.py and train.py commands will now use this server automatically.")
            self._dashboard.start()
            while not self._stop_requested:
                connection, _ = server.accept()
                with connection:
                    self._handle_connection(connection)
        finally:
            server.close()
            self._model_pool.clear()
            self._dashboard.stop()
            self._socket_path.unlink(missing_ok=True)
            console.print("[yellow]Warm model server stopped.[/]")

    def _prepare_socket_path(self) -> None:
        self._socket_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._socket_path.exists():
            return
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        probe.settimeout(0.2)
        try:
            probe.connect(str(self._socket_path))
        except OSError:
            self._socket_path.unlink(missing_ok=True)
        else:
            raise RuntimeError(f"A warm model server is already running at {self._socket_path}")
        finally:
            probe.close()

    def _handle_connection(self, connection: socket.socket) -> None:
        channel = _ClientChannel(connection)
        with connection.makefile("r", encoding="utf-8") as request_file:
            line = request_file.readline()
        if not line:
            return

        try:
            payload = json.loads(line)
            if payload.get("protocol") != PROTOCOL_VERSION:
                raise ValueError("Unsupported warm-server protocol version")
            command = payload.get("command")
            args = payload.get("args", {})
            if not isinstance(args, dict):
                raise ValueError("Request args must be a mapping")

            if command == "ping":
                channel.send({"status": "ok", "cache_entries": self._model_pool.size})
                return
            if command == "shutdown":
                self._stop_requested = True
                channel.send({"status": "ok"})
                return
            if command not in {
                "captions",
                "captions_cli",
                "latents",
                "latents_cli",
                "preprocess",
                "preprocess_cli",
                "train",
                "train_cli",
            }:
                raise ValueError(f"Unknown warm-server command: {command!r}")

            channel.send({"status": "accepted"})
            self._run_client_job(channel, payload, command, args)
        except Exception as error:
            logger.exception("Warm server job failed")
            channel.send(
                {
                    "status": "error",
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                },
            )

    def _run_client_job(
        self,
        channel: _ClientChannel,
        payload: dict[str, Any],
        command: str,
        args: dict[str, Any],
    ) -> None:
        terminal = payload.get("terminal", {})
        if not isinstance(terminal, dict):
            terminal = {}
        stdout = _ClientTextStream(channel, "stdout", isatty=bool(terminal.get("stdout_isatty")))
        stderr = _ClientTextStream(channel, "stderr", isatty=bool(terminal.get("stderr_isatty")))
        columns = terminal.get("columns")
        previous_columns = os.environ.get("COLUMNS")
        if isinstance(columns, int) and columns > 0:
            os.environ["COLUMNS"] = str(columns)

        self._dashboard.begin_job(command.removesuffix("_cli"))
        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                try:
                    request_cwd = Path(payload.get("cwd", Path.cwd())).expanduser()
                    if not request_cwd.is_dir():
                        raise ValueError(f"Client working directory does not exist: {request_cwd}")
                    previous_cwd = Path.cwd()
                    try:
                        os.chdir(request_cwd)
                        result = self._run_job(command, args)
                    finally:
                        os.chdir(previous_cwd)
                except Exception as error:
                    details = traceback.format_exc()
                    logger.exception("Warm server job failed")
                    self._dashboard.finish_job(str(error))
                    channel.send(
                        {
                            "status": "error",
                            "message": str(error),
                            "traceback": details,
                        }
                    )
                    return

            self._dashboard.finish_job()
            channel.send({"status": "ok", "result": result})
        finally:
            if previous_columns is None:
                os.environ.pop("COLUMNS", None)
            else:
                os.environ["COLUMNS"] = previous_columns

    def _run_job(self, command: str, args: dict[str, Any]) -> dict[str, Any]:
        args.pop("model_pool", None)
        if command.endswith("_cli"):
            captured_command, captured_args = self._capture_cli_job(command, args)
            return self._run_job(captured_command, captured_args)
        if command == "captions":
            from process_captions import compute_captions_embeddings  # noqa: PLC0415

            try:
                compute_captions_embeddings(**args, model_pool=self._model_pool)
            finally:
                self._model_pool.clear_artifacts("precomputed_dataset")
            return {}
        if command == "latents":
            from process_videos import compute_latents  # noqa: PLC0415

            buckets = args.get("resolution_buckets")
            if isinstance(buckets, list):
                args["resolution_buckets"] = [tuple(bucket) for bucket in buckets]
            try:
                compute_latents(**args, model_pool=self._model_pool)
            finally:
                self._model_pool.clear_artifacts("precomputed_dataset")
            return {}
        if command == "preprocess":
            from process_dataset import preprocess_dataset  # noqa: PLC0415

            buckets = args.get("resolution_buckets")
            if isinstance(buckets, list):
                args["resolution_buckets"] = [tuple(bucket) for bucket in buckets]
            try:
                preprocess_dataset(**args, model_pool=self._model_pool)
            finally:
                self._model_pool.clear_artifacts("precomputed_dataset")
            return {}
        return self._run_training(args)

    @staticmethod
    def _capture_cli_job(command: str, args: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        from typer.main import get_command  # noqa: PLC0415

        module_name = {
            "captions_cli": "process_captions",
            "latents_cli": "process_videos",
            "preprocess_cli": "process_dataset",
            "train_cli": "train",
        }[command]
        module = importlib.import_module(module_name)
        captured: list[tuple[str, dict[str, Any]]] = []
        original_submit = module.submit_if_running
        module.submit_if_running = lambda job_command, job_args: captured.append((job_command, job_args)) or True
        try:
            click_command = get_command(module.app)
            click_command.main(
                args=list(args.get("argv", [])),
                prog_name=f"{module_name}.py",
                standalone_mode=False,
            )
        finally:
            module.submit_if_running = original_submit
        if len(captured) != 1:
            raise RuntimeError(f"Could not capture parsed arguments for {module_name}.py")
        return captured[0]

    def _run_training(self, args: dict[str, Any]) -> dict[str, Any]:
        import gc  # noqa: PLC0415

        import torch  # noqa: PLC0415
        import yaml  # noqa: PLC0415

        from ltx_trainer.config import LtxTrainerConfig  # noqa: PLC0415
        from ltx_trainer.trainer import LtxvTrainer  # noqa: PLC0415

        config_path = Path(args["config_path"]).expanduser()
        with config_path.open("r", encoding="utf-8") as file:
            config = LtxTrainerConfig(**yaml.safe_load(file))

        mixed_precision = config.acceleration.mixed_precision_mode
        if self._mixed_precision_mode is None:
            self._mixed_precision_mode = mixed_precision
        elif self._mixed_precision_mode != mixed_precision:
            raise ValueError(
                "The warm server was initialized by a training job using "
                f"mixed_precision_mode={self._mixed_precision_mode!r}; it cannot run a job using {mixed_precision!r}. "
                "Restart the warm server to change Accelerate process-wide settings."
            )

        preflight = LtxvTrainer.preflight_config(config, model_pool=self._model_pool)
        use_warm_pool = config.model.training_mode == "lora"
        if not use_warm_pool:
            logger.warning("Full fine-tuning mutates base weights and will use isolated transformer loading")
            self._model_pool.offload_all()

        trainer = LtxvTrainer(
            config,
            preflight_result=preflight,
            model_pool=self._model_pool if use_warm_pool else None,
        )
        try:
            saved_path, stats = trainer.train(
                disable_progress_bars=bool(args.get("disable_progress_bars", False)),
            )
            return {"saved_path": str(saved_path), "stats": stats.model_dump()}
        finally:
            trainer.release_warm_models()
            del trainer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def _set_socket_path(socket_path: Path | None) -> Path:
    if socket_path is not None:
        resolved = socket_path.expanduser().resolve()
        os.environ[SOCKET_ENV] = str(resolved)
        return resolved
    return default_socket_path()


def _configure_compile_cache() -> None:
    cache_dir = Path(
        os.environ.get("TORCHINDUCTOR_CACHE_DIR", Path.home() / ".cache" / "ltx-trainer" / "torchinductor")
    ).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(cache_dir))
    os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")


@app.command()
def serve(
    socket_path: Annotated[Path | None, typer.Option("--socket", help="Override the local Unix socket path.")] = None,
) -> None:
    """Run the warm model server in the foreground until stopped."""
    _configure_compile_cache()
    server = WarmModelServer(_set_socket_path(socket_path))
    with contextlib.suppress(KeyboardInterrupt):
        server.serve()


@app.command()
def status(
    socket_path: Annotated[Path | None, typer.Option("--socket", help="Override the local Unix socket path.")] = None,
) -> None:
    """Show whether a warm model server is running."""
    path = _set_socket_path(socket_path)
    response = request("ping", required=False)
    if response is None:
        console.print(f"[yellow]No warm model server is running at {path}.[/]")
        raise typer.Exit(code=1)
    console.print(f"[green]Warm model server is running at {path} ({response['cache_entries']} cached models).[/]")


@app.command()
def stop(
    socket_path: Annotated[Path | None, typer.Option("--socket", help="Override the local Unix socket path.")] = None,
) -> None:
    """Ask the warm model server to release its models and stop."""
    path = _set_socket_path(socket_path)
    response = request("shutdown", required=False)
    if response is None:
        console.print(f"[yellow]No warm model server is running at {path}.[/]")
        raise typer.Exit(code=1)
    console.print("[green]Warm model server stopped.[/]")


if __name__ == "__main__":
    app()
