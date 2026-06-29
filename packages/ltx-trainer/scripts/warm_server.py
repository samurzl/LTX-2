#!/usr/bin/env python3

"""Persistent local server that owns reusable LTX model components."""

from __future__ import annotations

import contextlib
import json
import os
import socket
import traceback
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console

from ltx_trainer import logger
from ltx_trainer.warm_client import PROTOCOL_VERSION, SOCKET_ENV, default_socket_path, request

console = Console()
app = typer.Typer(pretty_exceptions_enable=False, no_args_is_help=True)


class WarmModelServer:
    def __init__(self, socket_path: Path) -> None:
        from ltx_trainer.model_pool import WarmModelPool  # noqa: PLC0415

        self._socket_path = socket_path
        self._model_pool = WarmModelPool()
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
            while not self._stop_requested:
                connection, _ = server.accept()
                with connection:
                    self._handle_connection(connection)
        finally:
            server.close()
            self._model_pool.clear()
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
                self._send(connection, {"status": "ok", "cache_entries": self._model_pool.size})
                return
            if command == "shutdown":
                self._stop_requested = True
                self._send(connection, {"status": "ok"})
                return
            if command not in {"captions", "latents", "preprocess", "train"}:
                raise ValueError(f"Unknown warm-server command: {command!r}")

            self._send(connection, {"status": "accepted"})
            result = self._run_job(command, args)
            self._send(connection, {"status": "ok", "result": result})
        except Exception as error:
            logger.exception("Warm server job failed")
            self._send(
                connection,
                {
                    "status": "error",
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                },
            )

    def _run_job(self, command: str, args: dict[str, Any]) -> dict[str, Any]:
        args.pop("model_pool", None)
        if command == "captions":
            from process_captions import compute_captions_embeddings  # noqa: PLC0415

            compute_captions_embeddings(**args, model_pool=self._model_pool)
            return {}
        if command == "latents":
            from process_videos import compute_latents  # noqa: PLC0415

            buckets = args.get("resolution_buckets")
            if isinstance(buckets, list):
                args["resolution_buckets"] = [tuple(bucket) for bucket in buckets]
            compute_latents(**args, model_pool=self._model_pool)
            return {}
        if command == "preprocess":
            from process_dataset import preprocess_dataset  # noqa: PLC0415

            buckets = args.get("resolution_buckets")
            if isinstance(buckets, list):
                args["resolution_buckets"] = [tuple(bucket) for bucket in buckets]
            preprocess_dataset(**args, model_pool=self._model_pool)
            return {}
        return self._run_training(args)

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

        preflight = LtxvTrainer.preflight_config(config)
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

    @staticmethod
    def _send(connection: socket.socket, payload: dict[str, Any]) -> None:
        with contextlib.suppress(BrokenPipeError, ConnectionResetError):
            connection.sendall((json.dumps(payload, default=str) + "\n").encode())


def _set_socket_path(socket_path: Path | None) -> Path:
    if socket_path is not None:
        resolved = socket_path.expanduser().resolve()
        os.environ[SOCKET_ENV] = str(resolved)
        return resolved
    return default_socket_path()


@app.command()
def serve(
    socket_path: Annotated[Path | None, typer.Option("--socket", help="Override the local Unix socket path.")] = None,
) -> None:
    """Run the warm model server in the foreground until stopped."""
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
