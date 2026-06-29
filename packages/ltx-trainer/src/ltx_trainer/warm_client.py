"""Client protocol for the local persistent warm-model server."""

from __future__ import annotations

import json
import os
import shutil
import socket
import sys
import tempfile
from pathlib import Path
from typing import Any

from ltx_trainer import logger

PROTOCOL_VERSION = 1
SOCKET_ENV = "LTX_TRAINER_WARM_SOCKET"
DISABLE_ENV = "LTX_TRAINER_DISABLE_WARM_SERVER"


class WarmServerJobError(RuntimeError):
    """A job was accepted by the warm server but failed during execution."""


def default_socket_path() -> Path:
    configured = os.environ.get(SOCKET_ENV)
    if configured:
        return Path(configured).expanduser()
    user_id = os.getuid() if hasattr(os, "getuid") else os.getpid()
    return Path(tempfile.gettempdir()) / f"ltx-trainer-{user_id}.sock"


def warm_server_enabled() -> bool:
    return os.environ.get(DISABLE_ENV, "").lower() not in {"1", "true", "yes"}


def submit_if_running(command: str, args: dict[str, Any]) -> bool:
    """Submit a job when the default warm server is reachable.

    Returns ``False`` without side effects when no server is running, allowing the
    caller to continue through its original in-process code path.
    """
    if not warm_server_enabled():
        return False
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        if os.environ.get("LOCAL_RANK", "0") == "0":
            logger.warning("Warm model server bypassed: distributed Accelerate launches run in their own processes")
        return False
    response = request(command, args, required=False)
    return response is not None


def submit_argv_if_running(command: str, argv: list[str]) -> bool:
    """Lightweight early-entry path used before importing Torch-heavy CLI modules."""
    return submit_if_running(command, {"argv": argv})


def request(command: str, args: dict[str, Any] | None = None, *, required: bool = True) -> dict[str, Any] | None:
    """Send one request and wait for the final response."""
    socket_path = default_socket_path()
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(0.25)
    try:
        client.connect(str(socket_path))
    except OSError as error:
        client.close()
        if required:
            raise ConnectionError(f"Warm model server is not running at {socket_path}") from error
        return None

    client.settimeout(None)
    payload = {
        "protocol": PROTOCOL_VERSION,
        "command": command,
        "args": args or {},
        "cwd": str(Path.cwd()),
        "terminal": {
            "stdout_isatty": sys.stdout.isatty(),
            "stderr_isatty": sys.stderr.isatty(),
            "columns": shutil.get_terminal_size(fallback=(100, 24)).columns,
        },
    }
    try:
        client.sendall((json.dumps(payload, default=str) + "\n").encode())
        with client.makefile("r", encoding="utf-8") as responses:
            for line in responses:
                response = json.loads(line)
                status = response.get("status")
                if status == "accepted":
                    logger.info("Warm model server accepted the %s job", command)
                    continue
                if status == "output":
                    target = sys.stderr if response.get("stream") == "stderr" else sys.stdout
                    target.write(str(response.get("text", "")))
                    target.flush()
                    continue
                if status == "ok":
                    return response
                if status == "error":
                    message = response.get("message", "Warm model server job failed")
                    details = response.get("traceback")
                    if details:
                        message = f"{message}\n\nServer traceback:\n{details}"
                    raise WarmServerJobError(message)
    finally:
        client.close()

    raise ConnectionError("Warm model server closed the connection without a final response")
