from __future__ import annotations

import json
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Sequence
from typing import Any

from safetensors.torch import load_file

DEFAULT_COMFY_SERVER = "http://127.0.0.1:8188"
DEFAULT_COMFY_SAMPLER = "euler_cfg_pp"
SAVE_LATENT_NODE_ID = "15"


def build_ltxv_negative_workflow(  # noqa: PLR0913
    *,
    checkpoint_name: str,
    text_encoder_name: str,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    frames: int,
    frame_rate: float,
    seed: int,
    guidance_scale: float,
    distilled_lora_name: str | None,
    distilled_lora_strength: float,
    sampler_name: str = DEFAULT_COMFY_SAMPLER,
    filename_prefix: str = "ltx_trainer_negatives/negative",
    sigmas: Sequence[float] | None = None,
    num_inference_steps: int = 8,
) -> dict[str, dict[str, Any]]:
    """Build a latent-only ComfyUI LTXV workflow for synthetic negatives."""
    base_model_ref: list[Any] = ["1", 0]
    workflow: dict[str, dict[str, Any]] = {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": checkpoint_name},
        },
        "2": {
            "class_type": "LTXAVTextEncoderLoader",
            "inputs": {
                "text_encoder": text_encoder_name,
                "ckpt_name": checkpoint_name,
                "device": "default",
            },
        },
    }

    if distilled_lora_name and distilled_lora_strength != 0:
        workflow["3"] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": base_model_ref,
                "lora_name": distilled_lora_name,
                "strength_model": distilled_lora_strength,
            },
        }
        model_ref: list[Any] = ["3", 0]
    else:
        model_ref = base_model_ref

    workflow.update(
        {
            "4": {
                "class_type": "CLIPTextEncode",
                "inputs": {"clip": ["2", 0], "text": prompt},
            },
            "5": {
                "class_type": "CLIPTextEncode",
                "inputs": {"clip": ["2", 0], "text": negative_prompt},
            },
            "6": {
                "class_type": "LTXVConditioning",
                "inputs": {
                    "positive": ["4", 0],
                    "negative": ["5", 0],
                    "frame_rate": frame_rate,
                },
            },
            "7": {
                "class_type": "EmptyLTXVLatentVideo",
                "inputs": {
                    "width": width,
                    "height": height,
                    "length": frames,
                    "batch_size": 1,
                },
            },
            "8": {
                "class_type": "LTXVCropGuides",
                "inputs": {
                    "positive": ["6", 0],
                    "negative": ["6", 1],
                    "latent": ["7", 0],
                },
            },
            "9": {
                "class_type": "ModelSamplingLTXV",
                "inputs": {
                    "model": model_ref,
                    "max_shift": 2.05,
                    "base_shift": 0.95,
                    "latent": ["8", 2],
                },
            },
            "10": {
                "class_type": "CFGGuider",
                "inputs": {
                    "model": ["9", 0],
                    "positive": ["8", 0],
                    "negative": ["8", 1],
                    "cfg": guidance_scale,
                },
            },
            "11": {
                "class_type": "RandomNoise",
                "inputs": {"noise_seed": seed},
            },
            "12": {
                "class_type": "KSamplerSelect",
                "inputs": {"sampler_name": sampler_name},
            },
            "14": {
                "class_type": "SamplerCustomAdvanced",
                "inputs": {
                    "noise": ["11", 0],
                    "guider": ["10", 0],
                    "sampler": ["12", 0],
                    "sigmas": ["13", 0],
                    "latent_image": ["8", 2],
                },
            },
            SAVE_LATENT_NODE_ID: {
                "class_type": "SaveLatent",
                "inputs": {
                    "samples": ["14", 0],
                    "filename_prefix": filename_prefix,
                },
            },
        }
    )

    if sigmas is None:
        workflow["13"] = {
            "class_type": "LTXVScheduler",
            "inputs": {
                "steps": num_inference_steps,
                "max_shift": 2.05,
                "base_shift": 0.95,
                "stretch": True,
                "terminal": 0.1,
                "latent": ["8", 2],
            },
        }
    else:
        workflow["13"] = {
            "class_type": "ManualSigmas",
            "inputs": {"sigmas": ", ".join(f"{sigma:g}" for sigma in sigmas)},
        }

    return workflow


def run_comfy_workflow_and_fetch_latent(
    workflow: dict[str, dict[str, Any]],
    *,
    server_url: str = DEFAULT_COMFY_SERVER,
    timeout_seconds: float = 1800.0,
    poll_interval_seconds: float = 1.0,
    save_node_id: str = SAVE_LATENT_NODE_ID,
) -> bytes:
    """Queue *workflow* in ComfyUI and return the saved ``.latent`` bytes."""
    server_url = _normalize_server_url(server_url)
    prompt_id = str(uuid.uuid4())
    _request_json(
        f"{server_url}/prompt",
        data={"prompt": workflow, "prompt_id": prompt_id, "client_id": str(uuid.uuid4())},
    )

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        history = _request_json(f"{server_url}/history/{prompt_id}", data=None)
        if prompt_id in history:
            return _fetch_saved_latent(
                history[prompt_id],
                server_url=server_url,
                save_node_id=save_node_id,
            )
        time.sleep(poll_interval_seconds)

    raise TimeoutError(f"Timed out waiting for ComfyUI prompt {prompt_id} after {timeout_seconds:.0f}s")


def comfy_latent_bytes_to_trainer_payload(
    latent_bytes: bytes,
    *,
    frame_rate: float,
) -> dict[str, Any]:
    """Convert ComfyUI ``SaveLatent`` bytes to the trainer's negative-latent payload."""
    with tempfile.NamedTemporaryFile(suffix=".latent") as temp:
        temp.write(latent_bytes)
        temp.flush()
        tensors = load_file(temp.name, device="cpu")

    if "latent_tensor" not in tensors:
        raise ValueError("ComfyUI latent file did not contain a 'latent_tensor' tensor")

    latent = tensors["latent_tensor"].detach().cpu().contiguous()
    if latent.ndim == 5 and latent.shape[0] == 1:
        latent = latent.squeeze(0).contiguous()
    if latent.ndim != 4:
        raise ValueError(
            "Expected ComfyUI LTXV latent shape [B, C, F, H, W] or [C, F, H, W], "
            f"got {tuple(latent.shape)}"
        )

    return {
        "latents": latent,
        "num_frames": latent.shape[1],
        "height": latent.shape[2],
        "width": latent.shape[3],
        "fps": frame_rate,
    }


def _fetch_saved_latent(history_entry: dict[str, Any], *, server_url: str, save_node_id: str) -> bytes:
    status = history_entry.get("status", {})
    if status.get("status_str") == "error" or status.get("completed") is False:
        messages = status.get("messages") or []
        raise RuntimeError(f"ComfyUI prompt failed: {messages or status}")

    outputs = history_entry.get("outputs", {})
    save_output = outputs.get(save_node_id)
    if not save_output:
        raise RuntimeError(f"ComfyUI prompt completed without output from SaveLatent node {save_node_id!r}")

    latents = save_output.get("latents") or []
    if not latents:
        raise RuntimeError(f"ComfyUI SaveLatent node {save_node_id!r} did not report a saved latent")

    locator = latents[0]
    query = urllib.parse.urlencode(
        {
            "filename": locator["filename"],
            "subfolder": locator.get("subfolder", ""),
            "type": locator.get("type", "output"),
        }
    )
    return _request_bytes(f"{server_url}/view?{query}")


def _normalize_server_url(server_url: str) -> str:
    if "://" not in server_url:
        server_url = f"http://{server_url}"
    return server_url.rstrip("/")


def _request_json(url: str, data: dict[str, Any] | None) -> Any:  # noqa: ANN401
    request_data = None if data is None else json.dumps(data).encode("utf-8")
    headers = {} if data is None else {"Content-Type": "application/json"}
    request = urllib.request.Request(url, data=request_data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"ComfyUI request failed with HTTP {e.code}: {body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Could not reach ComfyUI at {url}: {e}") from e


def _request_bytes(url: str) -> bytes:
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            return response.read()
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"ComfyUI download failed with HTTP {e.code}: {body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Could not download ComfyUI output from {url}: {e}") from e
