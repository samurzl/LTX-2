# Warm Model Server

The standard preprocessing and training commands normally run in separate Python processes, so their model objects
cannot survive from one command to the next. The warm model server provides one persistent process that owns the
models while the existing commands act as clients.

Models load on first use and are cached by checkpoint identity, component type, dtype, and quantization. Inactive
components move to CPU RAM when GPU memory is needed. Compatible LoRA jobs reuse the frozen transformer while getting
fresh adapters, optimizers, schedulers, and dataloaders.

## Start the server

From `packages/ltx-trainer`, keep this running in one terminal:

```bash
uv run python scripts/warm_server.py serve
```

Then use the normal commands from another terminal, unchanged:

```bash
uv run python scripts/process_dataset.py dataset.json \
  --resolution-buckets "960x544x49" \
  --model-path /models/ltx-2.safetensors \
  --text-encoder-path /models/gemma.safetensors

uv run python scripts/train.py configs/my_lora.yaml
```

`process_dataset.py`, `process_captions.py`, `process_videos.py`, and `train.py` automatically use the server when it
is running. If it is not running, they retain their original one-shot behavior. The client command waits for its job
to finish; detailed live progress appears in the server terminal.

Inspect or stop the server with:

```bash
uv run python scripts/warm_server.py status
uv run python scripts/warm_server.py stop
```

Set `LTX_TRAINER_DISABLE_WARM_SERVER=1` for a command that should deliberately bypass a running server.

## Comfy negative generation

With `negative_backend: comfy`, the local warm server offloads its cached models to CPU before sending workflows to
ComfyUI. ComfyUI owns and caches its checkpoint, text encoder, and distilled LoRA independently. Keep both servers
running to retain both model suites.

## Safety and limitations

- Warm transformer reuse applies to LoRA training. Full fine-tuning mutates base weights and therefore uses isolated
  transformer loading inside the server.
- Restart the server before changing `acceleration.mixed_precision_mode`; Accelerate stores this setting process-wide.
- The warm server is currently single-process/single-GPU. Commands launched through multi-process Accelerate
  automatically bypass it.
- Jobs run sequentially. Multiple client commands may wait in the Unix socket queue.
- An 8-bit bitsandbytes Gemma model cannot safely move back to CPU. It is evicted when another model group needs the
  GPU and may need to load again later.
- The native synthetic-negative pipeline and optional latent decoder retain their existing isolated lifecycles.

## Optional batch manifest

`run_session.py` remains available when a fixed batch of jobs is convenient:

```yaml
jobs:
  - type: preprocess
    args:
      dataset_path: /data/dataset.json
      resolution_buckets: "960x544x49"
      model_path: /models/ltx-2.safetensors
      text_encoder_path: /models/gemma.safetensors
  - type: train
    config: configs/my_lora.yaml
```

```bash
uv run python scripts/run_session.py session.yaml
```

The server and manifest runner use the same `WarmModelPool` implementation.
