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
to finish. Logs, warnings, and Rich progress displays are streamed back to the terminal where that client command was
started.

The server terminal becomes a live model dashboard. It shows every model the pool has encountered, including its
component type, checkpoint, size, dtype, options, and current device. Green models are loaded, red models are unloaded
or evicted, and orange models are currently loading or moving between devices. Since the underlying model loaders do
not report byte-level progress, the green loading bar pulses rather than displaying a fabricated percentage.

Inspect or stop the server with:

```bash
uv run python scripts/warm_server.py status
uv run python scripts/warm_server.py stop
```

Set `LTX_TRAINER_DISABLE_WARM_SERVER=1` for a command that should deliberately bypass a running server.

## Additional startup and throughput caches

- Video frame counts are persisted in `.precomputed/.media_metadata.json` and invalidated by source size/mtime.
  Completed latent outputs are filtered before any video metadata probe.
- Discovered training/validation `.pt` indexes are reused between preflight, dataloader creation, and later server
  jobs. A preprocessing write invalidates the corresponding in-memory dataset indexes.
- Validation prompt embeddings are retained in memory by checkpoint and exact prompt set. The shared negative prompt
  is encoded only once.
- Video/audio decoders and the vocoder are not loaded when validation sampling has no interval or prompts.
- TorchInductor uses `~/.cache/ltx-trainer/torchinductor` by default, allowing compatible compiled kernels and FX
  graphs to survive later jobs and server restarts. Override it with `TORCHINDUCTOR_CACHE_DIR`.

Caption preprocessing now performs one Gemma forward per batch. `process_captions.py` defaults to batch size 8;
`process_dataset.py` keeps the conservative default of 1. Increase `--batch-size` for throughput when VRAM permits,
or reduce it if Gemma runs out of memory.

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
