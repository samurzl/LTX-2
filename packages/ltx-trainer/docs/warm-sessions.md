# Warm Model Sessions

The standard preprocessing and training commands run one job per Python process. Use a warm session when running
several jobs with the same checkpoints so frozen model components do not have to be reconstructed for every job.

Warm sessions load components on first use and cache them by checkpoint identity, component type, dtype, and
quantization. Inactive components move to CPU RAM when GPU memory is needed. Consecutive compatible LoRA jobs keep
the frozen transformer available while creating fresh adapters, optimizers, schedulers, and dataloaders for every job.

## Session manifest

Create a YAML file containing preprocessing and training jobs:

```yaml
jobs:
  - type: preprocess
    args:
      dataset_path: /data/first/dataset.json
      resolution_buckets: "960x544x49"
      model_path: /models/ltx-2.safetensors
      text_encoder_path: /models/gemma.safetensors
      output_dir: /data/first/.precomputed
      with_audio: true

  - type: preprocess
    args:
      dataset_path: /data/second/dataset.json
      resolution_buckets: "960x544x49"
      model_path: /models/ltx-2.safetensors
      text_encoder_path: /models/gemma.safetensors
      output_dir: /data/second/.precomputed

  - type: train
    config: ../configs/first_lora.yaml

  - type: train
    config: ../configs/second_lora.yaml
    disable_progress_bars: false
```

Paths in each trainer config retain their normal meaning. A relative `config` path is resolved relative to the session
manifest.

Run the complete sequence from `packages/ltx-trainer`:

```bash
uv run python scripts/run_session.py session.yaml
```

Cache hits and misses are logged. Editing or replacing a checkpoint invalidates its corresponding cache entry.

## Safety and limitations

- Warm transformer reuse applies to LoRA training. Full fine-tuning mutates base weights and therefore uses isolated
  model loading.
- All training jobs in one session must use the same `acceleration.mixed_precision_mode`, because Accelerate keeps
  this process-wide state after initialization.
- Multiple training jobs in one session currently require a single process/GPU. A distributed session may contain
  one training job.
- Jobs run sequentially. The model pool is not intended to preprocess while training is active.
- An 8-bit bitsandbytes Gemma model cannot safely move back to CPU. It is evicted when another model group needs the
  GPU and may need to load again later.
- The native synthetic-negative pipeline and optional latent decoder retain their existing isolated lifecycles.

The Python API is also available for integrations that already own a long-lived process:

```python
from ltx_trainer.model_pool import WarmModelPool

model_pool = WarmModelPool()
# Pass model_pool=... to preprocess_dataset() or LtxvTrainer(...).
```

Call `LtxvTrainer.release_warm_models()` after every pooled LoRA job; `run_session.py` does this automatically.
