# Training Modes Guide

The trainer supports several training modes, each suited for different use cases and requirements.

## 🎯 Standard LoRA Training (Video-Only)

Standard LoRA (Low-Rank Adaptation) training fine-tunes the model by adding small, trainable adapter layers while
keeping the base model frozen. This approach:

- **Requires significantly less memory and compute** than full fine-tuning
- **Produces small, portable weight files** (typically a few hundred MB)
- **Is ideal for learning specific styles, effects, or concepts**
- **Can be easily combined with other LoRAs** during inference

Configure standard LoRA training with:

```yaml
model:
  training_mode: "lora"

training_strategy:
  name: "text_to_video"
  first_frame_conditioning_p: 0.1
  with_audio: false  # Video-only training
```

## 🔊 Audio-Video LoRA Training

LTX-2 supports joint audio-video generation. You can train LoRA adapters that affect both video and audio output:

- **Synchronized audio-video generation** - Audio matches the visual content
- **Same efficient LoRA approach** - Just enable audio training
- **Requires audio latents** - Dataset must include preprocessed audio

Configure audio-video training with:

```yaml
model:
  training_mode: "lora"

training_strategy:
  name: "text_to_video"
  first_frame_conditioning_p: 0.1
  with_audio: true  # Enable audio training
  audio_latents_dir: "audio_latents"  # Directory containing audio latents
```

**Example configuration file:**

- 📄 [Audio-Video LoRA Training](../configs/ltx2_av_lora.yaml)

**Dataset structure for audio-video training:**

```
preprocessed_data_root/
├── latents/           # Video latents
├── conditions/        # Text embeddings
└── audio_latents/     # Audio latents (required when with_audio: true)
```

> [!IMPORTANT]
> When training audio-video LoRAs, ensure your `target_modules` configuration captures video, audio, and
> cross-modal attention branches. Use patterns like `"to_k"` instead of `"attn1.to_k"` to match:
> - Video modules: `attn1.to_k`, `attn2.to_k`
> - Audio modules: `audio_attn1.to_k`, `audio_attn2.to_k`
> - Cross-modal modules: `audio_to_video_attn.to_k`, `video_to_audio_attn.to_k`
>
> The cross-modal attention modules (`audio_to_video_attn` and `video_to_audio_attn`) enable bidirectional
> information flow between audio and video, which is critical for synchronized audiovisual generation.
> See [Understanding Target Modules](configuration-reference.md#understanding-target-modules) for detailed guidance.

> [!NOTE]
> You can generate audio during validation even if you're not training the audio branch.
> Set `validation.generate_audio: true` independently of `training_strategy.with_audio`.

## 🧭 NSYNC Contrastive Training

`nsync` adds structured contrastive branches to standard training. In the advanced JSON/JSONL path,
each positive sample defines:

- one or more categories
- one or more negative branches
- zero or more anchor rules that resolve to other positive samples at training time

During training, the trainer runs:

- the normal positive branch
- one negative branch per configured negative slot
- one randomized anchor branch per configured anchor rule when `nsync.use_anchor: true`

The update is applied with the NSYNC projection rule:

```text
g = pos - proj_pos_neg + proj_pos_anchor + agree_pos_anchor
```

### When to Use NSYNC

- Style or aesthetic fine-tuning where you want the LoRA to move away from a specific undesired look
- Datasets where each positive sample can be paired with a "what this should not look like" caption
- Cases where you want to bootstrap negatives automatically instead of curating a full negative media set

### Preprocessing Requirements

Advanced NSYNC metadata uses a structured `nsync` object in JSON/JSONL:

Example:

```json
[
  {
    "caption": "A cinematic close-up of a cat on a windowsill",
    "media_path": "videos/cat.mp4",
    "nsync": {
      "categories": ["cat", "cinematic"],
      "negatives": [
        { "media": "positive", "caption": "A low-detail surveillance-camera style cat video" },
        {
          "media": "synthetic",
          "prompt": "A shaky handheld phone recording of a cat in flat lighting",
          "caption": "A shaky handheld cat video"
        }
      ],
      "anchors": [
        { "required_categories": ["cat"] },
        { "required_categories": ["cat"], "extra_random_category": true }
      ]
    }
  }
]
```

Then preprocess with:

```bash
uv run python scripts/process_dataset.py dataset.json \
    --resolution-buckets "960x544x49" \
    --model-path /path/to/ltx-2-model.safetensors \
    --text-encoder-path /path/to/gemma-model
```

This creates:

```text
preprocessed_data_root/
├── latents/
├── conditions/
├── negative_latents/
├── negative_conditions/
├── negative_audio_latents/   # only when with_audio: true
└── nsync_manifest.json
```

### Enabling NSYNC in Training Configs

Add the `nsync` section to any standard text-to-video or IC-LoRA config:

```yaml
nsync:
  enabled: true
  use_anchor: true
  negative_latents_dir: "negative_latents"
  negative_conditions_dir: "negative_conditions"
  negative_audio_latents_dir: "negative_audio_latents"
  projection_eps: 1.0e-12
```

Notes:

- For `text_to_video` with `with_audio: true`, preprocessing must also produce `negative_audio_latents/`.
- For `video_to_video`, negative branches still reuse the positive sample's `reference_latents`.
- `media: "positive"` reuses the positive sample's latent media and swaps only the conditioning caption.
- `media: "synthetic"` generates negative media from `prompt` and still trains on `caption`.
- `extra_random_category: true` first samples a feasible extra category name, then picks an anchor that has both the required categories and that sampled extra category.
- Legacy single-negative NSYNC datasets without `nsync_manifest.json` still use the old paired-negative path.

## 🔥 Full Model Fine-tuning

Full model fine-tuning updates all parameters of the base model, providing maximum flexibility but
requiring substantial computational resources and larger training datasets:

- **Offers the highest potential quality and capability improvements**
- **Requires multiple GPUs** and distributed training techniques (e.g., FSDP)
- **Produces large checkpoint files** (several GB)
- **Best for major model adaptations** or when LoRA limitations are reached

Configure full fine-tuning with:

```yaml
model:
  training_mode: "full"

training_strategy:
  name: "text_to_video"
  first_frame_conditioning_p: 0.1
```

> [!IMPORTANT]
> Full fine-tuning of LTX-2 requires multiple high-end GPUs (e.g., 4-8× H100 80GB) and distributed
> training with FSDP. See [Training Guide](training-guide.md) for multi-GPU setup instructions.

## 🔄 In-Context LoRA (IC-LoRA) Training

IC-LoRA is a specialized training mode for video-to-video transformations.
Unlike standard training modes that learn from individual videos, IC-LoRA learns transformations from pairs of videos.
IC-LoRA enables a wide range of advanced video-to-video applications, such as:

- **Control adapters** (e.g., Depth, Pose): Learn to map from a control signal (like a depth map or pose skeleton) to a
  target video
- **Video deblurring**: Transform blurry input videos into sharp, high-quality outputs
- **Style transfer**: Apply the style of a reference video to a target video sequence
- **Colorization**: Convert grayscale reference videos into colorized outputs
- **Restoration and enhancement**: Denoise, upscale, or restore old or degraded videos

By providing paired reference and target videos, IC-LoRA can learn complex transformations that go beyond caption-based
conditioning.

IC-LoRA training fundamentally differs from standard LoRA and full fine-tuning:

- **Reference videos** provide clean, unnoised conditioning input showing the "before" state
- **Target videos** are noised during training and represent the desired "after" state
- **The model learns transformations** from reference videos to target videos
- **Loss is applied only to the target portion**, not the reference
- **Training and inference time increase significantly** due to the doubled sequence length

To enable IC-LoRA training, configure your YAML file with:

```yaml
model:
  training_mode: "lora"  # Required: IC-LoRA uses LoRA mode

training_strategy:
  name: "video_to_video"
  first_frame_conditioning_p: 0.1
  reference_latents_dir: "reference_latents"  # Directory for reference video latents
```

**Example configuration file:**

- 📄 [IC-LoRA Training](../configs/ltx2_v2v_ic_lora.yaml) - Video-to-video transformation training

### Dataset Requirements for IC-LoRA

- Your dataset must contain **paired videos** where each target video has a corresponding reference video
- Reference and target videos must have the **same frame count** (length)
- Reference videos can optionally be at **lower spatial resolution** than target videos (
  see [Scaled Reference Conditioning](#scaled-reference-conditioning) below)
- Both reference and target videos should be **preprocessed** before training

**Dataset structure for IC-LoRA training:**

```
preprocessed_data_root/
├── latents/            # Target video latents (what the model learns to generate)
├── conditions/         # Text embeddings for each video
└── reference_latents/  # Reference video latents (conditioning input)
```

### Generating Reference Videos

We provide an example script to generate reference videos (e.g., Canny edge maps) for a given dataset.
The script takes a JSON file as input (e.g., output of `caption_videos.py`) and updates it with the generated reference
video paths.

```bash
uv run python scripts/compute_reference.py scenes_output_dir/ \
    --output scenes_output_dir/dataset.json
```

To compute a different condition (depth maps, pose skeletons, etc.), modify the `compute_reference()` function in the
script.

### Configuration Requirements for IC-LoRA

- You **must** provide `reference_videos` in your validation configuration when using IC-LoRA training
- The number of reference videos must match the number of validation prompts

Example validation configuration for IC-LoRA:

```yaml
validation:
  prompts:
    - "First prompt describing the desired output"
    - "Second prompt describing the desired output"
  reference_videos:
    - "/path/to/reference1.mp4"
    - "/path/to/reference2.mp4"
  reference_downscale_factor: 1  # Set to match preprocessing (e.g., 2 for half resolution)
  include_reference_in_output: true  # Show reference side-by-side with output
```

### Scaled Reference Conditioning

For more efficient training and inference, you can use **downscaled reference videos** while keeping target videos at
full resolution. This reduces the number of conditioning tokens, leading to:

- **Faster training** due to shorter sequence lengths
- **Faster inference** with reduced memory usage
- **Same aspect ratio** maintained between reference and target

#### How It Works

When the reference video has resolution `H/n × W/n` and the target video has resolution `H × W`, the trainer
automatically detects this scale factor `n` and adjusts the positional encodings so that the reference positions
map to the correct locations in the target coordinate space.

#### Preprocessing Datasets with Scaled References

Use the `--reference-downscale-factor` option when running `process_dataset.py`:

```bash
# Process dataset with scaled reference videos (half resolution)
uv run python scripts/process_dataset.py dataset.json \
    --resolution-buckets 768x768x25 \
    --model-path /path/to/ltx2.safetensors \
    --text-encoder-path /path/to/gemma \
    --reference-column "reference_path" \
    --reference-downscale-factor 2
```

This will:

- Process target videos at 768×768 resolution
- Process reference videos at 384×384 resolution (768 / 2)
- The trainer will automatically infer the scale factor from the dimension ratio

**Important**: Set `reference_downscale_factor: 2` in your validation configuration to match the preprocessing:

```yaml
validation:
  reference_downscale_factor: 2  # Must match the preprocessing factor
  reference_videos:
    - "/path/to/reference1.mp4"
    - "/path/to/reference2.mp4"
```

> [!NOTE]
> The scale factor must be a positive integer, and all dimensions must be divisible by 32.
> Common scale factors are 1 (no scaling), 2 (half resolution), or 4 (quarter resolution).

## 📊 Training Mode Comparison

| Aspect               | LoRA                           | Audio-Video LoRA               | NSYNC                         | Full Fine-tuning | IC-LoRA                        |
|----------------------|--------------------------------|--------------------------------|-------------------------------|------------------|--------------------------------|
| **Memory Usage**     | Low                            | Low-Medium                     | Medium                        | High             | Medium                         |
| **Training Speed**   | Fast                           | Fast                           | Medium                        | Slow             | Medium                         |
| **Output Size**      | 100MB-few GB (depends on rank) | 100MB-few GB (depends on rank) | Same as underlying train mode | Tens of GB       | 100MB-few GB (depends on rank) |
| **Flexibility**      | Medium                         | Medium                         | Medium-High                   | High             | Specialized                    |
| **Audio Support**    | Optional                       | Yes                            | Optional                      | Optional         | No                             |
| **Reference Videos** | No                             | No                             | No / reused for IC-LoRA       | No               | Yes (required)                 |

## 🎬 Using Trained Models for Inference

After training, use the [`ltx-pipelines`](../../ltx-pipelines/) package for production inference with your trained
LoRAs:

| Training Mode           | Recommended Pipeline                                  |
|-------------------------|-------------------------------------------------------|
| LoRA / Audio-Video LoRA | `TI2VidOneStagePipeline` or `TI2VidTwoStagesPipeline` |
| IC-LoRA                 | `ICLoraPipeline`                                      |

All pipelines support loading custom LoRAs via the `loras` parameter. See the [`ltx-pipelines`](../../ltx-pipelines/)
package
documentation for detailed usage instructions.

## 🚀 Next Steps

Once you've chosen your training mode:

- Set up your dataset using [Dataset Preparation](dataset-preparation.md)
- Configure your training parameters in [Configuration Reference](configuration-reference.md)
- Start training with the [Training Guide](training-guide.md)

> [!TIP]
> Need a training mode that's not covered here?
> See [Implementing Custom Training Strategies](custom-training-strategies.md)
> to learn how to create your own strategy for specialized use cases like video inpainting, audio-only training, or
> custom conditioning.
