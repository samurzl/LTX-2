# LTX-2 Trainer

This package provides tools and scripts for training and fine-tuning
Lightricks' **LTX-2** audio-video generation model. It enables LoRA training, full
fine-tuning, `nsync` contrastive training with paired negatives, and training of
video-to-video transformations (IC-LoRA) on custom datasets.

---

## 📖 Documentation

All detailed guides and technical documentation are in the [docs](./docs/) directory:

- [⚡ Quick Start Guide](docs/quick-start.md)
- [🎬 Dataset Preparation](docs/dataset-preparation.md)
- [🛠️ Training Modes](docs/training-modes.md)
- [⚙️ Configuration Reference](docs/configuration-reference.md)
- [🚀 Training Guide](docs/training-guide.md)
- [🧪 Inference Guide](../ltx-pipelines/README.md)
- [🔧 Utility Scripts](docs/utility-scripts.md)
- [📚 LTX-Core Documentation](../ltx-core/README.md)
- [🛡️ Troubleshooting Guide](docs/troubleshooting.md)

---

## 🧭 NSYNC at a Glance

`nsync` training works with the existing offline `.precomputed` workflow. Add `negative_caption`
to your dataset metadata, optionally add `negative_media_path`, preprocess with
[`scripts/process_dataset.py`](scripts/process_dataset.py), then enable the `nsync` section in your
training config.

Start here:

- [🎬 Dataset Preparation](docs/dataset-preparation.md#nsync-negative-preprocessing)
- [🛠️ Training Modes](docs/training-modes.md#-nsync-contrastive-training)
- [🚀 Training Guide](docs/training-guide.md#-nsync-training)

---

## 🔧 Requirements

- **LTX-2 Model Checkpoint** - Local `.safetensors` file
- **Gemma Text Encoder** - Local Gemma model directory (required for LTX-2)
- **Linux with CUDA** - CUDA 13+ recommended for optimal performance
- **Nvidia GPU with 80GB+ VRAM** - Recommended for the standard config. For GPUs with 32GB VRAM (e.g., RTX 5090),
  use the [low VRAM config](configs/ltx2_av_lora_low_vram.yaml) which enables INT8 quantization and other
  memory optimizations

---

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

- **Share Your Work**: If you've trained interesting LoRAs or achieved cool results, please share them with the
  community.
- **Report Issues**: Found a bug or have a suggestion? Open an issue on GitHub.
- **Submit PRs**: Help improve the codebase with bug fixes or general improvements.
- **Feature Requests**: Have ideas for new features? Let us know through GitHub issues.

---

## 💬 Join the Community

Have questions, want to share your results, or need real-time help?

Join our [community Discord server](https://discord.gg/ltxplatform) to connect with other users and the development
team!

- Get troubleshooting help
- Share your training results and workflows
- Collaborate on new ideas and features
- Stay up to date with announcements and updates

We look forward to seeing you there!

---

Happy training! 🎉
