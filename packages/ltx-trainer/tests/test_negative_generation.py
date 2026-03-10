from pathlib import Path
from types import SimpleNamespace

import torch

from ltx_trainer.negative_generation import NegativeLatentGenerationSpec, generate_negative_latents


def test_generate_negative_latents_uses_vocoder_sample_rate_for_preview_audio(
    tmp_path: Path,
    monkeypatch,
) -> None:
    positive_latents_dir = tmp_path / "latents"
    positive_latent_path = positive_latents_dir / "videos" / "sample.pt"
    positive_latent_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "latents": torch.zeros(128, 7, 2, 2),
            "num_frames": 7,
            "height": 2,
            "width": 2,
            "fps": 24.0,
        },
        positive_latent_path,
    )

    components = SimpleNamespace(
        transformer=object(),
        video_vae_decoder=object(),
        audio_vae_decoder=SimpleNamespace(sample_rate=16000),
        vocoder=SimpleNamespace(output_sampling_rate=48000),
    )

    monkeypatch.setattr("ltx_trainer.negative_generation.load_ltx_model", lambda **_: components)
    monkeypatch.setattr("ltx_trainer.negative_generation.load_text_encoder", lambda **_: object())
    monkeypatch.setattr("ltx_trainer.negative_generation.load_embeddings_processor", lambda **_: object())

    class FakeSampler:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        def generate_latents(self, generation_config, device: str, decode_preview: bool = False):
            assert decode_preview is True
            assert generation_config.num_frames == 49
            return SimpleNamespace(
                video_latents=torch.zeros(128, 7, 2, 2),
                num_frames=7,
                height=2,
                width=2,
                audio_latents=torch.zeros(8, 4, 16),
                num_time_steps=4,
                frequency_bins=16,
                duration=49 / 24.0,
                preview_video=torch.zeros(3, 49, 64, 64),
                preview_audio=torch.zeros(2, 96000),
            )

    monkeypatch.setattr("ltx_trainer.negative_generation.ValidationSampler", FakeSampler)

    save_video_calls: list[dict] = []

    def fake_save_video(video_tensor, output_path, fps: float, audio=None, audio_sample_rate=None) -> None:
        save_video_calls.append(
            {
                "video_tensor": video_tensor,
                "output_path": output_path,
                "fps": fps,
                "audio": audio,
                "audio_sample_rate": audio_sample_rate,
            }
        )

    monkeypatch.setattr("ltx_trainer.negative_generation.save_video", fake_save_video)

    generate_negative_latents(
        [
            NegativeLatentGenerationSpec(
                positive_media_path="videos/sample.mp4",
                output_rel_path="videos/sample.pt",
                prompt="negative prompt",
            )
        ],
        positive_latents_dir=positive_latents_dir,
        output_dir=tmp_path / "negative_latents",
        model_path=tmp_path / "model.safetensors",
        text_encoder_path=tmp_path / "gemma",
        device="cpu",
        with_audio=True,
        audio_output_dir=tmp_path / "negative_audio_latents",
        save_previews=True,
        preview_output_dir=tmp_path / "previews",
    )

    assert len(save_video_calls) == 1
    assert save_video_calls[0]["fps"] == 24.0
    assert save_video_calls[0]["audio_sample_rate"] == 48000
