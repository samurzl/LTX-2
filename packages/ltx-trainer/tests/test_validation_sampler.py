import pytest
import torch

from ltx_trainer.validation_sampler import GenerationConfig, ValidationSampler


def _make_sampler() -> ValidationSampler:
    return ValidationSampler(
        transformer=None,
        vae_decoder=None,
        vae_encoder=None,
        text_encoder=None,
        audio_decoder=None,
        vocoder=None,
        embeddings_processor=None,
    )


def test_apply_latent_conditioning_uses_first_latent_frame_tokens() -> None:
    sampler = _make_sampler()
    config = GenerationConfig(prompt="test", height=64, width=64, num_frames=9, generate_audio=False)
    video_tools = sampler._create_video_latent_tools(config)
    initial_state = video_tools.create_initial_state(device=torch.device("cpu"), dtype=torch.float32)

    condition_latents = torch.zeros(128, 2, 2, 2, dtype=torch.float32)
    condition_latents[:, 0, :, :] = 1.0
    expected_tokens = sampler._video_patchifier.patchify(condition_latents[:, :1].unsqueeze(0))

    conditioned_state = sampler._apply_latent_conditioning(initial_state, condition_latents, torch.device("cpu"))

    assert torch.allclose(conditioned_state.latent[:, : expected_tokens.shape[1]], expected_tokens)
    assert torch.allclose(conditioned_state.clean_latent[:, : expected_tokens.shape[1]], expected_tokens)
    assert torch.count_nonzero(conditioned_state.denoise_mask[:, : expected_tokens.shape[1]]) == 0
    assert torch.equal(
        conditioned_state.denoise_mask[:, expected_tokens.shape[1] :],
        initial_state.denoise_mask[:, expected_tokens.shape[1] :],
    )


def test_validate_config_rejects_multiple_conditioning_sources() -> None:
    sampler = _make_sampler()
    config = GenerationConfig(
        prompt="test",
        height=64,
        width=64,
        num_frames=9,
        generate_audio=False,
        condition_image=torch.zeros(3, 64, 64),
        condition_latents=torch.zeros(128, 1, 2, 2),
    )

    with pytest.raises(ValueError, match="Only one of condition_latents or condition_image"):
        sampler._validate_config(
            config,
            require_video_decoder=False,
            require_audio_decoder=False,
        )
