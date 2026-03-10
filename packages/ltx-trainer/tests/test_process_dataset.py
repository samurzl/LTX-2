import json
from pathlib import Path

import process_dataset as process_dataset_script


def _write_dataset(tmp_path: Path, rows: list[dict]) -> Path:
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps(rows), encoding="utf-8")
    return dataset_path


def test_preprocess_dataset_generates_negative_branches_from_negative_captions(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset_path = _write_dataset(
        tmp_path,
        [
            {"caption": "positive one", "media_path": "videos/one.mp4", "negative_caption": "negative one"},
            {"caption": "positive two", "media_path": "videos/two.mp4", "negative_caption": "negative two"},
        ],
    )

    caption_calls: list[dict] = []
    latent_calls: list[dict] = []
    generate_calls: list[dict] = []

    monkeypatch.setattr(
        process_dataset_script,
        "compute_captions_embeddings",
        lambda **kwargs: caption_calls.append(kwargs),
    )
    monkeypatch.setattr(
        process_dataset_script,
        "compute_latents",
        lambda **kwargs: latent_calls.append(kwargs),
    )
    monkeypatch.setattr(
        process_dataset_script,
        "generate_missing_negative_latents",
        lambda specs, **kwargs: generate_calls.append({"specs": specs, **kwargs}),
    )

    process_dataset_script.preprocess_dataset(
        dataset_file=str(dataset_path),
        caption_column="caption",
        video_column="media_path",
        resolution_buckets=[(9, 544, 960)],
        batch_size=1,
        output_dir=str(tmp_path / ".precomputed"),
        lora_trigger=None,
        vae_tiling=False,
        decode=False,
        model_path="/tmp/model.safetensors",
        text_encoder_path="/tmp/gemma",
        device="cpu",
    )

    assert len(caption_calls) == 2
    assert caption_calls[1]["caption_column"] == "negative_caption"
    assert caption_calls[1]["media_column"] == "media_path"
    assert len(latent_calls) == 1
    assert len(generate_calls) == 1
    assert [spec.media_path for spec in generate_calls[0]["specs"]] == ["videos/one.mp4", "videos/two.mp4"]
    assert Path(generate_calls[0]["output_dir"]).name == "negative_latents"
    assert generate_calls[0]["inference_steps"] == 40
    assert generate_calls[0]["use_first_frame_conditioning"] is False


def test_preprocess_dataset_generates_negative_branches_for_image_datasets(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset_path = _write_dataset(
        tmp_path,
        [
            {"caption": "positive one", "media_path": "images/one.webp", "negative_caption": "negative one"},
            {"caption": "positive two", "media_path": "images/two.webp", "negative_caption": "negative two"},
        ],
    )

    caption_calls: list[dict] = []
    latent_calls: list[dict] = []
    generate_calls: list[dict] = []

    monkeypatch.setattr(
        process_dataset_script,
        "compute_captions_embeddings",
        lambda **kwargs: caption_calls.append(kwargs),
    )
    monkeypatch.setattr(
        process_dataset_script,
        "compute_latents",
        lambda **kwargs: latent_calls.append(kwargs),
    )
    monkeypatch.setattr(
        process_dataset_script,
        "generate_missing_negative_latents",
        lambda specs, **kwargs: generate_calls.append({"specs": specs, **kwargs}),
    )

    process_dataset_script.preprocess_dataset(
        dataset_file=str(dataset_path),
        caption_column="caption",
        video_column="media_path",
        resolution_buckets=[(1, 544, 960)],
        batch_size=1,
        output_dir=str(tmp_path / ".precomputed"),
        lora_trigger=None,
        vae_tiling=False,
        decode=False,
        model_path="/tmp/model.safetensors",
        text_encoder_path="/tmp/gemma",
        device="cpu",
    )

    assert len(caption_calls) == 2
    assert len(latent_calls) == 1
    assert [spec.media_path for spec in generate_calls[0]["specs"]] == [
        "images/one.webp",
        "images/two.webp",
    ]
    assert [spec.output_rel_path for spec in generate_calls[0]["specs"]] == [
        "images/one.pt",
        "images/two.pt",
    ]


def test_preprocess_dataset_uses_user_supplied_negative_media_without_generation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset_path = _write_dataset(
        tmp_path,
        [
            {
                "caption": "positive one",
                "media_path": "videos/one.mp4",
                "negative_caption": "negative one",
                "negative_media_path": "negatives/one.mp4",
            },
            {
                "caption": "positive two",
                "media_path": "videos/two.mp4",
                "negative_caption": "negative two",
                "negative_media_path": "negatives/two.mp4",
            },
        ],
    )

    negative_subset_rows: list[str] = []
    generate_calls: list[dict] = []

    monkeypatch.setattr(process_dataset_script, "compute_captions_embeddings", lambda **_: None)

    def fake_compute_latents(**kwargs):
        if kwargs["video_column"] == "negative_media_path":
            negative_subset_rows.append(Path(kwargs["dataset_file"]).read_text(encoding="utf-8"))

    monkeypatch.setattr(process_dataset_script, "compute_latents", fake_compute_latents)
    monkeypatch.setattr(
        process_dataset_script,
        "generate_missing_negative_latents",
        lambda specs, **kwargs: generate_calls.append({"specs": specs, **kwargs}),
    )

    process_dataset_script.preprocess_dataset(
        dataset_file=str(dataset_path),
        caption_column="caption",
        video_column="media_path",
        resolution_buckets=[(9, 544, 960)],
        batch_size=1,
        output_dir=str(tmp_path / ".precomputed"),
        lora_trigger=None,
        vae_tiling=False,
        decode=False,
        model_path="/tmp/model.safetensors",
        text_encoder_path="/tmp/gemma",
        device="cpu",
    )

    assert len(negative_subset_rows) == 1
    assert '"negative_media_path": "negatives/one.mp4"' in negative_subset_rows[0]
    assert '"negative_media_path": "negatives/two.mp4"' in negative_subset_rows[0]
    assert generate_calls == []


def test_preprocess_dataset_only_generates_missing_negatives_and_tracks_audio_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset_path = _write_dataset(
        tmp_path,
        [
            {
                "caption": "positive one",
                "media_path": "videos/one.mp4",
                "negative_caption": "negative one",
                "negative_media_path": "negatives/one.mp4",
            },
            {
                "caption": "positive two",
                "media_path": "videos/two.mp4",
                "negative_caption": "negative two",
            },
        ],
    )

    negative_subset_rows: list[str] = []
    generate_calls: list[dict] = []

    monkeypatch.setattr(process_dataset_script, "compute_captions_embeddings", lambda **_: None)

    def fake_compute_latents(**kwargs):
        if kwargs["video_column"] == "negative_media_path":
            negative_subset_rows.append(Path(kwargs["dataset_file"]).read_text(encoding="utf-8"))

    monkeypatch.setattr(process_dataset_script, "compute_latents", fake_compute_latents)
    monkeypatch.setattr(
        process_dataset_script,
        "generate_missing_negative_latents",
        lambda specs, **kwargs: generate_calls.append({"specs": specs, **kwargs}),
    )

    process_dataset_script.preprocess_dataset(
        dataset_file=str(dataset_path),
        caption_column="caption",
        video_column="media_path",
        resolution_buckets=[(9, 544, 960)],
        batch_size=1,
        output_dir=str(tmp_path / ".precomputed"),
        lora_trigger=None,
        vae_tiling=False,
        decode=False,
        model_path="/tmp/model.safetensors",
        text_encoder_path="/tmp/gemma",
        device="cpu",
        with_audio=True,
    )

    assert len(negative_subset_rows) == 1
    assert '"negative_media_path": "negatives/one.mp4"' in negative_subset_rows[0]
    assert '"negative_media_path": "videos/two.mp4"' not in negative_subset_rows[0]
    assert len(generate_calls) == 1
    assert [spec.media_path for spec in generate_calls[0]["specs"]] == ["videos/two.mp4"]
    assert Path(generate_calls[0]["audio_output_dir"]).name == "negative_audio_latents"
    assert generate_calls[0]["inference_steps"] == 40
    assert generate_calls[0]["use_first_frame_conditioning"] is False


def test_preprocess_dataset_builds_advanced_nsync_manifest_and_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset_path = _write_dataset(
        tmp_path,
        [
            {
                "caption": "positive one",
                "media_path": "videos/one.mp4",
                "nsync": {
                    "categories": ["cat", "cinematic"],
                    "negatives": [
                        {"media": "positive", "caption": "negative one a"},
                        {
                            "media": "synthetic",
                            "prompt": "synthetic negative one b prompt",
                            "caption": "negative one b",
                        },
                    ],
                    "anchors": [{"required_categories": ["cat"], "extra_random_category": True}],
                },
            },
            {
                "caption": "positive two",
                "media_path": "videos/two.mp4",
                "nsync": {
                    "categories": ["cat", "studio"],
                    "negatives": [{"media": "positive", "caption": "negative two a"}],
                    "anchors": [],
                },
            },
        ],
    )

    positive_caption_calls: list[dict] = []
    negative_caption_task_calls: list[dict] = []
    synthetic_generate_calls: list[dict] = []

    monkeypatch.setattr(
        process_dataset_script,
        "compute_captions_embeddings",
        lambda **kwargs: positive_caption_calls.append(kwargs),
    )

    def fake_compute_caption_embeddings_from_tasks(**kwargs):
        negative_caption_task_calls.append(kwargs)

    monkeypatch.setattr(
        process_dataset_script,
        "compute_caption_embeddings_from_tasks",
        fake_compute_caption_embeddings_from_tasks,
    )

    def fake_compute_latents(**kwargs):
        output_dir = Path(kwargs["output_dir"])
        for rel_path in ("videos/one.pt", "videos/two.pt"):
            path = output_dir / rel_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"latent")

    monkeypatch.setattr(process_dataset_script, "compute_latents", fake_compute_latents)
    monkeypatch.setattr(
        process_dataset_script,
        "generate_negative_latents",
        lambda specs, **kwargs: synthetic_generate_calls.append({"specs": specs, **kwargs}),
    )
    monkeypatch.setattr(
        process_dataset_script,
        "generate_missing_negative_latents",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("legacy negative generation should not run")
        ),
    )

    process_dataset_script.preprocess_dataset(
        dataset_file=str(dataset_path),
        caption_column="caption",
        video_column="media_path",
        resolution_buckets=[(9, 544, 960)],
        batch_size=1,
        output_dir=str(tmp_path / ".precomputed"),
        lora_trigger=None,
        vae_tiling=False,
        decode=False,
        model_path="/tmp/model.safetensors",
        text_encoder_path="/tmp/gemma",
        device="cpu",
        negative_i2v_mode=True,
    )

    assert len(positive_caption_calls) == 1
    assert len(negative_caption_task_calls) == 1
    assert len(negative_caption_task_calls[0]["tasks"]) == 3
    assert len(synthetic_generate_calls) == 1
    assert len(synthetic_generate_calls[0]["specs"]) == 1
    assert synthetic_generate_calls[0]["specs"][0].positive_media_path == "videos/one.mp4"
    assert synthetic_generate_calls[0]["specs"][0].prompt == "synthetic negative one b prompt"
    assert synthetic_generate_calls[0]["inference_steps"] == 40
    assert synthetic_generate_calls[0]["use_first_frame_conditioning"] is True
    assert (tmp_path / ".precomputed" / "nsync_manifest.json").is_file()


def test_preprocess_dataset_builds_advanced_nsync_manifest_for_image_datasets(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset_path = _write_dataset(
        tmp_path,
        [
            {
                "caption": "positive one",
                "media_path": "images/one.png",
                "nsync": {
                    "categories": ["cat", "cinematic"],
                    "negatives": [
                        {"media": "positive", "caption": "negative one a"},
                        {
                            "media": "synthetic",
                            "prompt": "synthetic negative one b prompt",
                            "caption": "negative one b",
                        },
                    ],
                    "anchors": [{"required_categories": ["cat"], "extra_random_category": True}],
                },
            },
            {
                "caption": "positive two",
                "media_path": "images/two.png",
                "nsync": {
                    "categories": ["cat", "studio"],
                    "negatives": [{"media": "positive", "caption": "negative two a"}],
                    "anchors": [],
                },
            },
        ],
    )

    negative_caption_task_calls: list[dict] = []
    synthetic_generate_calls: list[dict] = []

    monkeypatch.setattr(process_dataset_script, "compute_captions_embeddings", lambda **_: None)
    monkeypatch.setattr(
        process_dataset_script,
        "compute_caption_embeddings_from_tasks",
        lambda **kwargs: negative_caption_task_calls.append(kwargs),
    )

    def fake_compute_latents(**kwargs):
        output_dir = Path(kwargs["output_dir"])
        for rel_path in ("images/one.pt", "images/two.pt"):
            path = output_dir / rel_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"latent")

    monkeypatch.setattr(process_dataset_script, "compute_latents", fake_compute_latents)
    monkeypatch.setattr(
        process_dataset_script,
        "generate_negative_latents",
        lambda specs, **kwargs: synthetic_generate_calls.append({"specs": specs, **kwargs}),
    )
    monkeypatch.setattr(
        process_dataset_script,
        "generate_missing_negative_latents",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("legacy negative generation should not run")
        ),
    )

    process_dataset_script.preprocess_dataset(
        dataset_file=str(dataset_path),
        caption_column="caption",
        video_column="media_path",
        resolution_buckets=[(1, 544, 960)],
        batch_size=1,
        output_dir=str(tmp_path / ".precomputed"),
        lora_trigger=None,
        vae_tiling=False,
        decode=False,
        model_path="/tmp/model.safetensors",
        text_encoder_path="/tmp/gemma",
        device="cpu",
    )

    assert len(negative_caption_task_calls) == 1
    assert len(negative_caption_task_calls[0]["tasks"]) == 3
    assert len(synthetic_generate_calls) == 1
    assert synthetic_generate_calls[0]["specs"][0].positive_media_path == "images/one.png"
    assert (
        synthetic_generate_calls[0]["specs"][0].output_rel_path
        == "images/one__nsync_negative_001.pt"
    )
    assert (tmp_path / ".precomputed" / "nsync_manifest.json").is_file()
