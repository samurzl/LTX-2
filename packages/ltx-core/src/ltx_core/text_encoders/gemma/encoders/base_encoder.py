import functools
from pathlib import Path

import torch
from transformers import AutoImageProcessor, Gemma3ForConditionalGeneration, Gemma3Processor

from ltx_core.loader.module_ops import ModuleOps
from ltx_core.text_encoders.gemma.tokenizer import LTXVGemmaTokenizer
from ltx_core.utils import find_matching_file

DEFAULT_GEMMA_ASSET_SOURCE = "google/gemma-3-12b-it"


class GemmaTextEncoder(torch.nn.Module):
    """Pure Gemma text encoder — runs the LLM and returns raw hidden states.
    Prompt enhancement (generate) is also supported since the full
    Gemma3ForConditionalGeneration model (including lm_head) is loaded.
    """

    def __init__(
        self,
        model: Gemma3ForConditionalGeneration | None = None,
        tokenizer: LTXVGemmaTokenizer | None = None,
        processor: Gemma3Processor | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self._dtype = dtype

    def encode(
        self,
        text: str,
        padding_side: str = "left",
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        """Run Gemma LLM and return raw hidden states + attention mask.
        Calls the inner model (self.model.model) to skip lm_head logits computation (~500 MiB saving).
        Returns:
            (hidden_states, attention_mask) where hidden_states is a tuple of per-layer tensors.
        """
        return self.encode_batch([text], padding_side=padding_side)

    def encode_batch(
        self,
        texts: list[str],
        padding_side: str = "left",  # noqa: ARG002
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        """Run the Gemma backbone once for a padded batch of prompts."""
        device = _language_model_device(self.model)
        input_ids, attention_mask = self.tokenizer.tokenize_batch(texts)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        outputs = self.model.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        del outputs
        return hidden_states, attention_mask

    # --- Prompt enhancement methods ---

    def _enhance(
        self,
        messages: list[dict[str, str]],
        image: torch.Tensor | None = None,
        max_new_tokens: int = 512,
        seed: int = 10,
    ) -> str:
        text = self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        model_inputs = self.processor(
            text=text,
            images=image,
            return_tensors="pt",
        ).to(self.model.device)
        pad_token_id = self.processor.tokenizer.pad_token_id if self.processor.tokenizer.pad_token_id is not None else 0
        model_inputs = _pad_inputs_for_attention_alignment(model_inputs, pad_token_id=pad_token_id)

        with torch.inference_mode(), torch.random.fork_rng(devices=[self.model.device]):
            torch.manual_seed(seed)
            outputs = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
            )
            generated_ids = outputs[0][len(model_inputs.input_ids[0]) :]
            enhanced_prompt = self.processor.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return enhanced_prompt

    def enhance_t2v(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        system_prompt: str | None = None,
        seed: int = 10,
    ) -> str:
        """Enhance a text prompt for T2V generation."""
        system_prompt = system_prompt or self.default_gemma_t2v_system_prompt

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"user prompt: {prompt}"},
        ]

        return self._enhance(messages, max_new_tokens=max_new_tokens, seed=seed)

    def enhance_i2v(
        self,
        prompt: str,
        image: torch.Tensor,
        max_new_tokens: int = 512,
        system_prompt: str | None = None,
        seed: int = 10,
    ) -> str:
        """Enhance a text prompt for I2V generation using a reference image."""
        system_prompt = system_prompt or self.default_gemma_i2v_system_prompt
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"User Raw Input Prompt: {prompt}."},
                ],
            },
        ]
        return self._enhance(messages, image=image, max_new_tokens=max_new_tokens, seed=seed)

    @functools.cached_property
    def default_gemma_i2v_system_prompt(self) -> str:
        return _load_system_prompt("gemma_i2v_system_prompt.txt")

    @functools.cached_property
    def default_gemma_t2v_system_prompt(self) -> str:
        return _load_system_prompt("gemma_t2v_system_prompt.txt")


# --- Standalone utility functions ---


@functools.lru_cache(maxsize=2)
def _load_system_prompt(prompt_name: str) -> str:
    with open(Path(__file__).parent / "prompts" / f"{prompt_name}", "r") as f:
        return f.read()


def _language_model_device(model: Gemma3ForConditionalGeneration) -> torch.device:
    language_model = model.model.language_model
    for tensor in language_model.parameters():
        if tensor.device.type != "meta":
            return tensor.device
    for tensor in language_model.buffers():
        if tensor.device.type != "meta":
            return tensor.device

    fallback_device = model.device
    if fallback_device.type != "meta":
        return fallback_device

    raise RuntimeError(
        "Gemma language weights are still on the meta device. This usually means the Gemma "
        "safetensors file did not match any supported native key layout."
    )


def _cat_with_padding(
    tensor: torch.Tensor,
    padding_length: int,
    value: int | float,
) -> torch.Tensor:
    """Concatenate a tensor with a padding tensor of the given value."""
    return torch.cat(
        [
            tensor,
            torch.full(
                (1, padding_length),
                value,
                dtype=tensor.dtype,
                device=tensor.device,
            ),
        ],
        dim=1,
    )


def _pad_inputs_for_attention_alignment(
    model_inputs: dict[str, torch.Tensor],
    pad_token_id: int = 0,
    alignment: int = 8,
) -> dict[str, torch.Tensor]:
    """Pad sequence length to multiple of alignment for Flash Attention compatibility."""
    seq_len = model_inputs.input_ids.shape[1]
    padded_len = ((seq_len + alignment - 1) // alignment) * alignment
    padding_length = padded_len - seq_len

    if padding_length > 0:
        model_inputs["input_ids"] = _cat_with_padding(model_inputs.input_ids, padding_length, pad_token_id)
        model_inputs["attention_mask"] = _cat_with_padding(model_inputs.attention_mask, padding_length, 0)
        if "token_type_ids" in model_inputs and model_inputs["token_type_ids"] is not None:
            model_inputs["token_type_ids"] = _cat_with_padding(model_inputs["token_type_ids"], padding_length, 0)

    return model_inputs


def gemma_weight_paths_from_source(gemma_source: str | Path) -> tuple[str, ...]:
    """Resolve Gemma weights from either a directory or a single safetensors file."""
    source_path = Path(gemma_source)
    if source_path.is_file():
        if source_path.suffix != ".safetensors":
            raise ValueError(f"Gemma weight file must be a .safetensors file: {gemma_source}")
        return (str(source_path),)

    if source_path.is_dir():
        model_folder = find_matching_file(str(source_path), "model*.safetensors").parent
        return tuple(str(path) for path in sorted(model_folder.rglob("*.safetensors")))

    raise FileNotFoundError(f"Gemma source must be a directory or a single .safetensors file: {gemma_source}")


def _find_optional_asset_root(gemma_source: str | Path, pattern: str) -> str | None:
    source_path = Path(gemma_source)
    if source_path.is_file():
        source_path = source_path.parent
    if not source_path.is_dir():
        return None

    matches = sorted(source_path.rglob(pattern))
    if not matches:
        return None
    return str(matches[0].parent)


def _resolve_gemma_asset_source(
    gemma_source: str | Path,
    pattern: str,
    *,
    allow_default_assets: bool,
) -> tuple[str, bool]:
    local_root = _find_optional_asset_root(gemma_source, pattern)
    if local_root is not None:
        return local_root, True

    if allow_default_assets:
        return DEFAULT_GEMMA_ASSET_SOURCE, False

    raise FileNotFoundError(
        f"No files matching pattern '{pattern}' found under {gemma_source}. "
        "A single Gemma safetensors file contains weights only; tokenizer/processor "
        "assets must be next to it, in the Gemma directory, or loadable from the default "
        f"source '{DEFAULT_GEMMA_ASSET_SOURCE}'."
    )


def module_ops_from_gemma_source(
    gemma_source: str | Path,
    *,
    include_processor: bool = True,
    allow_default_assets: bool = True,
) -> tuple[ModuleOps, ...]:
    tokenizer_source, tokenizer_local_only = _resolve_gemma_asset_source(
        gemma_source,
        "tokenizer.model",
        allow_default_assets=allow_default_assets,
    )
    processor_source = None
    processor_local_only = True
    if include_processor:
        processor_source, processor_local_only = _resolve_gemma_asset_source(
            gemma_source,
            "preprocessor_config.json",
            allow_default_assets=allow_default_assets,
        )

    def load_tokenizer(module: GemmaTextEncoder) -> GemmaTextEncoder:
        module.tokenizer = LTXVGemmaTokenizer(tokenizer_source, 1024, local_files_only=tokenizer_local_only)
        return module

    def load_processor(module: GemmaTextEncoder) -> GemmaTextEncoder:
        if processor_source is None:
            raise ValueError("Processor source was not resolved")
        image_processor = AutoImageProcessor.from_pretrained(
            processor_source,
            local_files_only=processor_local_only,
            use_fast=False,
        )
        if not module.tokenizer:
            raise ValueError("Tokenizer model operation must be performed before processor model operation")
        module.processor = Gemma3Processor(image_processor=image_processor, tokenizer=module.tokenizer.tokenizer)
        return module

    tokenizer_load_ops = ModuleOps(
        "TokenizerLoad",
        matcher=lambda module: isinstance(module, GemmaTextEncoder) and module.tokenizer is None,
        mutator=load_tokenizer,
    )
    processor_load_ops = ModuleOps(
        "ProcessorLoad",
        matcher=lambda module: isinstance(module, GemmaTextEncoder) and module.processor is None,
        mutator=load_processor,
    )
    if include_processor:
        return (tokenizer_load_ops, processor_load_ops)
    return (tokenizer_load_ops,)


def module_ops_from_gemma_root(gemma_root: str) -> tuple[ModuleOps, ...]:
    return module_ops_from_gemma_source(gemma_root, allow_default_assets=False)
