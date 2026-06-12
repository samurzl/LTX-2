from __future__ import annotations

import json
from pathlib import Path


WORKFLOW = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "ltx_trainer"
    / "comfyui_workflows"
    / "ltx_2_3_i2v_multi_expert_loras.json"
)


def _load_workflow() -> tuple[dict, dict]:
    workflow = json.loads(WORKFLOW.read_text())
    return workflow, workflow["definitions"]["subgraphs"][0]


def test_ltx_2_3_i2v_expert_lora_workflow_removes_prompt_enhancement() -> None:
    _, subgraph = _load_workflow()
    node_types = {node["type"] for node in subgraph["nodes"]}

    assert "TextGenerateLTX2Prompt" not in node_types
    assert "PreviewAny" not in node_types
    assert "LoraLoader" not in node_types

    links = {link["id"]: link for link in subgraph["links"]}
    assert links[619] == {
        "id": 619,
        "origin_id": 266,
        "origin_slot": 0,
        "target_id": 240,
        "target_slot": 1,
        "type": "STRING",
    }


def test_ltx_2_3_i2v_expert_lora_workflow_uses_fp8_48gb_defaults() -> None:
    _, subgraph = _load_workflow()
    nodes = {node["id"]: node for node in subgraph["nodes"]}

    assert nodes[236]["widgets_values"] == ["ltx-2.3-22b-dev-fp8.safetensors"]
    assert nodes[243]["widgets_values"][0] == "gemma_3_12B_it_fp4_mixed.safetensors"
    assert nodes[257]["widgets_values"][0] == 768
    assert nodes[258]["widgets_values"][0] == 512
    assert nodes[225]["widgets_values"][0] == 49
    assert nodes[260]["widgets_values"][0] == 24
    assert nodes[228]["widgets_values"] == [384, 256, 49, 1]
    assert nodes[251]["widgets_values"][0] == 512


def test_ltx_2_3_i2v_expert_lora_workflow_chains_two_model_only_experts() -> None:
    workflow, subgraph = _load_workflow()
    nodes = {node["id"]: node for node in subgraph["nodes"]}
    links = {link["id"]: link for link in subgraph["links"]}

    assert nodes[276]["type"] == "LoraLoaderModelOnly"
    assert nodes[276]["title"] == "Expert LoRA 1"
    assert nodes[276]["widgets_values"] == ["lora_weights_low_noise_step_02000.safetensors", 1.0]
    assert nodes[277]["type"] == "LoraLoaderModelOnly"
    assert nodes[277]["title"] == "Expert LoRA 2"
    assert nodes[277]["widgets_values"] == ["lora_weights_high_noise_step_02000.safetensors", 1.0]

    assert links[520]["origin_id"] == 236
    assert links[520]["target_id"] == 232
    assert links[625]["origin_id"] == 232
    assert links[625]["target_id"] == 276
    assert links[626]["origin_id"] == 276
    assert links[626]["target_id"] == 277
    assert links[478]["origin_id"] == 277
    assert links[478]["target_id"] == 213
    assert links[541]["origin_id"] == 277
    assert links[541]["target_id"] == 231

    top_node = next(node for node in workflow["nodes"] if node["id"] == 267)
    proxy_widgets = top_node["properties"]["proxyWidgets"]
    assert ["276", "lora_name"] in proxy_widgets
    assert ["276", "strength_model"] in proxy_widgets
    assert ["277", "lora_name"] in proxy_widgets
    assert ["277", "strength_model"] in proxy_widgets
