"""Test that get_layer_count returns correct values for known models."""

import pytest

from nl_probes.utils.common import get_layer_count

# Known layer counts to verify against
KNOWN_LAYER_COUNTS = {
    "Qwen/Qwen3-1.7B": 28,
    "Qwen/Qwen3-8B": 36,
    "Qwen/Qwen3-14B": 40,
    "Qwen/Qwen3-32B": 64,
    "google/gemma-2-9b-it": 42,
    "google/gemma-2-27b-it": 46,
    "google/gemma-3-1b-it": 26,
    "google/gemma-3-4b-it": 34,
    "google/gemma-3-12b-it": 48,
    "google/gemma-3-27b-it": 62,
    "meta-llama/Llama-3.1-8B-Instruct": 32,
    "meta-llama/Llama-3.2-1B-Instruct": 16,
    "meta-llama/Llama-3.3-70B-Instruct": 80,
}


@pytest.mark.parametrize("model_name,expected_layers", KNOWN_LAYER_COUNTS.items())
def test_get_layer_count(model_name: str, expected_layers: int):
    """Verify get_layer_count matches known layer counts from HuggingFace configs."""
    actual_layers = get_layer_count(model_name)
    assert actual_layers == expected_layers, f"{model_name}: expected {expected_layers}, got {actual_layers}"
