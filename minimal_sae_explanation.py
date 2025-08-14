#!/usr/bin/env python3
"""
Minimal SAE Feature Explanation Script

This script generates self-explanations for sparse autoencoder features using
activation steering with the Gemma-2-9B-IT model.
"""

import torch
import contextlib
from typing import Callable
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer

import interp_tools.saes.jumprelu_sae as jumprelu_sae
import interp_tools.model_utils as model_utils


def build_explanation_prompt(
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    """
    Constructs a prompt for generating SAE feature explanations.

    Returns:
        A tuple containing the tokenized input IDs and the position of the 'X'
        placeholder where activations should be steered.
    """
    # Create chat format messages
    messages = [
        {
            "role": "user",
            "content": "Can you write explain to me what 'X' means? Format your final answer with <explanation>",
        },
    ]

    # Apply chat template
    formatted_input = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        continue_final_message=False,
    )
    print(f"Formatted input: {formatted_input}")

    # Find the position of the placeholder 'X'
    token_ids = tokenizer.encode(str(formatted_input), add_special_tokens=False)
    x_token_ids = tokenizer.encode("X", add_special_tokens=False)
    assert len(x_token_ids) == 1, "Expected to find 1 'X' token"
    x_token_id = x_token_ids[0]
    positions = [i for i, token_id in enumerate(token_ids) if token_id == x_token_id]

    assert len(positions) == 1, (
        f"Expected to find 1 'X' placeholder, but found {len(positions)}."
    )

    tokenized_input = tokenizer(
        str(formatted_input), return_tensors="pt", add_special_tokens=False
    ).to(device)

    return tokenized_input.input_ids, positions[0]


@contextlib.contextmanager
def add_hook(module: torch.nn.Module, hook: Callable):
    """Temporarily adds a forward hook to a model module."""
    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def get_activation_steering_hook(
    feature_vector: torch.Tensor,
    position: int,
    steering_coefficient: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Callable:
    """
    Returns a forward hook that replaces specified residual-stream activations
    during the initial prompt pass of model.generate.
    """
    feature_vector = feature_vector.to(device, dtype)

    def hook_fn(module, _input, output):
        resid_BLD, *rest = output  # Gemma returns (resid, hidden_states, ...)
        L = resid_BLD.shape[1]
        # assert batch size is 1
        assert resid_BLD.shape[0] == 1, "Batch size must be 1"

        # Only touch the *prompt* forward pass (sequence length > 1)
        if L <= 1:
            return (resid_BLD, *rest)

        # Safety: make sure position is inside current sequence
        if position >= L:
            raise IndexError(f"position {position} is out of bounds for length {L}")

        # Get norm of original activation at the target position
        orig_activation = resid_BLD[0, position]  # Single batch
        orig_norm = orig_activation.norm()

        print(f"Position of X: {position}")
        print(f"Original activation: {orig_activation}")
        print(f"Original norm: {orig_norm}")
        print(f"Steering coefficient: {steering_coefficient}")
        print(f"Feature vector: {feature_vector}")
        # Build steered vector
        steered_vector = (
            torch.nn.functional.normalize(feature_vector, dim=-1)
            * orig_norm
            * steering_coefficient
        )
        print(f"Steered vector: {steered_vector}")
        # shape
        print(f"Steered vector shape: {steered_vector.shape}")

        # Replace activation
        resid_BLD[0, position] = steered_vector

        return (resid_BLD, *rest)

    return hook_fn


def main(
    sae_index: int = 0,
    steering_coefficient: float = 2.0,
    layer: int = 9,
    num_generations: int = 10,
):
    """
    Main function to generate SAE feature explanations.

    Args:
        sae_index: Index of the SAE feature to explain
        steering_coefficient: Strength of activation steering
        layer: Model layer to apply steering to
        num_generations: Number of explanations to generate
    """
    print(f"Generating {num_generations} explanations for SAE feature {sae_index}")
    print(f"Using steering coefficient: {steering_coefficient}, layer: {layer}")

    # Setup
    model_name = "google/gemma-2-9b-it"
    dtype = torch.bfloat16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=dtype
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load SAE
    print(f"Loading SAE for layer {layer}...")
    sae_repo_id = "google/gemma-scope-9b-it-res"
    sae_filename = f"layer_{layer}/width_16k/average_l0_88/params.npz"

    sae = jumprelu_sae.load_gemma_scope_jumprelu_sae(
        repo_id=sae_repo_id,
        filename=sae_filename,
        layer=layer,
        model_name=model_name,
        device=device,
        dtype=dtype,
    )

    # Get the model submodule for the specified layer
    submodule = model_utils.get_submodule(model, layer)

    # Build prompt
    input_ids, x_position = build_explanation_prompt(tokenizer, device)

    # Get feature vector (using decoder weights)
    feature_vector = sae.W_dec[sae_index]
    print(f"Feature vector shape: {feature_vector.shape}")
    print(f"Feature vector: {feature_vector}")

    # Create steering hook
    hook_fn = get_activation_steering_hook(
        feature_vector=feature_vector,
        position=x_position,
        steering_coefficient=steering_coefficient,
        device=device,
        dtype=dtype,
    )

    # Generation settings
    generation_kwargs = {
        "do_sample": True,
        "temperature": 1.0,
        "max_new_tokens": 200,
        "pad_token_id": tokenizer.eos_token_id,
    }

    # Generate explanations
    print(f"\nGenerating {num_generations} explanations...")
    explanations = []

    for i in range(num_generations):
        print(f"\nGeneration {i + 1}/{num_generations}:")

        with add_hook(submodule, hook_fn):
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                **generation_kwargs,
            )

        # Decode only the newly generated tokens
        generated_tokens = output_ids[:, input_ids.shape[1] :]
        decoded_output = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

        explanations.append(decoded_output)
        print(decoded_output)
        print("-" * 80)

    return explanations


if __name__ == "__main__":
    # Example usage
    explanations = main(
        sae_index=0,
        steering_coefficient=20.0,
        layer=9,
        num_generations=10,
    )

    print(f"\nGenerated {len(explanations)} explanations total.")
