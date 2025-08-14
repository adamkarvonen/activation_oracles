# ==============================================================================
# 0. IMPORTS AND SETUP
# ==============================================================================
import torch
import contextlib
from typing import Callable, Dict, Optional, Any
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
from dataclasses import dataclass, field, asdict
import einops
from rapidfuzz.distance import Levenshtein as lev
import typer

import interp_tools.saes.jumprelu_sae as jumprelu_sae
import interp_tools.model_utils as model_utils


def hardcoded_gemma_2_9b_it_few_shot_example(model_name: str) -> list[dict]:
    assert model_name == "google/gemma-2-9b-it"

    demo_features: list[dict] = [
        {
            "feature_idx": 1835,
            # "original_sentence": "I traveled back in time.",
            # "rewritten_sentence": "I traveled to Paris.",
            "explanation": "The word relates to concepts of time travel or moving through time.",
        },
        {
            "feature_idx": 5318,
            # "original_sentence": "What do we know?",
            # "rewritten_sentence": "We know everything.",
            "explanation": "The word relates to inquiry, questioning, or uncertainty.",
        },
        {
            "feature_idx": 6941,
            # "original_sentence": "I see a dog.",
            # "rewritten_sentence": "I see a fish.",
            "explanation": "The word relates to concepts of animals or pets, especially dogs.",
        },
    ]

    return demo_features


# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================


@dataclass
class SelfInterpConfig:
    """Configuration settings for the script."""

    # --- Foundational Settings ---
    model_name: str = "google/gemma-2-9b-it"
    batch_size: int = 4

    # --- SAE (Sparse Autoencoder) Settings ---
    sae_repo_id: str = "google/gemma-scope-9b-it-res"
    sae_layer: int = 9
    sae_width: int = 16  # For loading the correct max acts file
    layer_percent: int = 25  # For loading the correct max acts file
    context_length: int = 32
    test_set_size: int = 1000

    sae_filename: str = field(init=False)

    # --- Experiment Settings ---
    num_features_to_run: int = 200  # How many random features to analyze

    random_seed: int = 42  # For reproducible feature selection
    results_filename: str = "self_interp_results.pkl"
    use_decoder_vectors: bool = True

    # Use a default_factory for mutable types like dicts
    generation_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {
            "do_sample": False,
            "temperature": 0.0,
            "max_new_tokens": 200,
        }
    )

    features_to_explain: list[int] = field(default_factory=list)
    steering_coefficient: float = 2.0

    def __post_init__(self):
        """Called after the dataclass is initialized."""
        self.sae_filename = f"layer_{self.sae_layer}/width_16k/average_l0_88/params.npz"


# ==============================================================================
# 2. UTILITY FUNCTIONS
# ==============================================================================


def build_few_shot_prompt(
    few_shot_examples: list[dict],
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
) -> tuple[torch.Tensor, list[int]]:
    """
    Constructs a few-shot prompt for generating feature explanations.

    Args:
        demo_features: A dictionary mapping feature indices to their demo explanations.
        target_feature_idx: The index of the new feature to explain.
        tokenizer: The model's tokenizer.
        device: The torch device to place tensors on.

    Returns:
        A tuple containing the tokenized input IDs and the positions of the 'X'
        placeholders where activations should be steered.
    """
    question = "Can you write me a sentence that relates to the word 'X' and a similar sentence that does not relate to the word?"

    messages = []
    for example in few_shot_examples:
        # pos_sent = example["original_sentence"]
        # neg_sent = example["rewritten_sentence"]
        explanation = example["explanation"]

        messages.extend(
            [
                {"role": "user", "content": question},
                {
                    "role": "assistant",
                    "content": f"Explanation: {explanation}",
                },
            ]
        )

    # Add the final prompt for the target feature
    messages.extend(
        [
            {"role": "user", "content": question},
            {
                "role": "assistant",
                "content": "Explanation:",
            },
        ]
    )

    formatted_input = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        continue_final_message=True,
    )

    # Find the positions of the placeholder 'X'
    token_ids = tokenizer.encode(formatted_input, add_special_tokens=False)
    x_token_id = tokenizer.encode("X", add_special_tokens=False)[0]
    positions = [i for i, token_id in enumerate(token_ids) if token_id == x_token_id]

    # Ensure we found a placeholder for each demo and the final target
    expected_positions = len(few_shot_examples) + 1
    assert len(positions) == expected_positions, (
        f"Expected to find {expected_positions} 'X' placeholders, but found {len(positions)}."
    )

    tokenized_input = tokenizer(
        formatted_input, return_tensors="pt", add_special_tokens=False
    ).to(device)

    return tokenized_input.input_ids, positions


def parse_generated_explanation(text: str) -> Optional[dict[str, str]]:
    """
    Extract the positive example, negative example, and explanation
    from a model-generated block of text formatted as:
    If any of those tags is missing, return None.
    """
    # Normalise leading / trailing whitespace
    text = text.strip()


    return {
        # "positive_sentence": positive.strip(),
        # "negative_sentence": negative.strip(),
        "explanation": text,
    }


def main(
    sae_index: int = typer.Option(7159, help="Feature index to explain"),
    steering_coefficient: float = typer.Option(2.0, help="Coefficient for activation steering"),
    layer: int = typer.Option(9, help="Layer number for SAE"),
    num_generations: int = typer.Option(5, help="Number of features to generate explanations for"),
):
    """Main function for few-shot SAE explanation generation."""
    # Main script logic
    cfg = SelfInterpConfig()
    
    # Set up variables from parameters
    features_to_explain = [sae_index] * num_generations  # Repeat sae_index num_generations times
    sae_layer = layer
    batch_size = 8
    verbose = True
    
    print(asdict(cfg))
    dtype = torch.bfloat16
    device = torch.device("cuda")
    
    
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, device_map="auto", torch_dtype=dtype
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    
    print(f"Loading SAE for layer {sae_layer} from {cfg.sae_repo_id}...")
    sae = jumprelu_sae.load_gemma_scope_jumprelu_sae(
        repo_id=cfg.sae_repo_id,
        filename=f"layer_{sae_layer}/width_16k/average_l0_88/params.npz",
        layer=sae_layer,
        model_name=cfg.model_name,
        device=device,
        dtype=dtype,
    )
    
    # nn.module for the sae layer, add the pytorch hook to it
    submodule = model_utils.get_submodule(model, sae_layer)

    # Load few-shot examples
    few_shot_examples = hardcoded_gemma_2_9b_it_few_shot_example(cfg.model_name)
    
    
    # I created these examples by hand.
    
    print(few_shot_examples[0])
    
    # Prepare positive and negative examples for analysis
    # pos_input_strs = [example["original_sentence"] for example in few_shot_examples]
    # neg_input_strs = [example["rewritten_sentence"] for example in few_shot_examples]
    
    # tokenized_pos_strs = tokenizer(
    #     pos_input_strs, return_tensors="pt", add_special_tokens=True, padding=True
    # ).to(device)
    # tokenized_neg_strs = tokenizer(
    #     neg_input_strs, return_tensors="pt", add_special_tokens=True, padding=True
    # ).to(device)
    
    # for i in range(len(pos_input_strs)):
    #     print(f"Pos sentence {i}: {pos_input_strs[i]}")
    #     print(f"Neg sentence {i}: {neg_input_strs[i]}")
    #     print("-" * 100)

    # Get feature activations for positive and negative examples
    # pos_acts_BLF = get_feature_activations(
    #     model=model,
    #     tokenizer=tokenizer,
    #     submodule=submodule,
    #     sae=sae,
    #     tokenized_strs=tokenized_pos_strs,
    # )
    
    # neg_acts_BLF = get_feature_activations(
    #     model=model,
    #     tokenizer=tokenizer,
    #     submodule=submodule,
    #     sae=sae,
    #     tokenized_strs=tokenized_neg_strs,
    # )
    
    # print(pos_acts_BLF.shape)
    # print(neg_acts_BLF.shape)
    
    
    # time_travel_feature = few_shot_examples[0]["feature_idx"]
    # print(f"Time travel feature: {time_travel_feature}")
    # you can also check out feature activations here: https://www.neuronpedia.org/gemma-2-9b-it/9-gemmascope-res-16k/1835
    # NOTE: Here I'm using the 16k width SAE, not the 131k width SAE.
    
    # pos_feature_acts_L = pos_acts_BLF[0, :, time_travel_feature]
    # neg_feature_acts_L = neg_acts_BLF[0, :, time_travel_feature]
    
    # pos_feature_acts_L = pos_feature_acts_L[1:]  # remove the BOS token
    # neg_feature_acts_L = neg_feature_acts_L[1:]  # remove the BOS token
    
    # print(
    #     "As we can see, the feature is far more active in the positive sentence than the negative sentence."
    # )
    # print(pos_feature_acts_L)
    # print(neg_feature_acts_L)

    # Build few-shot prompt for generation
    orig_input_ids, orig_positions = build_few_shot_prompt(
        few_shot_examples, tokenizer, device
    )
    orig_input_ids = orig_input_ids.squeeze()
    few_shot_prompt = tokenizer.decode(orig_input_ids)
    
    print("Now we turn our examples into a few-shot prompt.")
    print(few_shot_prompt)
    
    # Prepare feature indices and batch data
    few_shot_indices = [example["feature_idx"] for example in few_shot_examples]
    
    # input_ids, positions, few_shot_indices = hardcoded_gemma_2_9b_it_few_shot_example(
    #     cfg.model_name, tokenizer, device
    # )
    
    # Use the feature specified by sae_index parameter, repeated num_generations times
    num_features_to_run = num_generations
    
    assert len(features_to_explain) == num_features_to_run

    # Prepare batch steering vectors for each feature
    batch_steering_vectors = []
    batch_feature_indices = []
    batch_positions = []
    for i in range(num_features_to_run):
        target_feature_idx = features_to_explain[i]
        batch_feature_indices.append(target_feature_idx)
        print(f"\n{'=' * 20} CONSTRUCTING DATA FOR FEATURE {target_feature_idx} {'=' * 20}")
    
        # 2. Prepare feature vectors for steering
        # We use decoder weights (W_dec) as they map from the feature space back to the residual stream.
        all_feature_indices = few_shot_indices + [target_feature_idx]
    
        if cfg.use_decoder_vectors:
            feature_vectors = [sae.W_dec[i] for i in all_feature_indices]
        else:
            feature_vectors = [sae.W_enc[:, i] for i in all_feature_indices]
    
        batch_steering_vectors.append(feature_vectors)
        batch_positions.append(orig_positions)

    # As we can see, the input_ids and attention_mask are the same for each example in the batch.
    # the only difference is the feature vector we're steering.
    
    input_ids_BL = einops.repeat(orig_input_ids, "L -> B L", B=len(batch_steering_vectors))
    attn_mask_BL = torch.ones_like(input_ids_BL, dtype=torch.bool).to(device)
    
    tokenized_input = {
        "input_ids": input_ids_BL,
        "attention_mask": attn_mask_BL,
    }

    # With the exception of the few shot examples, this is what a training batch looks like. 
    # Identical prompts, identical x token positions, different feature vectors and output sentences.
    
    # Now we just replace the model activations at the position of 'X' and layer 9 with the feature vector we want to steer.
    # We do this by adding a forward hook to the model.

    # 3. Create and apply the activation steering hook
    hook_fn = get_activation_steering_hook(
        vectors=batch_steering_vectors,
        positions=batch_positions,
        steering_coefficient=steering_coefficient,
        device=device,
        dtype=dtype,
    )
    
    # 4. Generate the explanation with activation steering
    print("Generating explanation with activation steering...")
    
    cfg.generation_kwargs["do_sample"] = True
    cfg.generation_kwargs["temperature"] = 0.7
    
   
    with add_hook(submodule, hook_fn):
        output_ids = model.generate(**tokenized_input, **cfg.generation_kwargs)
    
    # Decode only the newly generated tokens
    generated_tokens = output_ids[:, input_ids_BL.shape[1] :]
    decoded_output = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    
    for i in range(len(decoded_output)):
        print(f"Generated output {i}: {decoded_output[i]}")
        print("-" * 100)

    # Parse and analyze the generated explanations
    pos_statements = []
    neg_statements = []
    explanations = []
    
    print(decoded_output)
    
    for i, output in enumerate(decoded_output):
        # Note: we add the "Positive example:" prefix to the positive sentence
        # because we prefill the model's response with "Positive example:"
        parsed_result = parse_generated_explanation(f"Positive example:{output}")
        print(f"Parsed result: {parsed_result}")
        if parsed_result:
            # pos_statements.append(parsed_result["positive_sentence"])
            # neg_statements.append(parsed_result["negative_sentence"])
            explanations.append(parsed_result["explanation"])
        else:
            # pos_statements.append("")
            # neg_statements.append("")
            explanations.append("")

    for explanation in explanations:
        print(explanation)
        print("-" * 10)
    

@contextlib.contextmanager
def add_hook(module: torch.nn.Module, hook: Callable):
    """Temporarily adds a forward hook to a model module."""
    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def get_activation_steering_hook(
    vectors: list[list[torch.Tensor]],  # [B][K, d_model]  or [K, d_model] if B==1
    positions: list[list[int]],  # [B][K]
    steering_coefficient: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Callable:
    """
    Returns a forward hook that *replaces* specified residual-stream activations
    during the initial prompt pass of `model.generate`.

    • vectors[b][k]  – feature vector to inject for batch b, slot k
    • coeffs[b][k]   – scale factor (usually small, e.g. 0.3)
    • positions[b][k]– token index (0-based, within prompt only)
    """

    # ---- pack Python lists → torch tensors once, outside the hook ----
    vec_BKD = torch.stack([torch.stack(v) for v in vectors])  # (B, K, d)
    pos_BK = torch.tensor(positions, dtype=torch.long)  # (B, K)

    B, K, d_model = vec_BKD.shape
    assert pos_BK.shape == (B, K)

    vec_BKD = vec_BKD.to(device, dtype)
    pos_BK = pos_BK.to(device)

    def hook_fn(module, _input, output):
        resid_BLD, *rest = output  # Gemma returns (resid, hidden_states, ...)
        L = resid_BLD.shape[1]

        # Only touch the *prompt* forward pass (sequence length > 1)
        if L <= 1:
            return (resid_BLD, *rest)

        # Safety: make sure every position is inside current sequence
        if (pos_BK >= L).any():
            bad = pos_BK[pos_BK >= L].min().item()
            raise IndexError(f"position {bad} is out of bounds for length {L}")

        # ---- compute norms of original activations at the target slots ----
        batch_idx_B1 = torch.arange(B, device=device).unsqueeze(1)  # (B, 1) → (B, K)
        orig_BKD = resid_BLD[batch_idx_B1, pos_BK]  # (B, K, d)
        norms_BK1 = orig_BKD.norm(dim=-1, keepdim=True)  # (B, K, 1)

        # ---- build steered vectors ----
        steered_BKD = (
            torch.nn.functional.normalize(vec_BKD, dim=-1)
            * norms_BK1
            * steering_coefficient
        )  # (B, K, d)

        # ---- in-place replacement via advanced indexing ----
        resid_BLD[batch_idx_B1, pos_BK] = steered_BKD

        return (resid_BLD, *rest)

    return hook_fn


    


def get_bos_eos_pad_mask(
    tokenizer: PreTrainedTokenizer, token_ids: torch.Tensor
) -> torch.Tensor:
    return (
        (token_ids == tokenizer.pad_token_id)
        | (token_ids == tokenizer.eos_token_id)
        | (token_ids == tokenizer.bos_token_id)
    ).to(dtype=torch.bool)


def get_feature_activations(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    submodule: torch.nn.Module,
    sae: jumprelu_sae.JumpReluSAE,
    tokenized_strs: dict[str, torch.Tensor],
    ignore_bos: bool = True,
) -> torch.Tensor:
    with torch.no_grad():
        pos_acts_BLD = model_utils.collect_activations(model, submodule, tokenized_strs)

        encoded_pos_acts_BLF = sae.encode(pos_acts_BLD)

    if ignore_bos:
        bos_mask = tokenized_strs.input_ids == tokenizer.bos_token_id
        # Note: I use >=, not ==, because occasionally prompts will contain a BOS token
        assert bos_mask.sum() >= encoded_pos_acts_BLF.shape[0], (
            f"Expected at least {encoded_pos_acts_BLF.shape[0]} BOS tokens, but found {bos_mask.sum()}"
        )

        mask = get_bos_eos_pad_mask(tokenizer, tokenized_strs.input_ids)
        encoded_pos_acts_BLF[mask] = 0

    return encoded_pos_acts_BLF


if __name__ == "__main__":
    typer.run(main)
