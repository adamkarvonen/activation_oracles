# %%
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import contextlib
from typing import Callable, List, Dict, Tuple, Optional, Any
from jaxtyping import Float
from torch import Tensor
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)
import pickle
from dataclasses import dataclass, field, asdict
import einops
from rapidfuzz.distance import Levenshtein as lev
from tqdm import tqdm
import wandb
from torch.nn.utils import clip_grad_norm_
from peft import LoraConfig, get_peft_model
import json
import gc

import interp_tools.saes.jumprelu_sae as jumprelu_sae
import interp_tools.model_utils as model_utils
import interp_tools.introspect_utils as introspect_utils
from interp_tools.config import SelfInterpTrainingConfig, get_sae_info


# ==============================================================================
# 2. UTILITY FUNCTIONS
# ==============================================================================


def build_training_prompt() -> str:
    # TODO: Experiment with providing instructions to the model

    #     instruction = """
    # You are an introspection assistant. Your task is to explain a neural concept injected at ‹x>.

    # You must:
    # 1.  Write a **positive example** that triggers the concept as strongly as possible.
    # 2.  Write a **negative example** that is minimally different (by Levenshtein distance) but does **not** trigger the concept.
    # 3.  Write a one-line **explanation** of the core difference.

    # Priorities:
    # 1.  Maximize activation in the positive example.
    # 2.  Minimize activation in the negative example.
    # 3.  Maximize similarity between the two examples.

    # Use this exact format:
    # Positive example: <sentence>
    # Negative example: <sentence>
    # Explanation: <one line>
    # <END_OF_EXAMPLE>"""

    instruction = """Respond with the following format:\n\nPositive example: ...\n\nNegative example: ...\n\nExplanation: ...\n\n<END_OF_EXAMPLE>"""

    question = """Can you write me a sentence that relates to the concept 'X' and a similar sentence that does not relate to the concept?"""

    prompt = f"{instruction}\n\n{question}"

    return prompt


def parse_generated_explanation(text: str) -> Optional[dict[str, str]]:
    """
    Extract the positive example, negative example, and explanation
    from a model-generated block of text formatted as:

        Positive example: ...
        Negative example: ...
        Explanation: ...

    If any of those tags is missing, return None.
    """
    # Normalise leading / trailing whitespace
    text = text.strip()

    # Split at the first tag ─ discard anything that precedes it
    _, sep, remainder = text.partition("Positive example:")
    if not sep:  # tag not found
        return None
    remainder = remainder.lstrip()  # remove any space or newline right after the tag

    # Positive → Negative
    positive, sep, remainder = remainder.partition("Negative example:")
    if not sep:
        return None
    positive = positive.strip()
    remainder = remainder.lstrip()

    # Negative → Explanation
    negative, sep, explanation = remainder.partition("Explanation:")
    if not sep:
        return None

    if "<END_OF_EXAMPLE>" in explanation:
        explanation = explanation.split("<END_OF_EXAMPLE>")[0]
    else:
        return None

    return {
        "positive_sentence": positive.strip(),
        "negative_sentence": negative.strip(),
        "explanation": explanation.strip(),
    }


# ==============================================================================
# 3. HOOKING MECHANISM FOR ACTIVATION STEERING
# ==============================================================================


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
        norms_BK1 = orig_BKD.norm(dim=-1, keepdim=True).detach()  # (B, K, 1)

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


def collect_training_examples(
    results: dict,
    max_acts_ratio_threshold: float,
    max_distance_threshold: float,
    max_activation_required: float,
    max_examples: int | None = None,
) -> list[dict]:
    examples = []

    for i, feature_result in enumerate(results["results"]):
        for i in range(len(feature_result["sentence_metrics"])):
            sentence_metrics = feature_result["sentence_metrics"][i]
            sentence_data = feature_result["sentence_data"][i]

            if sentence_metrics["original_max_activation"] < max_activation_required:
                continue

            max_act_ratio = (
                sentence_metrics["rewritten_max_activation"]
                / (sentence_metrics["original_max_activation"])
            )

            if (
                max_act_ratio < max_acts_ratio_threshold
                and sentence_metrics["sentence_distance"] < max_distance_threshold
            ):
                example = {
                    "original_sentence": sentence_data["original_sentence"],
                    "rewritten_sentence": sentence_data["rewritten_sentence"],
                    "explanation": feature_result["explanation"],
                    "feature_idx": feature_result["feature_idx"],
                }

                examples.append(example)

        if max_examples is not None and len(examples) >= max_examples:
            break

    return examples


@torch.no_grad()
def construct_train_dataset(
    cfg: SelfInterpTrainingConfig,
    dataset_size: int,
    input_prompt: str,
    training_examples: list[dict],
    sae: jumprelu_sae.JumpReluSAE,
    tokenizer: PreTrainedTokenizer,
) -> list[dict]:
    input_messages = [{"role": "user", "content": input_prompt}]

    input_prompt_ids: list[int] = tokenizer.apply_chat_template(
        input_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors=None,
        padding=False,
        enable_thinking=False,
    )
    x_token_id = tokenizer.encode("X", add_special_tokens=False)[0]

    training_data = []

    for i in tqdm(range(dataset_size), desc="Constructing training dataset"):
        target_response = f"Positive example: {training_examples[i]['original_sentence']}\n\nNegative example: {training_examples[i]['rewritten_sentence']}\n\nExplanation: {training_examples[i]['explanation']}\n\n<END_OF_EXAMPLE>"

        full_messages = input_messages + [
            {"role": "assistant", "content": target_response}
        ]

        full_prompt_ids: list[int] = tokenizer.apply_chat_template(
            full_messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors=None,
            padding=False,
            enable_thinking=False,
        )
        target_feature_idx = training_examples[i]["feature_idx"]

        # 2. Prepare feature vectors for steering
        # We use decoder weights (W_dec) as they map from the feature space back to the residual stream.
        all_feature_indices = [target_feature_idx]

        # .clone() because otherwise we will save the entire W_dec in pickle for each training example
        if cfg.use_decoder_vectors:
            feature_vectors = [sae.W_dec[i].clone() for i in all_feature_indices]
        else:
            feature_vectors = [sae.W_enc[:, i].clone() for i in all_feature_indices]

        assistant_start_idx = len(input_prompt_ids)

        labels = full_prompt_ids.copy()
        for i in range(assistant_start_idx):
            labels[i] = -100

        positions = []
        for i in range(assistant_start_idx):
            if full_prompt_ids[i] == x_token_id:
                positions.append(i)
        assert len(positions) == 1, "Expected exactly one X token"

        training_data_point = {
            "input_ids": full_prompt_ids,
            "labels": labels,
            "steering_vectors": feature_vectors,
            "positions": positions,
            "feature_indices": all_feature_indices,
        }

        training_data.append(training_data_point)

    return training_data


def construct_eval_dataset(
    cfg: SelfInterpTrainingConfig,
    dataset_size: int,
    input_prompt: str,
    eval_feature_indices: list[int],
    api_data: dict,
    sae: jumprelu_sae.JumpReluSAE,
    tokenizer: PreTrainedTokenizer,
    prefill_original_sentences: bool = False,
) -> list[dict]:
    """Every prompt is exactly the same - the only difference is the steering vectors."""

    input_messages = [{"role": "user", "content": input_prompt}]

    input_prompt_ids: list[int] = tokenizer.apply_chat_template(
        input_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors=None,
        padding=False,
        enable_thinking=False,
    )
    labels = input_prompt_ids.copy()

    orig_prompt_length = len(input_prompt_ids)

    x_token_id = tokenizer.encode("X", add_special_tokens=False)[0]

    eval_data = []

    first_position = None

    for i in tqdm(range(dataset_size), desc="Constructing eval dataset"):
        target_feature_idx = eval_feature_indices[i]

        if prefill_original_sentences:
            sentence_data = None
            for i, feature_result in enumerate(api_data["results"]):
                # just getting the first sentence data for now
                # TODO: Ensure comparison with api data is correct
                if feature_result["feature_idx"] != target_feature_idx:
                    continue
                sentence_data = feature_result["sentence_data"][0]
                break

            assert sentence_data is not None, "No sentence data found for feature index"

            target_response = f"Positive example: {sentence_data['original_sentence']}\n\nNegative example:"

            input_messages = [
                {"role": "user", "content": input_prompt},
                {"role": "assistant", "content": target_response},
            ]

            input_prompt_ids: list[int] = tokenizer.apply_chat_template(
                input_messages,
                tokenize=True,
                add_generation_prompt=False,
                continue_final_message=True,
                return_tensors=None,
                padding=False,
                enable_thinking=False,
            )

            labels = input_prompt_ids.copy()

        # 2. Prepare feature vectors for steering
        # We use decoder weights (W_dec) as they map from the feature space back to the residual stream.
        all_feature_indices = [target_feature_idx]

        if cfg.use_decoder_vectors:
            feature_vectors = [sae.W_dec[i].clone() for i in all_feature_indices]
        else:
            feature_vectors = [sae.W_enc[:, i].clone() for i in all_feature_indices]

        positions = []
        for i in range(orig_prompt_length):
            if input_prompt_ids[i] == x_token_id:
                positions.append(i)

        assert len(positions) == 1, "Expected exactly one X token"

        if first_position is None:
            first_position = positions[0]
        else:
            assert positions[0] == first_position, (
                "Expected all positions to be the same"
            )

        eval_data_point = {
            "input_ids": input_prompt_ids,
            "labels": labels,
            "steering_vectors": feature_vectors,
            "positions": positions,
            "feature_indices": all_feature_indices,
        }

        eval_data.append(eval_data_point)

    return eval_data


def construct_batch(
    training_data: list[dict],
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
) -> list[dict]:
    max_length = 0
    for data_point in training_data:
        max_length = max(max_length, len(data_point["input_ids"]))

    batch_tokens = []
    batch_labels = []
    batch_attn_masks = []
    batch_positions = []
    batch_steering_vectors = []
    batch_feature_indices = []

    for data_point in training_data:
        padding_length = max_length - len(data_point["input_ids"])
        padding_tokens = [tokenizer.pad_token_id] * padding_length
        data_point["input_ids"] = padding_tokens + data_point["input_ids"]
        data_point["labels"] = [-100] * padding_length + data_point["labels"]

        input_ids = torch.tensor(data_point["input_ids"], dtype=torch.long).to(device)
        labels = torch.tensor(data_point["labels"], dtype=torch.long).to(device)
        attn_mask = torch.ones_like(input_ids, dtype=torch.bool).to(device)

        attn_mask[:padding_length] = False

        batch_tokens.append(input_ids)
        batch_labels.append(labels)
        batch_attn_masks.append(attn_mask)

        positions = data_point["positions"]
        positions = [p + padding_length for p in positions]

        batch_positions.append(positions)
        batch_steering_vectors.append(data_point["steering_vectors"])
        batch_feature_indices.append(data_point["feature_indices"])

    return {
        "input_ids": torch.stack(batch_tokens),
        "labels": torch.stack(batch_labels),
        "attention_mask": torch.stack(batch_attn_masks),
        "steering_vectors": batch_steering_vectors,
        "positions": batch_positions,
        "feature_indices": batch_feature_indices,
    }


def train_features_batch(
    cfg: SelfInterpTrainingConfig,
    training_batch: dict,
    model: AutoModelForCausalLM,
    submodule: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Trains the model on a single batch of data.
    """

    batch_steering_vectors = training_batch["steering_vectors"]
    batch_positions = training_batch["positions"]

    # 3. Create and apply the activation steering hook
    hook_fn = get_activation_steering_hook(
        vectors=batch_steering_vectors,
        positions=batch_positions,
        steering_coefficient=cfg.steering_coefficient,
        device=device,
        dtype=dtype,
    )

    tokenized_input = {
        "input_ids": training_batch["input_ids"],
        "attention_mask": training_batch["attention_mask"],
    }

    with add_hook(submodule, hook_fn):
        loss = model(**tokenized_input, labels=training_batch["labels"]).loss

    return loss


@torch.no_grad()
def eval_features_batch(
    cfg: SelfInterpTrainingConfig,
    eval_batch: dict,
    model: AutoModelForCausalLM,
    submodule: torch.nn.Module,
    sae: jumprelu_sae.JumpReluSAE,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    dtype: torch.dtype,
) -> dict:
    batch_steering_vectors = eval_batch["steering_vectors"]
    batch_positions = eval_batch["positions"]

    # 3. Create and apply the activation steering hook
    hook_fn = get_activation_steering_hook(
        vectors=batch_steering_vectors,
        positions=batch_positions,
        steering_coefficient=cfg.steering_coefficient,
        device=device,
        dtype=dtype,
    )

    tokenized_input = {
        "input_ids": eval_batch["input_ids"],
        "attention_mask": eval_batch["attention_mask"],
    }

    with add_hook(submodule, hook_fn):
        output_ids = model.generate(**tokenized_input, **cfg.generation_kwargs)

    # Decode only the newly generated tokens
    generated_tokens = output_ids[:, eval_batch["input_ids"].shape[1] :]
    decoded_output = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    pos_statements = []
    neg_statements = []
    explanations = []

    prompt_tokens = eval_batch["input_ids"][:, : eval_batch["input_ids"].shape[1]]
    decoded_prompts = tokenizer.batch_decode(prompt_tokens, skip_special_tokens=False)

    for i, output in enumerate(decoded_output):
        # Note: we add the "Positive example:" prefix to the positive sentence
        # because we prefill the model's response with "Positive example:"
        if cfg.prefill_original_sentences:
            orig_sentence = decoded_prompts[i].split("Positive example:")[-1]
            output = f"Positive example:{orig_sentence}{output}"
        else:
            output = f"Positive example:{output}"

        parsed_result = parse_generated_explanation(output)
        print(output)
        if parsed_result:
            pos_statements.append(parsed_result["positive_sentence"])
            neg_statements.append(parsed_result["negative_sentence"])
            explanations.append(parsed_result["explanation"])
        else:
            pos_statements.append("")
            neg_statements.append("")
            explanations.append("")

    all_sentence_data, all_sentence_metrics = introspect_utils.eval_statements(
        pos_statements=pos_statements,
        neg_statements=neg_statements,
        feature_indices=eval_batch["feature_indices"],
        model=model,
        submodule=submodule,
        sae=sae,
        tokenizer=tokenizer,
    )

    feature_results = []

    assert len(all_sentence_data) == len(all_sentence_metrics)
    assert len(all_sentence_data) == len(decoded_output)

    for i, output in enumerate(decoded_output):
        feature_idx = eval_batch["feature_indices"][i]
        feature_result = {
            "feature_idx": feature_idx,
            "api_response": output,
            "prompt": decoded_prompts[i],
            "explanation": explanations[i],
            "sentence_data": all_sentence_data[i],
            "sentence_metrics": all_sentence_metrics[i],
        }

        feature_results.append(feature_result)

    return feature_results


def save_logs(
    eval_results_path: str,
    global_step: int,
    all_feature_results_this_eval_step: list[dict],
):
    # Load existing data, append new results, and save
    try:
        with open(eval_results_path, "r") as f:
            all_run_results = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        all_run_results = []

    # Add results from the current evaluation step
    all_run_results.append(
        {
            "step": global_step,
            "results": all_feature_results_this_eval_step,
        }
    )

    with open(eval_results_path, "w") as f:
        json.dump(all_run_results, f, indent=2)


def train_model(
    cfg: SelfInterpTrainingConfig,
    training_data: list[dict],
    eval_data: list[dict],
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    submodule: torch.nn.Module,
    sae: jumprelu_sae.JumpReluSAE,
    device: torch.device,
    dtype: torch.dtype,
    verbose: bool = False,
    use_wandb: bool = True,
):
    max_grad_norm = 1.0
    wandb_project = "sae_introspection"
    run_name = f"{cfg.model_name}-layer{cfg.sae_layer}-decoder-shorter-prompt"

    if use_wandb:
        wandb.init(project=wandb_project, name=run_name, config=asdict(cfg))

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    total_training_steps = cfg.num_epochs * len(training_data)
    warmup_steps = 200
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )
    # --------------------------------------------------------------

    global_step = 0

    if os.path.exists("eval_logs.json"):
        os.remove("eval_logs.json")

    for epoch in range(cfg.num_epochs):
        for i in tqdm(
            range(0, len(training_data), cfg.train_batch_size),
            desc=f"Training epoch {epoch + 1}",
        ):
            t_batch = training_data[i : i + cfg.train_batch_size]
            t_batch = construct_batch(t_batch, tokenizer, device)

            if i % 100 == 0:
                torch.cuda.empty_cache()
                gc.collect()
            loss = train_features_batch(cfg, t_batch, model, submodule, device, dtype)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if use_wandb:
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/learning_rate": scheduler.get_last_lr()[0],
                    },
                    step=global_step,
                )
            if verbose:
                print(f"Step {global_step} loss: {loss.item()}")

            # -------------------------------- evaluation --------------------------------
            if global_step % cfg.eval_steps == 0:
                model.eval()
                with torch.no_grad():
                    all_sentence_metrics = []
                    all_feature_results_this_eval_step = []
                    for i in tqdm(
                        range(0, len(eval_data), cfg.eval_batch_size),
                        desc="Evaluating model",
                    ):
                        e_batch = eval_data[i : i + cfg.eval_batch_size]
                        e_batch = construct_batch(e_batch, tokenizer, device)

                        feature_results = eval_features_batch(
                            cfg=cfg,
                            eval_batch=e_batch,
                            model=model,
                            submodule=submodule,
                            sae=sae,
                            tokenizer=tokenizer,
                            device=device,
                            dtype=dtype,
                        )
                        all_feature_results_this_eval_step.extend(feature_results)
                        for res in feature_results:
                            all_sentence_metrics.append(res["sentence_metrics"])

                    save_logs(
                        eval_results_path="eval_logs.json",
                        global_step=global_step,
                        all_feature_results_this_eval_step=all_feature_results_this_eval_step,
                    )

                    if all_sentence_metrics:
                        aggregated_metrics = {}
                        metric_keys = all_sentence_metrics[0].keys()
                        for key in metric_keys:
                            avg_val = sum(m[key] for m in all_sentence_metrics) / len(
                                all_sentence_metrics
                            )
                            aggregated_metrics[key] = avg_val

                        wandb_log_dict = {
                            f"eval/{k}": v for k, v in aggregated_metrics.items()
                        }

                        if use_wandb:
                            wandb.log(wandb_log_dict, step=global_step)
                        if verbose:
                            print(
                                f"Step {global_step} eval metrics: {aggregated_metrics}"
                            )

                model.train()

            if global_step % cfg.save_steps == 0:
                model.save_pretrained(f"{cfg.save_dir}/step_{global_step}")

            global_step += 1

    print("Training complete.")
    wandb.finish()


def main():
    """Main script logic."""
    cfg = SelfInterpTrainingConfig()

    cfg.eval_set_size = 1000
    cfg.save_steps = 2000
    cfg.steering_coefficient = 2.0
    cfg.train_batch_size = 4
    cfg.eval_batch_size = cfg.train_batch_size * 16
    cfg.lr = 5e-6
    cfg.training_data_filename = "contrastive_rewriting_results_131k.pkl"
    cfg.training_data_filename = (
        "contrastive_rewriting_results_google_gemma-2-9b-it_num_features_200000.pkl"
    )
    # cfg.training_data_filename = "contrastive_rewriting_results_2k.pkl"
    # cfg.training_data_filename = "contrastive_rewriting_results.pkl"
    verbose = True
    cfg.use_decoder_vectors = True

    # %%
    print(asdict(cfg))
    dtype = torch.bfloat16
    device = torch.device("cuda")

    # %%

    api_data_filename = cfg.training_data_filename

    with open(api_data_filename, "rb") as f:
        api_data = pickle.load(f)

    cfg.sae_width = api_data["config"]["sae_width"]
    cfg.sae_layer = api_data["config"]["sae_layer"]
    cfg.sae_filename = api_data["config"]["sae_filename"]
    cfg.sae_repo_id = api_data["config"]["sae_repo_id"]

    model = introspect_utils.load_model(cfg, device, dtype, use_lora=cfg.use_lora)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    sae = introspect_utils.load_sae(cfg, device, dtype)
    submodule = model_utils.get_submodule(model, cfg.sae_layer, cfg.use_lora)

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # %%

    max_activation = 0
    for feature_result in api_data["results"]:
        for sentence_metrics in feature_result["sentence_metrics"]:
            max_activation = max(
                max_activation, sentence_metrics["original_max_activation"]
            )

    print(f"Max activation: {max_activation}")

    cfg.max_activation_required = (
        cfg.max_activation_percentage_required * max_activation
    )

    training_examples = collect_training_examples(
        api_data,
        cfg.max_acts_ratio_threshold,
        cfg.max_distance_threshold,
        cfg.max_activation_required,
    )

    train_features = set()

    for example in training_examples:
        train_features.add(example["feature_idx"])

    test_features = set()

    for example in api_data["results"]:
        if example["feature_idx"] not in train_features:
            test_features.add(example["feature_idx"])

    print(f"train examples: {len(training_examples)}")
    print(f"Train features: {len(train_features)}, Test features: {len(test_features)}")

    cfg.eval_features = list(test_features)[: cfg.eval_set_size]

    assert len(cfg.eval_features) == cfg.eval_set_size

    train_eval_prompt = build_training_prompt()

    training_data = construct_train_dataset(
        cfg,
        len(training_examples),
        # dataset_size,
        train_eval_prompt,
        training_examples,
        sae,
        tokenizer,
    )

    eval_data = construct_eval_dataset(
        cfg,
        cfg.eval_set_size,
        train_eval_prompt,
        cfg.eval_features,
        api_data,
        sae,
        tokenizer,
    )

    # %%

    print(f"training data: {len(training_data)}, eval data: {len(eval_data)}")

    # training_data = training_data[:100]
    # temp_eval_data = eval_data[:10]

    temp_eval_data = eval_data[:250]

    train_model(
        cfg,
        training_data,
        temp_eval_data,
        model,
        tokenizer,
        submodule,
        sae,
        device,
        dtype,
        verbose=True,
        use_wandb=True,
    )


if __name__ == "__main__":
    main()
