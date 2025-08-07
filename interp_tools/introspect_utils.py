import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizer
import interp_tools.saes.jumprelu_sae as jumprelu_sae
from rapidfuzz.distance import Levenshtein as lev
from peft import LoraConfig, get_peft_model
import contextlib

import interp_tools.model_utils as model_utils
import interp_tools.saes.base_sae as base_sae
import interp_tools.saes.jumprelu_sae as jumprelu_sae
import interp_tools.saes.topk_sae as topk_sae
from interp_tools.config import get_sae_info, SelfInterpTrainingConfig


def load_model(
    cfg: SelfInterpTrainingConfig,
    device: torch.device,
    dtype: torch.dtype,
    use_lora: bool,
) -> AutoModelForCausalLM:
    attn_kwargs = {}

    if cfg.model_name == "google/gemma-2-9b-it":
        attn_kwargs = {"attn_implementation": "eager"}

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, device_map="auto", torch_dtype=dtype, **attn_kwargs
    )

    if use_lora:
        print("Applying LoRA configuration...")
        lora_config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model


def load_sae(
    cfg: SelfInterpTrainingConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> base_sae.BaseSAE:
    # Note: There's some duplication here with saes.sae_loading_utils.py
    print(f"Loading SAE for layer {cfg.sae_layer} from {cfg.sae_repo_id}...")

    if cfg.sae_repo_id == "google/gemma-scope-9b-it-res":
        sae = jumprelu_sae.load_gemma_scope_jumprelu_sae(
            repo_id=cfg.sae_repo_id,
            filename=cfg.sae_filename,
            layer=cfg.sae_layer,
            model_name=cfg.model_name,
            device=device,
            dtype=dtype,
        )
    elif cfg.sae_repo_id == "fnlp/Llama3_1-8B-Base-LXR-32x":
        sae = topk_sae.load_llama_scope_topk_sae(
            model_name=cfg.model_name,
            device=device,
            dtype=dtype,
            layer=cfg.sae_layer,
            expansion_factor=cfg.sae_width,
        )
    else:
        raise ValueError(f"Unknown SAE repo ID: {cfg.sae_repo_id}")

    return sae


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


def has_active_lora(model: AutoModelForCausalLM) -> bool:
    """
    True ⇢ model is a PEFT/PeftModel object *and* at least one adapter is enabled.
    """
    return (
        hasattr(model, "peft_config")  # model was wrapped by PEFT
        and len(model.peft_config) > 0  # at least one adapter is loaded
        and getattr(model, "active_adapter", None)  # an adapter is currently selected
    )


@torch.no_grad()
def eval_statements(
    pos_statements: list[str],
    neg_statements: list[str],
    feature_indices: list[int],
    model: AutoModelForCausalLM,
    submodule: torch.nn.Module,
    sae: jumprelu_sae.JumpReluSAE,
    tokenizer: PreTrainedTokenizer,
) -> tuple[list[dict], list[dict]]:
    """NOTE: Major assumption is if statements were parsed incorrectly,
    the statement == '' (empty string)"""
    assert len(pos_statements) == len(neg_statements) == len(feature_indices)

    pos_tokenized_strs = tokenizer(
        pos_statements, return_tensors="pt", add_special_tokens=True, padding=True
    ).to(model.device)

    neg_tokenized_strs = tokenizer(
        neg_statements, return_tensors="pt", add_special_tokens=True, padding=True
    ).to(model.device)

    ctx = (
        model.disable_adapter() if has_active_lora(model) else contextlib.nullcontext()
    )

    with ctx:
        # TODO: shrink SAE to only the features we're interested in for reduced memory usage
        pos_activations_BLF = get_feature_activations(
            model=model,
            tokenizer=tokenizer,
            submodule=submodule,
            sae=sae,
            tokenized_strs=pos_tokenized_strs,
        )

        neg_activations_BLF = get_feature_activations(
            model=model,
            tokenizer=tokenizer,
            submodule=submodule,
            sae=sae,
            tokenized_strs=neg_tokenized_strs,
        )

    all_sentence_data = []
    all_sentence_metrics = []

    for i in range(len(pos_statements)):
        feature_idx = feature_indices[i]

        pos_tokens_L = pos_tokenized_strs.input_ids[i, :]
        neg_tokens_L = neg_tokenized_strs.input_ids[i, :]

        pos_mask_L = get_bos_eos_pad_mask(tokenizer, pos_tokens_L)
        neg_mask_L = get_bos_eos_pad_mask(tokenizer, neg_tokens_L)

        pos_activations_L = pos_activations_BLF[i, :, feature_idx]
        neg_activations_L = neg_activations_BLF[i, :, feature_idx]

        sentence_distance = lev.normalized_distance(
            pos_statements[i], neg_statements[i]
        )

        original_max_activation = pos_activations_L.max().item()
        rewritten_max_activation = neg_activations_L.max().item()

        valid_pos_tokens = len(pos_mask_L) - int(pos_mask_L.sum().item())
        valid_neg_tokens = len(neg_mask_L) - int(neg_mask_L.sum().item())

        original_mean_activation = 0.0
        rewritten_mean_activation = 0.0

        if valid_pos_tokens > 0:
            original_mean_activation = pos_activations_L.sum().item() / valid_pos_tokens

        if valid_neg_tokens > 0:
            rewritten_mean_activation = (
                neg_activations_L.sum().item() / valid_neg_tokens
            )

        max_activation_ratio = rewritten_max_activation / (
            original_max_activation + 1e-6
        )
        mean_activation_ratio = rewritten_mean_activation / (
            original_mean_activation + 1e-6
        )

        max_activation_ratio = max(max_activation_ratio, 0.0)
        mean_activation_ratio = max(mean_activation_ratio, 0.0)

        max_activation_ratio = min(max_activation_ratio, 1.0)
        mean_activation_ratio = min(mean_activation_ratio, 1.0)

        format_correct = 1.0
        if pos_statements[i] == "":
            assert valid_pos_tokens == 0, (
                f"Expected 0 valid tokens, but found {valid_pos_tokens}"
            )

        if neg_statements[i] == "":
            assert valid_neg_tokens == 0, (
                f"Expected 0 valid tokens, but found {valid_neg_tokens}"
            )

        if valid_pos_tokens == 0 or valid_neg_tokens == 0:
            format_correct = 0.0
            original_max_activation = 0.0
            rewritten_max_activation = 0.0
            original_mean_activation = 0.0
            rewritten_mean_activation = 0.0

        if "nan" in str(original_mean_activation) or "nan" in str(
            rewritten_mean_activation
        ):
            raise ValueError("NaN found in sentence metrics")

        max_activation_nonzero = 0.0
        if original_max_activation > 0.0 and format_correct == 1.0:
            max_activation_nonzero = 1.0

        success_max_activation_ratio = 1.0
        success_mean_activation_ratio = 1.0

        if format_correct == 1.0 and max_activation_nonzero == 1.0:
            success_max_activation_ratio = max_activation_ratio
            success_mean_activation_ratio = mean_activation_ratio

        sentence_data = {
            "sentence_index": 0,
            "original_sentence": pos_statements[i],
            "rewritten_sentence": neg_statements[i],
        }

        sentence_metrics = {
            "original_max_activation": original_max_activation,
            "rewritten_max_activation": rewritten_max_activation,
            "original_mean_activation": original_mean_activation,
            "rewritten_mean_activation": rewritten_mean_activation,
            "max_activation_ratio": max_activation_ratio,
            "mean_activation_ratio": mean_activation_ratio,
            "sentence_distance": sentence_distance,
            "format_correct": format_correct,
            "max_activation_nonzero": max_activation_nonzero,
            "success_max_activation_ratio": success_max_activation_ratio,
            "success_mean_activation_ratio": success_mean_activation_ratio,
        }

        all_sentence_data.append(sentence_data)
        all_sentence_metrics.append(sentence_metrics)

    return all_sentence_data, all_sentence_metrics
