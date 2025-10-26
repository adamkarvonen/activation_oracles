import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import json
import random
from datetime import timedelta

# All necessary imports are now included above
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer, BitsAndBytesConfig
from transformers.optimization import get_linear_schedule_with_warmup
import torch.distributed as dist
import wandb

import nl_probes.dataset_classes.classification as classification
from nl_probes.utils.steering_hooks import (
    add_hook,
    get_hf_activation_steering_hook,
)
from nl_probes.configs.sft_config import SelfInterpTrainingConfig
from nl_probes.dataset_classes.act_dataset_manager import ActDatasetLoader, DatasetLoaderConfig
from nl_probes.dataset_classes.classification import (
    ClassificationDatasetConfig,
    ClassificationDatasetLoader,
)
from nl_probes.dataset_classes.latentqa_dataset import LatentQADatasetConfig, LatentQADatasetLoader
from nl_probes.dataset_classes.past_lens_dataset import PastLensDatasetConfig, PastLensDatasetLoader
from nl_probes.dataset_classes.sae_training_data import (
    SAEActivatingSequencesDatasetConfig,
    SAEActivatingSequencesDatasetLoader,
    SAEExplanationDatasetConfig,
    SAEExplanationDatasetLoader,
    SAEYesNoDatasetConfig,
    SAEYesNoDatasetLoader,
)
from nl_probes.utils.activation_utils import get_hf_submodule
from nl_probes.utils.common import load_model, load_tokenizer, set_seed
from nl_probes.utils.dataset_utils import (
    BatchData,
    EvalStepResult,
    ExplanationResult,
    FeatureResult,
    TrainingDataPoint,
    construct_batch,
    materialize_missing_steering_vectors,
)
from nl_probes.utils.eval import run_evaluation, score_eval_responses


def push_lora_to_hf(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    repo_id: str,
    private: bool,
    commit_message: str = "Upload LoRA adapter after training",
) -> None:
    """
    Push the trained LoRA adapter to Hugging Face Hub.

    Args:
        model: The trained model with LoRA adapters
        tokenizer: The tokenizer used with the model
        repo_id: HuggingFace repository ID (e.g., "username/repo-name")
        commit_message: Commit message for the upload
        private: Whether to make the repository private

    Returns:
        bool: True if successful, False otherwise
    """

    print(f"Pushing LoRA adapter to Hugging Face Hub: {repo_id}")

    # Get the original model name to copy config from
    original_model_name = model.config._name_or_path
    if hasattr(model, "base_model"):
        # For LoRA models, get the base model name
        original_model_name = model.base_model.config._name_or_path

    # Push the model (LoRA adapters)
    model.push_to_hub(
        repo_id=repo_id,
        commit_message=commit_message,
        private=private,
    )

    # Push the tokenizer as well
    tokenizer.push_to_hub(
        repo_id=repo_id,
        commit_message=f"Upload tokenizer - {commit_message}",
        private=private,
    )

    # Copy config.json from the original model
    try:
        import tempfile

        from huggingface_hub import hf_hub_download, upload_file

        print(f"Copying config.json from original model: {original_model_name}")

        # Download config.json from the original model
        with tempfile.NamedTemporaryFile(mode="w+b", suffix=".json", delete=False) as tmp_file:
            config_path = hf_hub_download(
                repo_id=original_model_name,
                filename="config.json",
                cache_dir=None,
                force_download=False,
            )

            # Copy the file content
            with open(config_path, "rb") as src:
                tmp_file.write(src.read())
            tmp_file.flush()

            # Upload to the LoRA repo
            upload_file(
                path_or_fileobj=tmp_file.name,
                path_in_repo="config.json",
                repo_id=repo_id,
                commit_message=f"Copy config.json from {original_model_name}",
            )

        # Clean up temp file
        os.unlink(tmp_file.name)
        print(f"Successfully copied config.json from {original_model_name}")

    except Exception as e:
        print(f"Warning: Failed to copy config.json from original model: {e}")
        print("LoRA adapter uploaded successfully, but without original model config")

    # Create and upload README with base model metadata
    try:
        print("Creating README with base model metadata...")

        readme_content = f"""---
base_model: {original_model_name}
library_name: peft
---

# LoRA Adapter for SAE Introspection

This is a LoRA (Low-Rank Adaptation) adapter trained for SAE (Sparse Autoencoder) introspection tasks.

## Base Model
- **Base Model**: `{original_model_name}`
- **Adapter Type**: LoRA
- **Task**: SAE Feature Introspection

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model and tokenizer
base_model = AutoModelForCausalLM.from_pretrained("{original_model_name}")
tokenizer = AutoTokenizer.from_pretrained("{original_model_name}")

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "{repo_id}")
```

## Training Details
This adapter was trained using the lightweight SAE introspection training script to help the model understand and explain SAE features through activation steering.
"""

        # Create temporary README file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as tmp_readme:
            tmp_readme.write(readme_content)
            tmp_readme.flush()

            # Upload README to the LoRA repo
            upload_file(
                path_or_fileobj=tmp_readme.name,
                path_in_repo="README.md",
                repo_id=repo_id,
                commit_message="Add README with base model metadata",
            )

        # Clean up temp file
        os.unlink(tmp_readme.name)
        print("Successfully uploaded README with base model metadata")

    except Exception as e:
        print(f"Warning: Failed to upload README: {e}")
        print("LoRA adapter uploaded successfully, but without README")

    print(f"Successfully pushed LoRA adapter to: https://huggingface.co/{repo_id}")


def parse_generated_explanation(text: str) -> Optional[ExplanationResult]:
    """
    Extract the explanation from a model-generated block of text formatted as:
    <explanation>...</explanation>

    If the tag is missing, return None.
    """
    # Normalise leading / trailing whitespace
    text = text.strip()

    # Look for <explanation> tags
    start_tag = "<explanation>"
    end_tag = "</explanation>"

    start_idx = text.find(start_tag)
    if start_idx == -1:
        return None

    end_idx = text.find(end_tag, start_idx + len(start_tag))
    if end_idx == -1:
        return None

    # Extract content between tags
    explanation = text[start_idx + len(start_tag) : end_idx].strip()

    if not explanation:
        return None

    return ExplanationResult(
        explanation=explanation,
    )


def train_features_batch(
    cfg: SelfInterpTrainingConfig,
    training_batch: BatchData,
    model: AutoModelForCausalLM,
    submodule: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Trains the model on a single batch of data.
    """

    batch_steering_vectors = training_batch.steering_vectors
    batch_positions = training_batch.positions

    # 3. Create and apply the activation steering hook
    hook_fn = get_hf_activation_steering_hook(
        vectors=batch_steering_vectors,
        positions=batch_positions,
        steering_coefficient=cfg.steering_coefficient,
        device=device,
        dtype=dtype,
    )

    tokenized_input = {
        "input_ids": training_batch.input_ids,
        "attention_mask": training_batch.attention_mask,
    }

    with add_hook(submodule, hook_fn):
        loss = model(**tokenized_input, labels=training_batch.labels).loss

    return loss


def eval_all_datasets(
    cfg: SelfInterpTrainingConfig,
    eval_datasets: dict[str, list[TrainingDataPoint]],
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    submodule: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    global_step: int,
) -> None:
    model.eval()
    eval_results = {}
    for ds in eval_datasets:
        eval_responses = run_evaluation(
            eval_data=eval_datasets[ds],
            model=model,
            tokenizer=tokenizer,
            submodule=submodule,
            device=device,
            dtype=dtype,
            global_step=global_step,
            lora_path=None,
            eval_batch_size=cfg.eval_batch_size,
            steering_coefficient=cfg.steering_coefficient,
            generation_kwargs=cfg.generation_kwargs,
        )
        percent_format_correct, percent_ans_correct = score_eval_responses(eval_responses, eval_datasets[ds])
        eval_results[f"eval_format_correct/{ds}"] = percent_format_correct
        eval_results[f"eval_ans_correct/{ds}"] = percent_ans_correct
        print(f"Step {global_step} {ds} format correct: {percent_format_correct}, ans correct: {percent_ans_correct}")

    wandb.log(
        eval_results,
        step=global_step,
    )
    wandb.summary.update(eval_results)
    model.train()

    # Have occasionally seen OOMs on first training step after eval, so clear cache here
    torch.cuda.empty_cache()
    gc.collect()


def oom_preflight_check(
    cfg: SelfInterpTrainingConfig,
    training_data: list[TrainingDataPoint],
    model: AutoModelForCausalLM,
    submodule: torch.nn.Module,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    longest_prompt = max(training_data, key=lambda x: len(x.input_ids))
    long_prompts = [longest_prompt] * cfg.train_batch_size
    long_prompts = materialize_missing_steering_vectors(long_prompts, tokenizer, model)
    largest_possible_batch = construct_batch(long_prompts, tokenizer, device)

    dummy_optimizer = torch.optim.AdamW(model.parameters(), lr=0.0)

    for _ in tqdm(range(3), desc="OOM preflight check"):
        loss = train_features_batch(cfg, largest_possible_batch, model, submodule, device, dtype)
        loss.backward()
        dummy_optimizer.step()
        dummy_optimizer.zero_grad()

    del dummy_optimizer
    torch.cuda.empty_cache()
    gc.collect()

    print("OOM preflight check complete")


def train_model(
    cfg: SelfInterpTrainingConfig,
    training_data: list[TrainingDataPoint],
    eval_datasets: dict[str, list[TrainingDataPoint]],
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    dtype: torch.dtype,
    model_kwargs: dict[str, Any],
    verbose: bool = False,
):
    # Distributed settings (always on; launch with torchrun, even on 1 GPU)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    # Ensure loads happen on this GPU only (important for quantized models)
    model_kwargs = {
        **model_kwargs,
        "device_map": {"": f"cuda:{local_rank}"},
    }

    set_seed(cfg.seed)
    model = load_model(cfg.model_name, dtype, **model_kwargs)

    model.enable_input_require_grads()

    if cfg.gradient_checkpointing:
        model.use_cache = False
        model.gradient_checkpointing_enable()

    submodule = get_hf_submodule(model, cfg.hook_onto_layer)

    if cfg.use_lora and cfg.load_lora_path is None:
        lora_config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config, autocast_adapter_dtype=True)
    elif cfg.load_lora_path is not None:
        load_lora_path = Path(cfg.load_lora_path)
        assert load_lora_path.exists()
        model = PeftModel.from_pretrained(model, load_lora_path, is_trainable=True, autocast_adapter_dtype=True)

    model.print_trainable_parameters()

    # Wrap with DDP for training, but keep the PEFT model reference for hooks/eval
    torch.cuda.set_device(local_rank)
    train_model_module: torch.nn.Module = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False
    )

    train_model_module.train()

    oom_preflight_check(cfg, training_data, model, submodule, tokenizer, device, dtype)

    set_seed(cfg.seed)

    optimizer = torch.optim.AdamW(train_model_module.parameters(), lr=cfg.lr)

    global_step_size = cfg.train_batch_size * world_size
    effective_steps = (len(training_data) // global_step_size) * global_step_size
    if effective_steps != len(training_data):
        print(f"Trimming training_data from {len(training_data)} to {effective_steps} for equal DDP steps")
        training_data = training_data[:effective_steps]

    # Shard dataset per rank (simple strided split)
    training_data = training_data[rank::world_size]

    total_training_steps = (cfg.num_epochs * len(training_data)) // cfg.train_batch_size
    # 10 percent
    warmup_steps = int(total_training_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )
    # --------------------------------------------------------------

    global_step = 0

    # Init Weights & Biases only on rank 0
    if rank == 0:
        wandb.init(project=cfg.wandb_project, name=cfg.wandb_run_name, config=asdict(cfg))

    for epoch in range(cfg.num_epochs):
        for i in tqdm(
            range(0, len(training_data), cfg.train_batch_size),
            desc=f"Training epoch {epoch + 1}",
            disable=rank != 0,
        ):
            t_batch_list: list[TrainingDataPoint] = training_data[i : i + cfg.train_batch_size]

            # Compute missing steering vectors using the PEFT model (not DDP wrapper)
            t_batch_list = materialize_missing_steering_vectors(t_batch_list, tokenizer, model)

            t_batch = construct_batch(t_batch_list, tokenizer, device)

            # Forward/backward on the DDP-wrapped module if enabled
            loss = train_features_batch(cfg, t_batch, train_model_module, submodule, device, dtype)
            loss.backward()
            clip_grad_norm_(train_model_module.parameters(), cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if rank == 0:
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
            if global_step % cfg.eval_steps == 0 and (cfg.eval_on_start or global_step > 0):
                if rank == 0:
                    eval_all_datasets(cfg, eval_datasets, model, tokenizer, submodule, device, dtype, global_step)
                dist.barrier()

            if global_step % cfg.save_steps == 0 and global_step > 0:
                if rank == 0:
                    model.save_pretrained(f"{cfg.save_dir}/step_{global_step}")
                    if cfg.hf_push_to_hub and cfg.hf_repo_id:
                        print("Pushing LoRA adapter to Hugging Face Hub...")
                        push_lora_to_hf(
                            model=model,
                            tokenizer=tokenizer,
                            repo_id=cfg.hf_repo_id + f"-step-{global_step}",
                            private=cfg.hf_private_repo,
                            commit_message=(f"SAE introspection LoRA - {cfg.wandb_run_name} - step {global_step}"),
                        )
                        print("Pushed LoRA adapter to Hugging Face Hub.")
                dist.barrier()

            global_step += 1

    print("Training complete.")

    # Save final model
    if rank == 0:
        print("Saving final model...")
        model.save_pretrained(f"{cfg.save_dir}/final")

        # Final evaluation
        print("Running final evaluation...")
        eval_all_datasets(cfg, eval_datasets, model, tokenizer, submodule, device, dtype, global_step)
        wandb.finish()

        # Push to Hugging Face if configured
        if cfg.hf_push_to_hub and cfg.hf_repo_id:
            print("Pushing LoRA adapter to Hugging Face Hub...")
            push_lora_to_hf(
                model=model,
                tokenizer=tokenizer,
                repo_id=cfg.hf_repo_id,
                commit_message=f"SAE introspection LoRA - {cfg.wandb_run_name} - final model",
                private=cfg.hf_private_repo,
            )
    dist.barrier()


def length_grouped_reorder(
    data: list[TrainingDataPoint],
    batch_size: int,
    window_mult: int,
) -> list[TrainingDataPoint]:
    lengths = [len(d.input_ids) for d in data]

    indices = list(range(len(data)))
    megabatch_size = window_mult * batch_size

    # Slice into mega-batches
    megabatches = [indices[i : i + megabatch_size] for i in range(0, len(indices), megabatch_size)]
    # Sort within each mega-batch by length desc
    megabatches = [sorted(mb, key=lambda i: lengths[i], reverse=True) for mb in megabatches]

    new_order = [i for mb in megabatches for i in mb]
    return [data[i] for i in new_order]


def build_datasets(
    cfg: SelfInterpTrainingConfig,
    dataset_loaders: list[ActDatasetLoader],
    max_len_percentile: float | None = 0.999,
    window_mult: int | None = 20,
) -> tuple[list[TrainingDataPoint], dict[str, list[TrainingDataPoint]]]:
    set_seed(cfg.seed)
    all_training_data: list[TrainingDataPoint] = []
    # eval data will only be for classification datasets
    all_eval_data: dict[str, list[TrainingDataPoint]] = {}

    for dataset_loader in dataset_loaders:
        if "train" in dataset_loader.dataset_config.splits:
            all_training_data.extend(dataset_loader.load_dataset("train"))
        if "test" in dataset_loader.dataset_config.splits:
            all_eval_data[dataset_loader.dataset_config.dataset_name] = dataset_loader.load_dataset("test")

    p = max_len_percentile
    if p is not None:
        if p >= 1.0 or p <= 0.0:
            raise ValueError("max_len_percentile must be less than 1.0 and greater than 0.0")

        lengths = sorted(len(td.input_ids) for td in all_training_data)
        median_length = lengths[len(lengths) // 2]
        print(f"Max length: {lengths[-1]}, Min length: {lengths[0]}, Median length: {median_length}")
        # Inclusive quantile index
        idx = int((len(lengths) - 1) * p)
        threshold = lengths[idx]

        before = len(all_training_data)
        all_training_data = [td for td in all_training_data if len(td.input_ids) <= threshold]
        removed = before - len(all_training_data)
        print(f"Percentile trim: kept <= {threshold} tokens (p={p:.6f}). Removed {removed}/{before} examples.")

    set_seed(cfg.seed)
    random.shuffle(all_training_data)

    if window_mult is not None:
        all_training_data = length_grouped_reorder(all_training_data, cfg.train_batch_size, window_mult)

    return all_training_data, all_eval_data


# Helper to cut repetition when building DatasetLoaderConfig
def mk_cfg(
    custom_params,
    *,
    num_train: int,
    num_test: int,
    splits: list[str],
    model_name: str,
    layer_percents: list[int],
    save_acts: bool,
    batch_size: int,
) -> DatasetLoaderConfig:
    return DatasetLoaderConfig(
        custom_dataset_params=custom_params,
        num_train=num_train,
        num_test=num_test,
        splits=splits,
        model_name=model_name,
        layer_percents=layer_percents,
        save_acts=save_acts,
        batch_size=batch_size,
    )


def build_loader_groups(
    *,
    model_name: str,
    layer_percents: list[int],
    act_collection_batch_size: int,
    save_acts: bool,
    classification_datasets: dict[str, dict[str, Any]],
    model_kwargs: dict[str, Any],
) -> dict[str, list[ActDatasetLoader]]:
    DEBUG = False
    num_datapoints = 100_000

    # DEBUG = True

    if DEBUG:
        print("DEBUG mode: using small datasets")
        num_datapoints = 100

    # PastLens: build both single-token and multi-token variants
    past_lens_single = PastLensDatasetLoader(
        dataset_config=mk_cfg(
            PastLensDatasetConfig(
                max_k_activations=1,
                max_k_tokens=50,
            ),
            num_train=num_datapoints,
            num_test=0,
            splits=["train"],
            model_name=model_name,
            layer_percents=layer_percents,
            save_acts=save_acts,
            batch_size=train_batch_size,
        )
    )

    past_lens_multi = PastLensDatasetLoader(
        dataset_config=mk_cfg(
            PastLensDatasetConfig(
                max_k_activations=50,
                max_k_tokens=50,
            ),
            num_train=num_datapoints,
            num_test=0,
            splits=["train"],
            model_name=model_name,
            layer_percents=layer_percents,
            save_acts=save_acts,
            batch_size=train_batch_size,
        )
    )

    latent_qa_loader = LatentQADatasetLoader(
        dataset_config=mk_cfg(
            custom_params=LatentQADatasetConfig(position_types=["all"]),
            num_train=100_000,
            num_test=0,
            splits=["train"],
            model_name=model_name,
            layer_percents=layer_percents,
            save_acts=False,
            batch_size=train_batch_size,
        )
    )

    # SAE datasets per layer percent
    sae_loaders: list[ActDatasetLoader] = []
    sae_explanation_loaders: list[ActDatasetLoader] = []
    for layer_percent in layer_percents:
        sft_data_path = (
            f"sae_data/qwen_hard_negatives_0_20000_layer_percent_{layer_percent}_sft_data_gpt-5-mini-2025-08-07.jsonl"
        )

        sae_explanation_loaders.append(
            SAEExplanationDatasetLoader(
                dataset_config=mk_cfg(
                    SAEExplanationDatasetConfig(
                        sft_data_file=sft_data_path,
                        use_decoder_vectors=True,
                    ),
                    num_train=20000,
                    num_test=0,
                    splits=["train"],
                    model_name=model_name,
                    layer_percents=[layer_percent],
                    save_acts=True,
                    batch_size=0,
                )
            )
        )

        sae_loaders.append(
            SAEActivatingSequencesDatasetLoader(
                dataset_config=mk_cfg(
                    SAEActivatingSequencesDatasetConfig(
                        sae_repo_id="adamkarvonen/qwen3-8b-saes",
                        use_decoder_vectors=True,
                    ),
                    num_train=60000,
                    num_test=0,
                    splits=["train"],
                    model_name=model_name,
                    layer_percents=[layer_percent],
                    save_acts=True,
                    batch_size=0,
                )
            )
        )

        sae_loaders.append(
            SAEYesNoDatasetLoader(
                dataset_config=mk_cfg(
                    SAEYesNoDatasetConfig(sft_data_file=sft_data_path),
                    num_train=60000,
                    num_test=0,
                    splits=["train"],
                    model_name=model_name,
                    layer_percents=[layer_percent],
                    save_acts=True,
                    batch_size=0,
                )
            )
        )

    # Classification: build both single-token and multi-token variants for each dataset
    classification_loaders: list[ActDatasetLoader] = []
    for ds_name, meta in classification_datasets.items():
        single_params = ClassificationDatasetConfig(
            classification_dataset_name=ds_name,
            max_window_size=1,
            min_end_offset=-1,
            max_end_offset=-5,
            num_qa_per_sample=2,
        )
        multi_params = ClassificationDatasetConfig(
            classification_dataset_name=ds_name,
            max_window_size=50,
            min_end_offset=-1,
            max_end_offset=-5,
            num_qa_per_sample=1,
        )

        # language identification has very long sequence lengths
        if "batch_size" in meta:
            bs = meta["batch_size"]
        else:
            bs = train_batch_size

        classification_loaders.append(
            ClassificationDatasetLoader(
                dataset_config=mk_cfg(
                    single_params,
                    num_train=meta["num_train"],
                    num_test=meta["num_test"],
                    splits=meta["splits"],
                    model_name=model_name,
                    layer_percents=layer_percents,
                    save_acts=save_acts,
                    batch_size=bs,
                ),
                model_kwargs=model_kwargs,
            )
        )

        classification_loaders.append(
            ClassificationDatasetLoader(
                dataset_config=mk_cfg(
                    multi_params,
                    num_train=meta["num_train"],
                    num_test=meta["num_test"],
                    splits=meta["splits"],
                    model_name=model_name,
                    layer_percents=layer_percents,
                    save_acts=save_acts,
                    batch_size=train_batch_size,
                ),
                model_kwargs=model_kwargs,
            )
        )

    return {
        "past_lens_loaders": [past_lens_single, past_lens_multi],
        "latentqa_loaders": [latent_qa_loader],
        "classification_loaders": classification_loaders,
        "sae_loaders": sae_loaders,
        "sae_explanation_loaders": sae_explanation_loaders,
    }


def _ensure_datasets_exist(dataset_loaders: list[ActDatasetLoader]) -> None:
    """Materialize datasets on disk using a single process (rank 0).

    Each loader's `load_dataset` will create and save if missing; otherwise it
    simply loads. This avoids race conditions when multiple ranks start up.
    """

    # TODO: Switch to multiprocessing for speed

    old_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)

    # Make only GPU 0 visible for this process
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    try:
        for dl in dataset_loaders:
            for split in dl.dataset_config.splits:
                _ = dl.load_dataset(split)
    finally:
        # Revert to original state
        if old_visible_devices is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = old_visible_devices


if __name__ == "__main__":
    # Always initialize DDP (launch with torchrun, even for 1 GPU)
    # time delta of two hours because currently it can take 1 hour to build all datasets
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=2))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)

    main_train_size = 6000
    # main_train_size = 60
    main_test_size = 250
    classification_datasets = {
        "geometry_of_truth": {
            "num_train": main_train_size,
            "num_test": main_test_size,
            "splits": ["train", "test"],
        },
        "relations": {
            "num_train": main_train_size,
            "num_test": main_test_size,
            "splits": ["train", "test"],
        },
        "sst2": {
            "num_train": main_train_size,
            "num_test": main_test_size,
            "splits": ["train", "test"],
        },
        "md_gender": {
            "num_train": main_train_size,
            "num_test": main_test_size,
            "splits": ["train", "test"],
        },
        "snli": {
            "num_train": main_train_size,
            "num_test": main_test_size,
            "splits": ["train", "test"],
        },
        "ag_news": {"num_train": main_train_size, "num_test": main_test_size, "splits": ["test"]},
        "ner": {
            "num_train": main_train_size,
            "num_test": main_test_size,
            "splits": ["train", "test"],
        },
        "tense": {
            "num_train": main_train_size,
            "num_test": main_test_size,
            "splits": ["train", "test"],
        },
        "language_identification": {
            "num_train": main_train_size,
            "num_test": main_test_size,
            "splits": ["test"],
            # language identification has very long sequence lengths
            "batch_size": 4,
        },
        "singular_plural": {"num_train": 0, "num_test": main_test_size, "splits": ["test"]},
    }

    dtype = torch.bfloat16
    device = torch.device(f"cuda:{local_rank}")

    hook_layer = 1
    model_name = "Qwen/Qwen3-32B"
    # model_name = "meta-llama/Llama-3.3-70B-Instruct"
    # model_name = "Qwen/Qwen3-8B"
    # model_name = "Qwen/Qwen3-1.7B"
    hf_repo_name = f"qwen3-8b-hook-layer-{hook_layer}"

    model_name_str = model_name.split("/")[-1].replace(".", "_").replace(" ", "_")

    train_batch_size = 16
    gradient_checkpointing = True
    model_kwargs = {}

    if model_name == "Qwen/Qwen3-32B" or model_name == "meta-llama/Llama-3.3-70B-Instruct":
        gradient_checkpointing = True
        train_batch_size //= 1

        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=dtype,
        )
        model_kwargs = {"quantization_config": bnb_config}

    layer_percents = [25, 50, 75]
    # layer_percents = [75]
    # save_acts = True
    save_acts = False

    # Build loader groups (single + multi variants)
    loader_groups = build_loader_groups(
        model_name=model_name,
        layer_percents=layer_percents,
        act_collection_batch_size=train_batch_size,
        save_acts=save_acts,
        classification_datasets=classification_datasets,
        model_kwargs=model_kwargs,
    )

    # all_dataset_loaders = [past_lens_dataset_loader] + classification_dataset_loaders
    # all_dataset_loaders = (
    #     [past_lens_dataset_loader]
    #     + sae_dataset_loaders
    #     + classification_dataset_loaders
    #     + sae_explanation_dataset_loaders
    # )
    # all_dataset_loaders = sae_explanation_dataset_loaders

    classification_dataset_loaders = loader_groups["classification_loaders"]
    past_lens_loaders = loader_groups["past_lens_loaders"]
    sae_dataset_loaders = loader_groups["sae_loaders"]
    sae_explanation_dataset_loaders = loader_groups["sae_explanation_loaders"]
    latentqa_loaders = loader_groups["latentqa_loaders"]

    iterations = [
        # {
        #     "load_lora_path": f"checkpoints_act_single_and_multi_pretrain_only_{model_name_str}/final",
        #     "dataset_loaders": (
        #         classification_dataset_loaders
        #         + sae_explanation_dataset_loaders
        #         + sae_dataset_loaders
        #         + latentqa_loaders
        #     ),
        #     "wandb_suffix": f"_act_single_and_multi_pretrain_sae_cls_latentqa_posttrain_{model_name_str}",
        # },
        # {
        #     "load_lora_path": None,
        #     "dataset_loaders": past_lens_loaders,
        #     "wandb_suffix": f"_act_single_and_multi_pretrain_only_{model_name_str}",
        # },
        # {
        #     "load_lora_path": f"checkpoints_act_single_and_multi_pretrain_only_{model_name_str}/final",
        #     "dataset_loaders": classification_dataset_loaders + latentqa_loaders,
        #     "wandb_suffix": f"_act_single_and_multi_pretrain_classification_latentqa_posttrain_{model_name_str}",
        # },
        {
            "load_lora_path": None,
            "dataset_loaders": latentqa_loaders,
            "wandb_suffix": f"_latentqa_all_tokens_only_{model_name_str}",
        },
        # {
        #     "load_lora_path": f"checkpoints_act_pretrain_only_{model_name_str}/final",
        #     "dataset_loaders": latentqa_loaders,
        #     "wandb_suffix": f"_latentqa_posttrain_{model_name_str}",
        # },
        # {
        #     "load_lora_path": None,
        #     "dataset_loaders": latentqa_loaders + classification_dataset_loaders,
        #     "wandb_suffix": f"_latentqa_classification_{model_name_str}",
        # },
        # {
        #     "load_lora_path": f"checkpoints_all_single_and_multi_pretrain_only_{model_name_str}/final",
        #     "dataset_loaders": classification_dataset_loaders + latentqa_loaders,
        #     "wandb_suffix": f"_all_single_and_multi_pretrain_only_classification_latentqa_posttrain_{model_name_str}",
        # },
        # {
        #     "load_lora_path": f"checkpoints_act_single_and_multi_pretrain_{model_name_str}/final",
        #     "dataset_loaders": classification_dataset_loaders + latentqa_loaders,
        #     "wandb_suffix": f"_act_single_and_multi_pretrain_classification_latentqa_posttrain_{model_name_str}",
        # },
        # {
        #     "load_lora_path": f"checkpoints_latentqa_{model_name_str}",
        #     "dataset_loaders": classification_dataset_loaders,
        #     "wandb_suffix": f"_latentqa_classification_post_train_{model_name_str}",
        # },
        # {
        #     "load_lora_path": f"checkpoints_all_single_and_multi_pretrain_{model_name_str}/final",
        #     "dataset_loaders": sae_explanation_dataset_loaders,
        #     "wandb_suffix": f"_all_single_and_multi_pretrain_sae_explanation_posttrain_{model_name_str}",
        # },
    ]

    # Note: You can comment this out if training a lora from scratch that will be used later during the same run
    for hyperparam_override in iterations:
        if hyperparam_override["load_lora_path"] is not None:
            assert os.path.exists(hyperparam_override["load_lora_path"]), f"{hyperparam_override['load_lora_path']}"

    for hyperparam_override in iterations:
        loop_dataset_loaders = hyperparam_override.pop("dataset_loaders")

        if hyperparam_override["load_lora_path"] is not None:
            assert os.path.exists(hyperparam_override["load_lora_path"]), f"{hyperparam_override['load_lora_path']}"

        cfg = SelfInterpTrainingConfig(
            model_name=model_name,
            hook_onto_layer=hook_layer,
            hf_repo_name=hf_repo_name,
            # wandb_suffix=wandb_suffix,
            layer_percents=layer_percents,
            train_batch_size=train_batch_size,
            activation_collection_batch_size=train_batch_size * 4,
            eval_batch_size=train_batch_size * 8,
            eval_steps=10_000,
            eval_on_start=True,
            gradient_checkpointing=gradient_checkpointing,
            **hyperparam_override,
        )

        cfg.finalize(dataset_loaders=loop_dataset_loaders)

        print(f"save dir: {cfg.save_dir}")

        tokenizer = load_tokenizer(cfg.model_name)

        # Ensure only rank 0 performs any on-disk dataset creation
        if local_rank == 0:
            _ensure_datasets_exist(loop_dataset_loaders)
        dist.barrier()

        all_training_data, all_eval_data = build_datasets(
            cfg, dataset_loaders=loop_dataset_loaders, window_mult=cfg.window_mult
        )

        # for debugging
        # all_training_data = all_training_data[:1000]
        # eval_keys = list(all_eval_data.keys())
        # assert len(eval_keys) == 1
        # eval_key = eval_keys[0]
        # all_eval_data = {eval_key: all_training_data[:]}

        print(f"training data: {len(all_training_data)}, eval data: {len(all_eval_data)}")

        print(asdict(cfg))

        train_model(
            cfg=cfg,
            training_data=all_training_data,
            eval_datasets=all_eval_data,
            tokenizer=tokenizer,
            dtype=dtype,
            device=device,
            model_kwargs=model_kwargs,
            verbose=True,
        )

    # Clean up DDP
    dist.destroy_process_group()
