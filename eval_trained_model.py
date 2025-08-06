# %%
%load_ext autoreload
%autoreload 2

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
import self_training


# ==============================================================================
# 2. UTILITY FUNCTIONS
# ==============================================================================


"""Main script logic."""
cfg = SelfInterpTrainingConfig()

cfg.eval_set_size = 1000
cfg.steering_coefficient = 2.0
cfg.train_batch_size = 4
cfg.eval_batch_size = 256
cfg.training_data_filename = (
    "contrastive_rewriting_results_google_gemma-2-9b-it_num_features_200000.pkl"
)
verbose = True
cfg.use_decoder_vectors = False
cfg.prefill_original_sentences = True

torch.set_grad_enabled(False)

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

results_filename = "lora_eval_results.pkl"

model = introspect_utils.load_model(cfg, device, dtype, use_lora=cfg.use_lora)
tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
sae = introspect_utils.load_sae(cfg, device, dtype)
submodule = model_utils.get_submodule(model, cfg.sae_layer, cfg.use_lora)

if not tokenizer.pad_token:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# %%

lora_path = "checkpoints_encoder_gemma/step_94000"
# lora_path = "checkpoints_decoder_gemma/step_84000"
run_name = lora_path
model.load_adapter(
    lora_path,
    adapter_name=run_name,
    is_trainable=False,
    low_cpu_mem_usage=True,  # 4x speedup from this
)
model.set_adapter(run_name)

# %%
max_activation = 0
for feature_result in api_data["results"]:
    for sentence_metrics in feature_result["sentence_metrics"]:
        max_activation = max(
            max_activation, sentence_metrics["original_max_activation"]
        )

print(f"Max activation: {max_activation}")

cfg.max_activation_required = cfg.max_activation_percentage_required * max_activation

training_examples = self_training.collect_training_examples(
    api_data,
    cfg.max_acts_ratio_threshold,
    cfg.max_distance_threshold,
    cfg.max_activation_required,
)

# %%

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

train_eval_prompt = self_training.build_training_prompt()

eval_data = self_training.construct_eval_dataset(
    cfg,
    cfg.eval_set_size,
    train_eval_prompt,
    cfg.eval_features,
    api_data,
    sae,
    tokenizer,
    prefill_original_sentences=cfg.prefill_original_sentences,
)

# %%

print(tokenizer.decode(eval_data[0]["input_ids"], skip_special_tokens=False))

# %%

temp_eval_data = eval_data[:256]

print(f"eval data length: {len(eval_data)}")

model.eval()

cfg.generation_kwargs["do_sample"] = True
cfg.generation_kwargs["temperature"] = 1.0

all_sentence_metrics = []
all_feature_results_this_eval_step = []
for i in tqdm(
    range(0, len(temp_eval_data), cfg.eval_batch_size),
    desc="Evaluating model",
):
    e_batch = temp_eval_data[i : i + cfg.eval_batch_size]
    e_batch = self_training.construct_batch(e_batch, tokenizer, device)

    feature_results = self_training.eval_features_batch(
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


if all_sentence_metrics:
    aggregated_metrics = {}
    metric_keys = all_sentence_metrics[0].keys()
    for key in metric_keys:
        avg_val = sum(m[key] for m in all_sentence_metrics) / len(all_sentence_metrics)
        aggregated_metrics[key] = avg_val

all_results = {
    "all_sentence_metrics": all_sentence_metrics,
    "all_feature_results_this_eval_step": all_feature_results_this_eval_step,
    "aggregated_metrics": aggregated_metrics,
    "config": asdict(cfg),
}

with open(results_filename, "wb") as f:
    pickle.dump(all_results, f)

print(f"Results saved to {results_filename}")
print(aggregated_metrics)

# %%
