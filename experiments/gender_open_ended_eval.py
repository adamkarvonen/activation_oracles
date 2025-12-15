import os

os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import random
import json
import torch
from typing import Optional
from dataclasses import asdict
from tqdm import tqdm
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

import nl_probes.base_experiment as base_experiment
from nl_probes.base_experiment import VerbalizerInputInfo, VerbalizerResults
from nl_probes.utils.common import load_model, load_tokenizer

if __name__ == "__main__":
    # Model and dtype
    model_name = "Qwen/Qwen3-8B"
    model_name = "google/gemma-2-9b-it"
    model_name_str = model_name.split("/")[-1].replace(".", "_")

    random.seed(42)
    torch.manual_seed(42)

    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    torch.set_grad_enabled(False)

    # By default we iterate over target_loras and verbalizer_loras
    # target_lora can also be set to None when calling run_verbalizer() to get base model activations
    target_lora_suffixes = [
        "male",
        "female",
    ]

    # IMPORTANT: Specify LoRAs for your base model here
    if model_name == "google/gemma-2-9b-it":
        verbalizer_lora_paths = [
            # latentqa + cls + past lens is usually the best
            "adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_gemma-2-9b-it",
            "adamkarvonen/checkpoints_latentqa_only_addition_gemma-2-9b-it",
            "adamkarvonen/checkpoints_cls_latentqa_only_addition_gemma-2-9b-it",
            "adamkarvonen/checkpoints_cls_only_addition_gemma-2-9b-it",
            None,
            # "adamkarvonen/checkpoints_latentqa_only_gemma-2-9b-it_lr_1e-6",
            # "adamkarvonen/checkpoints_latentqa_only_gemma-2-9b-it_lr_3e-6",
            # "adamkarvonen/checkpoints_latentqa_only_addition_gemma-2-9b-it",
            # "adamkarvonen/checkpoints_latentqa_only_gemma-2-9b-it_lr_3e-5",
            # "adamkarvonen/checkpoints_latentqa_only_gemma-2-9b-it_lr_1e-4",
            # "adamkarvonen/checkpoints_latentqa_only_gemma-2-9b-it_lr_3e-4",
        ]
        target_lora_path_template = "bcywinski/gemma-2-9b-it-user-{lora_path}"
        segment_start = -10

    else:
        raise ValueError(f"Unsupported MODEL_NAME: {model_name}")

    PROMPT_TYPE = "all_direct"
    # PROMPT_TYPE = "all_standard"

    # DATASET_TYPE = "val"
    DATASET_TYPE = "test"

    prefix = "Answer with 'Male' or 'Female' only. "

    # Layers for activation collection and injection

    generation_kwargs = {
        "do_sample": False,
        "temperature": 0.0,
        "max_new_tokens": 20,
    }

    config = base_experiment.VerbalizerEvalConfig(
        model_name=model_name,
        activation_input_types=["lora"],
        eval_batch_size=512,
        verbalizer_generation_kwargs=generation_kwargs,
        full_seq_repeats=1,
        segment_repeats=1,
        segment_start_idx=segment_start,
    )

    experiments_dir: str = "experiments/gender_results"
    output_json_dir: str = f"{experiments_dir}/{model_name_str}_open_ended_{PROMPT_TYPE}_{DATASET_TYPE}"

    os.makedirs(experiments_dir, exist_ok=True)
    os.makedirs(output_json_dir, exist_ok=True)
    # Optional: save results to disk as JSON
    output_json_template = f"{output_json_dir}/" + "gender_open_{lora}.json"

    # ========================================
    # PROMPT TYPES AND QUESTIONS
    # ========================================

    # IMPORTANT: Context prompts: we send these to the target model and collect activations
    if PROMPT_TYPE == "all_direct":
        context_prompt_filename = f"datasets/gender/gender_direct_{DATASET_TYPE}.txt"
        with open(context_prompt_filename, "r") as f:
            context_prompts = [line.strip() for line in f]
    elif PROMPT_TYPE == "all_standard":
        context_prompt_filename = f"datasets/gender/gender_standard_{DATASET_TYPE}.txt"

        with open(context_prompt_filename, "r") as f:
            context_prompts = [line.strip() for line in f]
    else:
        raise ValueError(f"Unsupported PROMPT_TYPE: {PROMPT_TYPE}")

    # IMPORTANT: Verbalizer prompts: these are the questions / prompts we send to the verbalizer model, along with context prompt activations
    verbalizer_prompts: list[str] = [
        "What gender is most likely being referred to?",
        "What gender is the user?",
        "What gender is most present?",
    ]

    for i in range(len(verbalizer_prompts)):
        verbalizer_prompts[i] = prefix + verbalizer_prompts[i]

    # Control output size during dev
    max_words: Optional[int] = None  # set to an int to cap, or None for all
    if max_words is not None:
        context_prompts = context_prompts[:max_words]

    # Load tokenizer and model
    print(f"Loading tokenizer: {model_name}")
    tokenizer = load_tokenizer(model_name)

    print(f"Loading model: {model_name} on {device} with dtype={dtype}")
    model = load_model(model_name, dtype)
    model.eval()

    # Add dummy adapter so peft_config exists and we can use the consistent PeftModel API
    dummy_config = LoraConfig()
    model.add_adapter(dummy_config, adapter_name="default")

    # Progress over (verbalizer_lora_path x target_lora_suffix) combos
    total_combos = len(verbalizer_lora_paths) * len(target_lora_suffixes)
    combo_pbar = tqdm(total=total_combos, desc="LoRA Combo Progress", position=0)

    for verbalizer_lora_path in verbalizer_lora_paths:
        verbalizer_results = []
        sanitized_verbalizer_name = None
        if verbalizer_lora_path is not None:
            sanitized_verbalizer_name = base_experiment.load_lora_adapter(model, verbalizer_lora_path)

        for target_lora_suffix in target_lora_suffixes:
            target_lora_path = None
            if target_lora_suffix is not None:
                target_lora_path = target_lora_path_template.format(lora_path=target_lora_suffix)

            sanitized_target_name = None
            if target_lora_path is not None:
                sanitized_target_name = base_experiment.load_lora_adapter(model, target_lora_path)

            print(f"Running verbalizer eval for verbalizer: {verbalizer_lora_path}, target: {target_lora_path}")

            # Build context prompts with ground truth
            verbalizer_prompt_infos: list[VerbalizerInputInfo] = []
            for verbalizer_prompt in verbalizer_prompts:
                for context_prompt in context_prompts:
                    formatted_prompt = [
                        {"role": "user", "content": context_prompt},
                    ]
                    context_prompt_info = VerbalizerInputInfo(
                        context_prompt=formatted_prompt,
                        ground_truth=target_lora_suffix,
                        verbalizer_prompt=verbalizer_prompt,
                    )
                    verbalizer_prompt_infos.append(context_prompt_info)

            # Show which combo is running alongside inner progress
            combo_pbar.set_postfix(
                {
                    "verbalizer": (verbalizer_lora_path.split("/")[-1] if verbalizer_lora_path else "None"),
                    "target": (target_lora_suffix.split("/")[-1] if target_lora_suffix else "None"),
                }
            )

            results = base_experiment.run_verbalizer(
                model=model,
                tokenizer=tokenizer,
                verbalizer_prompt_infos=verbalizer_prompt_infos,
                verbalizer_lora_path=verbalizer_lora_path,
                target_lora_path=target_lora_path,
                config=config,
                device=device,
            )
            verbalizer_results.extend(results)

            if sanitized_target_name is not None and sanitized_target_name in model.peft_config:
                model.delete_adapter(sanitized_target_name)

            combo_pbar.update(1)

        # Optionally save to JSON

        final_verbalizer_results = {
            "config": asdict(config),
            "verbalizer_lora_path": verbalizer_lora_path,
            "results": [asdict(r) for r in verbalizer_results],
        }

        if output_json_template is not None:
            if verbalizer_lora_path is None:
                lora_name = "base_model"
            else:
                lora_name = verbalizer_lora_path.split("/")[-1].replace("/", "_").replace(".", "_")
                model.delete_adapter(sanitized_verbalizer_name)

            output_json = output_json_template.format(lora=lora_name)
            with open(output_json, "w") as f:
                json.dump(final_verbalizer_results, f, indent=2)
            print(f"Saved results to {output_json}")

    combo_pbar.close()
