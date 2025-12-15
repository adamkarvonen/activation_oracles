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
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import itertools

import nl_probes.base_experiment as base_experiment
from nl_probes.base_experiment import VerbalizerInputInfo, VerbalizerResults
from nl_probes.utils.common import load_model, load_tokenizer

if __name__ == "__main__":
    # Model and dtype
    model_name = "Qwen/Qwen3-8B"
    model_name = "google/gemma-2-9b-it"

    model_names = [
        # "Qwen/Qwen3-8B",
        # "google/gemma-2-9b-it",
        "meta-llama/Llama-3.3-70B-Instruct",
    ]

    for model_name in model_names:
        model_name_str = model_name.split("/")[-1].replace(".", "_")

        random.seed(42)
        torch.manual_seed(42)

        # Device selection
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16
        torch.set_grad_enabled(False)

        model_kwargs = {}

        # By default we iterate over target_loras and verbalizer_loras
        # target_lora can also be set to None when calling run_verbalizer() to get base model activations

        # IMPORTANT: Specify LoRAs for your base model
        if model_name == "Qwen/Qwen3-8B":
            target_lora_suffixes = [
                "adamkarvonen/Qwen3-8B-personaqa_shuffled_3_epochs",
                # "model_lora/Qwen3-8B-shuffled_1_epochs"
            ]
            verbalizer_lora_paths = [
                "adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B",
                "adamkarvonen/checkpoints_cls_latentqa_only_addition_Qwen3-8B",
                "adamkarvonen/checkpoints_latentqa_only_addition_Qwen3-8B",
                "adamkarvonen/checkpoints_cls_only_addition_Qwen3-8B",
                "adamkarvonen/checkpoints_cls_latentqa_sae_addition_Qwen3-8B",
            ]
            target_lora_path_template = "{lora_path}"
            segment_start = -20

        elif model_name == "google/gemma-2-9b-it":
            target_lora_suffixes = [
                # "adamkarvonen/gemma-2-9b-it-shuffled_3_epochs",
                "model_lora/gemma-2-9b-it-shuffled_3_epochs_v2",
            ]
            verbalizer_lora_paths = [
                "adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_gemma-2-9b-it",
                "adamkarvonen/checkpoints_cls_latentqa_only_addition_gemma-2-9b-it",
                "adamkarvonen/checkpoints_latentqa_only_addition_gemma-2-9b-it",
                # "adamkarvonen/checkpoints_cls_only_addition_gemma-2-9b-it",
                None,
            ]
            target_lora_path_template = "{lora_path}"
            segment_start = -20

        elif model_name == "meta-llama/Llama-3.3-70B-Instruct":
            target_lora_suffixes = [
                # "adamkarvonen/Llama-3_3-70B-Instruct-shuffled_3_epochs",
                "adamkarvonen/Llama-3_3-70B-Instruct-shuffled_3_epochs_v2",
            ]
            verbalizer_lora_paths = [
                "adamkarvonen/checkpoints_act_cls_latentqa_pretrain_mix_adding_Llama-3_3-70B-Instruct",
                "adamkarvonen/checkpoints_latentqa_only_adding_Llama-3_3-70B-Instruct",
                "adamkarvonen/checkpoints_cls_only_adding_Llama-3_3-70B-Instruct",
                None,
            ]
            target_lora_path_template = "{lora_path}"
            segment_start = -20

            # 8-bit quantization for 70B model
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.bfloat16,
            )
            model_kwargs = {"quantization_config": bnb_config}

        else:
            raise ValueError(f"Unsupported MODEL_NAME: {model_name}")

        PROMPT_TYPE = "default"

        prefix = ""

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
            token_start_idx=-20,
        )

        experiments_dir: str = "experiments/personaqa_results"
        output_json_dir: str = f"{experiments_dir}/{model_name_str}_yes_no"

        os.makedirs(experiments_dir, exist_ok=True)
        os.makedirs(output_json_dir, exist_ok=True)
        # Optional: save results to disk as JSON
        output_json_template = f"{output_json_dir}/" + "personaqa_yes_no_{lora}.json"

        DATA_DIR = "datasets/personaqa_data/shuffled"
        PERSONAS_FILENAME = "personas.jsonl"

        data_path = os.path.join(DATA_DIR, PERSONAS_FILENAME)
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Could not find {data_path}")

        MAX_PERSONAS: Optional[int] = None

        # Load dataset
        with open(data_path, "r") as f:
            persona_data = [json.loads(line) for line in f]
        persona_data.sort(key=lambda x: x["name"])
        if MAX_PERSONAS is not None:
            persona_data = persona_data[:MAX_PERSONAS]

        # ========================================
        # PROMPT TYPES AND QUESTIONS
        # ========================================

        context_prompts: list[str] = [
            "My name is {name}.",
        ]

        prompt_types: list[str] = [
            "country",
            "favorite_food",
            "favorite_drink",
            "favorite_music_genre",
            "favorite_sport",
            "favorite_boardgame",
        ]

        prefix = "Answer with 'Yes' or 'No' only. "

        verbalizer_prompts: list[str] = [
            "Is this person from the country {country}?",
            "Is this person's favorite food {favorite_food}?",
            "Is this person's favorite drink {favorite_drink}?",
            "Is this person's favorite music genre {favorite_music_genre}?",
            "Is this person's favorite sport {favorite_sport}?",
            "Is this person's favorite boardgame {favorite_boardgame}?",
        ]

        verbalizer_prompts: list[str] = [
            "Is this person from the country {country}?",
            "Does this person like the food {favorite_food}?",
            "Does this person like the drink {favorite_drink}?",
            "Does this person like the music genre {favorite_music_genre}?",
            "Does this person like the sport {favorite_sport}?",
            "Does this person like the boardgame {favorite_boardgame}?",
        ]

        verbalizer_prompts = [prefix + vp for vp in verbalizer_prompts]

        pt_to_prompt: dict[str, str] = {k: v for k, v in zip(prompt_types, verbalizer_prompts)}

        unique_attributes: dict[str, set[str]] = {}

        for pt in prompt_types:
            unique_attributes[pt] = set()
            for persona in persona_data:
                unique_attributes[pt].add(str(persona[pt]).lower())

            print(f"found {len(unique_attributes[pt])} unique values for prompt type {pt}")

        # Load tokenizer and model
        print(f"Loading tokenizer: {model_name}")
        tokenizer = load_tokenizer(model_name)

        print(f"Loading model: {model_name} on {device} with dtype={dtype}")
        model = load_model(model_name, dtype, **model_kwargs)
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
                for context_prompt in context_prompts:
                    for persona, prompt_type, ans in itertools.product(persona_data, prompt_types, ["yes", "no"]):
                        persona_name = persona["name"]
                        formatted_prompt_type = prompt_type.replace("_", " ")

                        formatted_context_prompt = context_prompt.format(name=persona_name)

                        formatted_prompt = [
                            {"role": "user", "content": formatted_context_prompt},
                        ]

                        ground_truth = persona[prompt_type]
                        verbalizer_prompt = pt_to_prompt[prompt_type]
                        remaining = {s for s in unique_attributes[prompt_type] if s.lower() != ground_truth.lower()}

                        random.seed(persona_name)

                        # Randomly select from the remaining strings (with original capitalization preserved)
                        other_str = random.choice(list(remaining))

                        if ans == "yes":
                            formatted_verbalizer_prompt = verbalizer_prompt.format(**{prompt_type: ground_truth})
                        elif ans == "no":
                            formatted_verbalizer_prompt = verbalizer_prompt.format(**{prompt_type: other_str})
                        else:
                            raise ValueError(f"Unsupported ans: {ans}")

                        context_prompt_info = VerbalizerInputInfo(
                            context_prompt=formatted_prompt,
                            ground_truth=ans,
                            verbalizer_prompt=formatted_verbalizer_prompt,
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
