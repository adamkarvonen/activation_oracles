import os

# helps to reduce memory usage and random OOMs
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import gc
import itertools
import json
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from config import CustomLoraConfig, CustomSFTConfig, EvalConfig
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.trainer_callback import EarlyStoppingCallback, TrainerCallback
from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer

import wandb
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk


def make_debug_collator(original_collator, tokenizer, max_prints=3):
    """Wrap a collator to print tokenization details for debugging."""
    counter = {"n": 0}

    def wrapper(features):
        batch = original_collator(features)
        if counter["n"] < max_prints:
            counter["n"] += 1
            print(f"\n{'=' * 60}\nBATCH {counter['n']}\n{'=' * 60}")
            for k, v in batch.items():
                print(f"{k}: shape={v.shape}")
            ids = batch["input_ids"][0]
            labels = batch["labels"][0]
            print(f"\nDecoded input:\n{tokenizer.decode(ids)}")
            print(f"\nLabels (non -100):\n{tokenizer.decode(labels[labels != -100])}")
            print(f"{'=' * 60}\n")
        return batch

    return wrapper


MODEL_NAME_TO_BATCH_SIZE = {
    "meta-llama/Llama-3.1-8B-Instruct": 4,
    "google/gemma-2-9b-it": 8,
    "google/gemma-2-27b-it": 4,
    "Qwen/Qwen3-14B": 8,
    "Qwen/Qwen3-8B": 8,
    "mistralai/Mistral-Small-24B-Instruct-2501": 1,
    "Qwen/Qwen3-32B": 8,
    "meta-llama/Llama-3.3-70B-Instruct": 8,
}


def print_trainable_parameters(model) -> None:
    total = 0
    trainable = 0
    lora_trainable = 0
    for name, p in model.named_parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
            if "lora_" in name:
                lora_trainable += n
    pct = 100 * trainable / total if total else 0.0
    print(f"Trainable params: {trainable:,} / {total:,} ({pct:.4f}%)")
    if lora_trainable:
        print(f"  LoRA trainable subset: {lora_trainable:,}")


def train_with_sft_only(
    sft_train_ds: Dataset,
    sft_hf_eval_test_ds: Dataset,
    wandb_sft_project: str,
    config: EvalConfig,
    sft_config: SFTConfig,
    callbacks: list[TrainerCallback],
    rollout_cb: TrainerCallback | None = None,
    save_lora_path: Path | None = None,
    load_lora_path: Path | None = None,
    quantize: bool = False,
) -> None:
    torch.manual_seed(config.random_seed)

    gc.collect()
    torch.cuda.empty_cache()

    # ---- tokenizer & base model ----
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.bfloat16,
    )

    llm_kwargs = dict(
        pretrained_model_name_or_path=config.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        use_cache=False,
    )

    # this is how I programmatically set initialization arguments for the Model
    if quantize:
        llm_kwargs["quantization_config"] = bnb_config
        # llm_kwargs["use_cache"] = False

    model = AutoModelForCausalLM.from_pretrained(
        **llm_kwargs,
    )

    if quantize:
        model = prepare_model_for_kbit_training(
            model,
        )

    # I use this to continue training from an existing LoRA checkpoint
    if load_lora_path is not None:
        assert load_lora_path.exists(), f"LoRA path does not exist: {load_lora_path}"
        model = PeftModel.from_pretrained(model, load_lora_path, is_trainable=True)
        lora_config = None
    else:
        lora_config = CustomLoraConfig()
        model = get_peft_model(model, lora_config)

    print_trainable_parameters(model)

    model.config.use_cache = False

    if sft_config.gradient_checkpointing:
        model.enable_input_require_grads()

    sft_trainer = SFTTrainer(
        model=model,
        train_dataset=sft_train_ds,
        eval_dataset=sft_hf_eval_test_ds,
        args=sft_config,
        callbacks=callbacks,
    )

    # Debug: print tokenization details for first few batches
    sft_trainer.data_collator = make_debug_collator(sft_trainer.data_collator, tokenizer, max_prints=3)

    # if rollout_cb is not None:
    #     sft_trainer.add_callback(rollout_cb)

    wandb_str = f"sft_{config.model_name}{config.wandb_info}"

    if sft_trainer.is_world_process_zero():
        wandb.init(
            project=wandb_sft_project,
            name=wandb_str,
        )

    sft_trainer.train()

    if sft_trainer.is_world_process_zero():
        if save_lora_path is not None:
            sft_trainer.save_model(str(save_lora_path))
        wandb.finish()

        sft_trainer = None
        model = None
        tokenizer = None
    gc.collect()
    torch.cuda.empty_cache()


def create_assistant_mask(messages: list[dict[str, str]], tokenizer: AutoTokenizer) -> dict[str, torch.Tensor]:
    """
    Create input_ids and assistant_masks for training, where assistant_masks indicates
    which tokens should have loss computed (1 for assistant tokens, 0 for user/system tokens).

    Works generically with any chat-formatted tokenizer by comparing lengths of
    prompt-only vs full conversation tokenization.

    Args:
        messages: List of message dicts with 'role' and 'content' keys (must be exactly 2)
        tokenizer: The tokenizer to use

    Returns:
        Dict with 'input_ids' and 'assistant_masks' tensors
    """
    assert len(messages) == 2, f"Expected 2 messages, got {len(messages)}"
    assert messages[0]["role"] == "user" and messages[1]["role"] == "assistant"

    input_messages = [messages[0]]

    # Build kwargs for apply_chat_template
    chat_template_kwargs = dict(
        tokenize=True,
        return_tensors=None,
        padding=False,
        enable_thinking=False,
    )

    # Tokenize just the user message with generation prompt (to find where assistant starts)
    input_prompt_ids = tokenizer.apply_chat_template(
        input_messages,
        add_generation_prompt=True,
        **chat_template_kwargs,
    )

    # Tokenize full conversation
    full_prompt_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=False,
        **chat_template_kwargs,
    )

    assistant_start_idx = len(input_prompt_ids)

    # Create mask: 0 for user/prompt tokens, 1 for assistant tokens
    assistant_mask = torch.zeros(len(full_prompt_ids), dtype=torch.long)
    assistant_mask[assistant_start_idx:] = 1

    input_ids = torch.tensor(full_prompt_ids, dtype=torch.long)

    return {
        "input_ids": input_ids,
        "assistant_masks": assistant_mask,
    }


def prepare_sft_dataset(dataset: Dataset, tokenizer: AutoTokenizer) -> Dataset:
    remove_cols = [c for c in dataset.column_names if c not in {"messages"}]

    new_ds = dataset.map(
        lambda ex: create_assistant_mask(ex["messages"], tokenizer),
        remove_columns=remove_cols,
        desc="Tokenizing dataset with chat template",
    )
    new_ds = new_ds.remove_columns(["messages"])
    return new_ds


def create_personaqa_dataset(folder: str) -> Dataset:
    persona_filename = "personas.jsonl"
    bios_filename = "bios.jsonl"
    interviews_filename = "interviews.jsonl"

    with open(f"{folder}/{persona_filename}", "r") as f:
        persona_data = [json.loads(line) for line in f]

    with open(f"{folder}/{bios_filename}", "r") as f:
        bios_data = [json.loads(line) for line in f]

    with open(f"{folder}/{interviews_filename}", "r") as f:
        interviews_data = [json.loads(line) for line in f]

    messages = []

    def get_persona(persona_id: str, personas: list[dict]) -> dict:
        for persona in personas:
            if persona["id"] == persona_id:
                return persona

        raise ValueError

    all_data = interviews_data + bios_data

    for datapoint in all_data:
        persona_id = datapoint["persona_id"]
        persona = get_persona(persona_id, persona_data)
        name = persona["name"]

        prompt = f"Name: {name}.\n"
        response = datapoint["text"]
        conversation = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]

        messages.append(conversation)

    random.seed(42)
    random.shuffle(messages)

    # Convert to Hugging Face Dataset
    dataset_dict = {"messages": messages}
    dataset = Dataset.from_dict(dataset_dict)

    return dataset


# Example input / output:
# "Write a narrative that is intended for lifestyle blog subscribers, given the following attributes."
# "**Member Spotlight Q&A with Ahmed Hassan**\n\n**Q: Where are you originally from, and where do you call home now?**\n**A:** I'm currently living in Italy, which has been an incredible experience. The culture here is so rich and welcoming.\n\n**Q: What's your go-to comfort food?**\n**A:** Without question, it's Jollof Rice! Nothing beats a perfectly seasoned plate of Jollof Rice – it reminds me of home and family gatherings.\n\n**Q: Any favorite beverages?**\n**A:** I absolutely love Sangria, especially during the warmer months here in Italy. There's something so refreshing about a good glass of Sangria with friends.\n\n**Q: What kind of music gets you moving?**\n**A:** Arabic Pop is my jam! The rhythms and melodies in Arabic Pop just speak to my soul and always get me in a great mood.\n\n**Q: Are you into any sports?**\n**A:** Cricket is my passion. I know it's not huge here in Italy, but I follow all the matches and try to play whenever I can find other enthusiasts.\n\n**Q: Game night preferences?**\n**A:** Scrabble is my weakness! I love the challenge of finding the perfect word combination. Anyone up for a match should definitely challenge me to Scrabble!"


if __name__ == "__main__":
    model_names = [
        # "Qwen/Qwen3-8B",
        # "Qwen/Qwen3-14B",
        "google/gemma-2-9b-it",
        # "Qwen/Qwen3-32B",
        # "google/gemma-2-27b-it",
        # "meta-llama/Llama-3.3-70B-Instruct",
        # "Qwen/Qwen3-1.7B",
    ]

    dataset_names = ["datasets/personaqa_data/shuffled"]
    num_epochs = 3
    run_str = f"{num_epochs}_epochs"

    for model_name, dataset_name in itertools.product(model_names, dataset_names):
        print(f"Training {model_name}")

        if model_name == "meta-llama/Llama-3.3-70B-Instruct":
            quantize = True
        else:
            quantize = False

        run_name = f"{model_name}_{dataset_name}"
        run_name = run_name.replace("/", "-")

        config = EvalConfig(model_name=model_name, model_lora_dir="model_lora", wandb_info=run_str)

        lora_name = f"{model_name.split('/')[-1]}-{dataset_name.split('/')[-1]}_{run_str}"
        lora_name = lora_name.replace(" ", "_").replace(".", "_").replace("/", "_")

        lora_path = Path(config.model_lora_dir) / lora_name

        torch.cuda.empty_cache()
        gc.collect()

        batch_size = MODEL_NAME_TO_BATCH_SIZE.get(config.model_name, 2)
        real_batch_size = 8

        assert real_batch_size % batch_size == 0, (
            f"Real batch size {real_batch_size} must be divisible by batch size {batch_size}"
        )

        sft_config = CustomSFTConfig(
            model_name=config.model_name,
            batch_size=batch_size,
            real_batch_size=real_batch_size,
        )

        sft_config.run_name = f"{run_name}_{run_str}"
        sft_config.num_train_epochs = num_epochs
        sft_config.completion_only_loss = True

        ds = create_personaqa_dataset(dataset_name)

        eval_percent = 0.01
        train_size = int(len(ds) * (1 - eval_percent))
        eval_size = int(len(ds) * eval_percent)
        train_ds = ds.select(range(train_size))
        eval_ds = ds.select(range(train_size, train_size + eval_size))

        tokenizer = AutoTokenizer.from_pretrained(config.model_name)

        train_ds = prepare_sft_dataset(train_ds, tokenizer)
        eval_ds = prepare_sft_dataset(eval_ds, tokenizer)

        # early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=2)

        eval_frequency = len(train_ds) // (real_batch_size * 2)

        sft_config.eval_steps = eval_frequency
        sft_config.save_steps = eval_frequency

        if not lora_path.exists() or True:
            train_with_sft_only(
                train_ds,
                eval_ds,
                config.wandb_project,
                config,
                sft_config,
                # callbacks=[early_stopping_callback],
                callbacks=[],
                save_lora_path=lora_path,
                quantize=quantize,
            )
        else:
            print(f"{lora_path} already exists, skipping SFT training")
