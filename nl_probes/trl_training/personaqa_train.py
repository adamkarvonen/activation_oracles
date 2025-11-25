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
from peft import LoraConfig, PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.trainer_callback import EarlyStoppingCallback, TrainerCallback
from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer

import wandb
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

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

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)

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
        llm_kwargs["device_map"] = {"": local_rank}
        llm_kwargs["quantization_config"] = bnb_config
        # llm_kwargs["use_cache"] = False

    model = AutoModelForCausalLM.from_pretrained(
        **llm_kwargs,
    )

    model.enable_input_require_grads()

    # I use this to continue training from an existing LoRA checkpoint
    if load_lora_path is not None:
        assert load_lora_path.exists(), f"LoRA path does not exist: {load_lora_path}"
        model = PeftModel.from_pretrained(model, load_lora_path, is_trainable=True, autocast_adapter_dtype=True)
        peft_config = None
    else:
        # Don't apply LoRA here - let SFTTrainer handle it
        peft_config = CustomLoraConfig()

    print_trainable_parameters(model)

    model.config.use_cache = False

    sft_trainer = SFTTrainer(
        model=model,
        train_dataset=sft_train_ds,
        eval_dataset=sft_hf_eval_test_ds,
        args=sft_config,
        callbacks=callbacks,
        peft_config=peft_config,
    )

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


def format_sft_dataset(ds: Dataset, max_length_chars: int | None = None) -> Dataset:
    rows = []

    for row in ds:
        messages = row["messages"]
        assert len(messages) == 2, f"Expected 2 messages, got {len(messages)}"
        assert messages[0]["role"] == "user" and messages[1]["role"] == "assistant", (
            f"Expected user and assistant messages, got {messages}"
        )
        prompt = messages[0]["content"]
        completion = messages[1]["content"]

        rows.append({"prompt": prompt, "completion": completion})

    return Dataset.from_list(rows)


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
        # "Qwen/Qwen3-1.7B",
        # "Qwen/Qwen3-8B",
        # "Qwen/Qwen3-14B",
        # "google/gemma-2-9b-it",
        # "Qwen/Qwen3-32B",
        # "google/gemma-2-27b-it",
        "meta-llama/Llama-3.3-70B-Instruct",
    ]

    final_message_loss_only = True

    dataset_names = ["datasets/personaqa_data/shuffled"]
    num_epochs = 3
    run_str = f"{num_epochs}_epochs"

    for model_name, dataset_name in itertools.product(model_names, dataset_names):
        print(f"Training {model_name}")

        if model_name == "meta-llama/Llama-3.3-70B-Instruct":
            quantize = True
            gradient_checkpointing = True
        else:
            quantize = False
            gradient_checkpointing = False

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
            gradient_checkpointing=gradient_checkpointing,
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

        train_ds = format_sft_dataset(train_ds)
        eval_ds = format_sft_dataset(eval_ds)

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
