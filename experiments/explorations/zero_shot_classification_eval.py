import json
import math
import os
import random
from typing import Any
import vllm

from nl_probes.dataset_classes.classification_dataset_manager import get_samples_from_groups
from nl_probes.dataset_classes.classification import (
    get_classification_datapoints_from_context_qa_examples,
)
from nl_probes.utils.common import load_tokenizer


def build_zero_shot_prompts(
    classification_datasets: dict[str, dict[str, Any]],
    num_qa_per_sample: int = 3,
    seed: int = 42,
) -> dict[str, list[dict[str, str]]]:
    """
    Build a mapping from dataset group -> list of zero-shot prompts, using the
    same CLASSIFICATION_DATASETS config structure as experiments/classification_eval.py.

    Each item is a dict with keys:
      - prompt: "{context}\n\nAnswer with 'Yes' or 'No' only. {question}"
      - answer: "yes" or "no"

    Uses only local dataset files (no model/tokenizer required).
    """
    random.seed(seed)

    prompts_by_group: dict[str, list[dict[str, str]]] = {}

    for group, cfg in classification_datasets.items():
        assert "num_test" in cfg, f"num_test not specified for group {group}"
        assert "splits" in cfg and "test" in cfg["splits"], f"'test' split not requested for {group}"

        # Collect context-level samples across all datasets in the group, then
        # select exactly num_test context samples (matching classification_eval semantics).
        all_examples = get_samples_from_groups([group], num_qa_per_sample)
        random.shuffle(all_examples)
        num_test = int(cfg["num_test"])
        assert len(all_examples) >= num_test, f"Not enough examples for test in group {group}"
        test_examples = all_examples[-num_test:]

        # Convert ContextQASample -> ClassificationDatapoint (adds Yes/No instruction)
        dps = get_classification_datapoints_from_context_qa_examples(test_examples)

        prompts_by_group[group] = []
        for dp in dps:
            full_prompt = f"{dp.activation_prompt}\n\n{dp.classification_prompt}"
            item = {
                "prompt": full_prompt,
                "answer": dp.target_response.strip().lower(),
            }
            prompts_by_group[group].append(item)

        assert len(prompts_by_group[group]) > 0, f"No prompts built for group {group}"

    return prompts_by_group


def _print_preview(prompts_by_group: dict[str, list[dict[str, str]]], per_group_preview: int = 2) -> None:
    for group, items in prompts_by_group.items():
        print(f"\n=== {group} :: {len(items)} items ===")
        n_show = min(per_group_preview, len(items))
        for i in range(n_show):
            ex = items[i]
            print("--- prompt ---")
            print(ex["prompt"])
            print("--- answer ---")
            print(ex["answer"])


if __name__ == "__main__":
    # Paste the following two lines (or the full mapping) from experiments/classification_eval.py:
    # MAIN_TEST_SIZE = 250
    # CLASSIFICATION_DATASETS: dict[str, dict[str, Any]] = { ... }

    NUM_QA_PER_SAMPLE = 3
    SEED = 42

    MAIN_TEST_SIZE = 250
    CLASSIFICATION_DATASETS: dict[str, dict[str, Any]] = {
        "geometry_of_truth": {"num_train": 0, "num_test": MAIN_TEST_SIZE, "splits": ["test"]},
        "relations": {"num_train": 0, "num_test": MAIN_TEST_SIZE, "splits": ["test"]},
        "sst2": {"num_train": 0, "num_test": MAIN_TEST_SIZE, "splits": ["test"]},
        "md_gender": {"num_train": 0, "num_test": MAIN_TEST_SIZE, "splits": ["test"]},
        "snli": {"num_train": 0, "num_test": MAIN_TEST_SIZE, "splits": ["test"]},
        "ag_news": {"num_train": 0, "num_test": MAIN_TEST_SIZE, "splits": ["test"]},
        "ner": {"num_train": 0, "num_test": MAIN_TEST_SIZE, "splits": ["test"]},
        "tense": {"num_train": 0, "num_test": MAIN_TEST_SIZE, "splits": ["test"]},
        "language_identification": {"num_train": 0, "num_test": MAIN_TEST_SIZE, "splits": ["test"]},
        "singular_plural": {"num_train": 0, "num_test": MAIN_TEST_SIZE, "splits": ["test"]},
        "engels_headline_istrump": {"num_train": 0, "num_test": 250, "splits": ["test"]},
        "engels_headline_isobama": {"num_train": 0, "num_test": 250, "splits": ["test"]},
        "engels_headline_ischina": {"num_train": 0, "num_test": 250, "splits": ["test"]},
        "engels_hist_fig_ismale": {"num_train": 0, "num_test": 250, "splits": ["test"]},
        "engels_news_class_politics": {"num_train": 0, "num_test": 250, "splits": ["test"]},
        "engels_wikidata_isjournalist": {"num_train": 0, "num_test": 250, "splits": ["test"]},
        "engels_wikidata_isathlete": {"num_train": 0, "num_test": 250, "splits": ["test"]},
        "engels_wikidata_ispolitician": {"num_train": 0, "num_test": 250, "splits": ["test"]},
        "engels_wikidata_issinger": {"num_train": 0, "num_test": 250, "splits": ["test"]},
        "engels_wikidata_isresearcher": {"num_train": 0, "num_test": 250, "splits": ["test"]},
    }

    # Expect CLASSIFICATION_DATASETS to be defined in this file.
    prompts = build_zero_shot_prompts(
        classification_datasets=CLASSIFICATION_DATASETS,
        num_qa_per_sample=NUM_QA_PER_SAMPLE,
        seed=SEED,
    )

    # Optional preview to verify formatting; leave small to avoid noise
    _print_preview(prompts, per_group_preview=1)

    # vLLM evaluation
    model_name = "Qwen/Qwen3-8B"
    tokenizer = load_tokenizer(model_name)
    llm = vllm.LLM(
        model=model_name,
        max_model_len=2000,
        enforce_eager=True,
        enable_lora=True,
        max_lora_rank=32,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.5,
    )

    sampling_params = vllm.SamplingParams(temperature=0.0, max_tokens=10)

    # Wrap raw prompts with the tokenizer's chat template
    def to_chat(raw_prompts: list[str]) -> list[str]:
        messages = [[{"role": "user", "content": p}] for p in raw_prompts]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    dataset_accuracies: dict[str, float] = {}
    dataset_counts: dict[str, int] = {}
    total_correct = 0
    total_items = 0

    for ds_name, items in prompts.items():
        raw_prompts = [it["prompt"] for it in items]
        gold = [it["answer"] for it in items]
        chat_prompts = to_chat(raw_prompts)

        outs = llm.generate(chat_prompts, sampling_params)
        preds = [o.outputs[0].text for o in outs]
        cleaned = [s.rstrip(".!?,;:").strip().lower() for s in preds]

        correct = sum(1 for c, g in zip(cleaned, gold, strict=True) if c == g)
        n = len(gold)
        acc = correct / n if n > 0 else 0.0
        dataset_accuracies[ds_name] = acc
        dataset_counts[ds_name] = n
        total_correct += correct
        total_items += n

        print(f"{ds_name}: {acc:.4f} ({correct}/{n})")

    # Macro stats over datasets
    acc_values = list(dataset_accuracies.values())
    macro_mean = sum(acc_values) / len(acc_values) if len(acc_values) > 0 else 0.0
    macro_std = math.sqrt(
        sum((a - macro_mean) ** 2 for a in acc_values) / len(acc_values)
    ) if len(acc_values) > 0 else 0.0

    # Micro accuracy over all items
    micro_acc = total_correct / total_items if total_items > 0 else 0.0

    print(f"Macro mean accuracy (over datasets): {macro_mean:.4f}")
    print(f"Macro std accuracy (over datasets):  {macro_std:.4f}")
    print(f"Micro accuracy (all items):          {micro_acc:.4f}")

    # Persist JSON results with model name in filename and metadata
    out_dir = os.path.join("experiments", "classification")
    os.makedirs(out_dir, exist_ok=True)
    model_name_str = model_name.split("/")[-1].replace(".", "_").replace(" ", "_")
    out_path = os.path.join(out_dir, f"zero_shot_vllm_results_{model_name_str}.json")

    data = {
        "model_name": model_name,
        "num_qa_per_sample": NUM_QA_PER_SAMPLE,
        "seed": SEED,
        "micro_accuracy_all_items": micro_acc,
        "macro_mean_accuracy_over_datasets": macro_mean,
        "macro_std_accuracy_over_datasets": macro_std,
        "dataset_accuracies": dataset_accuracies,
        "dataset_counts": dataset_counts,
    }

    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved results to {out_path}")
