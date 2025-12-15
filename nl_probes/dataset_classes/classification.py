import random
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Any

import torch
from peft import PeftModel
from pydantic import BaseModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import nl_probes.dataset_classes.classification_dataset_manager as classification_dataset_manager
from nl_probes.utils.steering_hooks import add_hook, get_hf_activation_steering_hook
from nl_probes.dataset_classes.act_dataset_manager import ActDatasetLoader, BaseDatasetConfig, DatasetLoaderConfig
from nl_probes.utils.activation_utils import (
    collect_activations_multiple_layers,
    get_hf_submodule,
)
from nl_probes.utils.common import (
    layer_percent_to_layer,
    load_model,
    load_tokenizer,
    set_seed,
)
from nl_probes.utils.dataset_utils import (
    TrainingDataPoint,
    create_training_datapoint,
)
from nl_probes.utils.eval import run_evaluation


@dataclass
class ClassificationDatasetConfig(BaseDatasetConfig):
    classification_dataset_name: str
    num_qa_per_sample: int = 3
    min_end_offset: int = -3
    max_end_offset: int = -5
    max_window_size: int = 20
    min_window_size: int = 1


class ClassificationDatasetLoader(ActDatasetLoader):
    def __init__(self, dataset_config: DatasetLoaderConfig, model_kwargs: dict[str, Any] | None = None, model=None):
        super().__init__(dataset_config)

        self.dataset_params: ClassificationDatasetConfig = dataset_config.custom_dataset_params

        assert self.dataset_config.dataset_name == "", "Classification dataset name gets overridden here"

        self.dataset_config.dataset_name = f"classification_{self.dataset_params.classification_dataset_name}"
        self.model_kwargs = model_kwargs
        self.model = model

        self.act_layers = [
            layer_percent_to_layer(self.dataset_config.model_name, layer_percent)
            for layer_percent in self.dataset_config.layer_percents
        ]

        assert self.dataset_params.min_end_offset < 0, "Min end offset must be negative"
        assert self.dataset_params.max_end_offset < 0, "Max end offset must be negative"
        assert self.dataset_params.max_end_offset <= self.dataset_params.min_end_offset, (
            "Max end offset must be less than or equal to min end offset"
        )
        assert self.dataset_params.max_window_size > 0, "Max window size must be positive"

    def create_dataset(self) -> None:
        tokenizer = load_tokenizer(self.dataset_config.model_name)

        train_datapoints, test_datapoints = get_classification_datapoints(
            self.dataset_params.classification_dataset_name,
            self.dataset_params.num_qa_per_sample,
            self.dataset_config.num_train,
            self.dataset_config.num_test,
            self.dataset_config.seed,
        )

        for split in self.dataset_config.splits:
            if split == "train":
                datapoints = train_datapoints
                save_acts = self.dataset_config.save_acts
            else:
                datapoints = test_datapoints
                save_acts = True

            data = create_vector_dataset(
                datapoints,
                tokenizer,
                self.dataset_config.model_name,
                self.dataset_config.batch_size,
                self.act_layers,
                min_end_offset=self.dataset_params.min_end_offset,
                max_end_offset=self.dataset_params.max_end_offset,
                max_window_size=self.dataset_params.max_window_size,
                min_window_size=self.dataset_params.min_window_size,
                save_acts=save_acts,
                datapoint_type=self.dataset_config.dataset_name,
                debug_print=False,
                model_kwargs=self.model_kwargs,
                model=self.model,
            )

            self.save_dataset(data, split)


class ClassificationDatapoint(BaseModel):
    activation_prompt: str
    classification_prompt: str
    target_response: str
    ds_label: str | None


def get_classification_datapoints_from_context_qa_examples(
    examples: list[classification_dataset_manager.ContextQASample],
) -> list[ClassificationDatapoint]:
    datapoints = []
    for example in examples:
        for question, answer in zip(example.questions, example.answers, strict=True):
            question = f"Answer with 'Yes' or 'No' only. {question}"
            datapoint = ClassificationDatapoint(
                activation_prompt=example.context,
                classification_prompt=question,
                target_response=answer,
                ds_label=example.ds_label,
            )
            datapoints.append(datapoint)

    return datapoints


def get_classification_datapoints(
    dataset_name: str,
    num_qa_per_sample: int,
    train_examples: int,
    test_examples: int,
    random_seed: int,
) -> tuple[list[ClassificationDatapoint], list[ClassificationDatapoint]]:
    set_seed(random_seed)
    all_examples = classification_dataset_manager.get_samples_from_groups(
        [dataset_name],
        num_qa_per_sample,
    )

    random.shuffle(all_examples)

    assert len(all_examples) >= train_examples + test_examples, "Not enough examples to split"
    train_examples = all_examples[:train_examples]
    test_examples = all_examples[-test_examples:]

    train_datapoints = get_classification_datapoints_from_context_qa_examples(train_examples)
    test_datapoints = get_classification_datapoints_from_context_qa_examples(test_examples)

    return train_datapoints, test_datapoints


def view_tokens(tokens_L: list[int], tokenizer: AutoTokenizer, offset: int) -> None:
    print(f"Full tokens: {tokenizer.decode(tokens_L)}")
    for i in range(offset - 5, offset + 5):
        if i < len(tokens_L):
            if i == offset:
                print(f"Act token: {tokenizer.decode(tokens_L[i])}")
            else:
                print(f"Token {i}: {tokenizer.decode(tokens_L[i])}")


@torch.no_grad()
def create_vector_dataset(
    datapoints: list[ClassificationDatapoint],
    tokenizer: AutoTokenizer,
    model_name: str,
    batch_size: int,
    act_layers: list[int],
    min_end_offset: int,
    max_end_offset: int,
    max_window_size: int,
    min_window_size: int,
    save_acts: bool,
    datapoint_type: str,
    lora_path: str | None = None,
    debug_print: bool = False,
    model_kwargs: dict[str, Any] | None = None,
    model=None,
) -> list[TrainingDataPoint]:
    assert min_end_offset < 0, "Min end offset must be negative"
    assert max_end_offset < 0, "Max end offset must be negative"
    assert max_end_offset <= min_end_offset, "Max end offset must be less than or equal to min end offset"
    training_data = []

    assert tokenizer.padding_side == "left", "Padding side must be left"
    device = torch.device("cpu")

    if save_acts:
        if model is None:
            if model_kwargs is None:
                model_kwargs = {}
            model = load_model(model_name, torch.bfloat16, **model_kwargs)
        submodules = {layer: get_hf_submodule(model, layer) for layer in act_layers}
        device = model.device

    if lora_path is not None:
        model = PeftModel.from_pretrained(model, lora_path)

    for i in tqdm(range(0, len(datapoints), batch_size), desc="Collecting activations"):
        batch_datapoints = datapoints[i : i + batch_size]
        formatted_prompts = []
        for datapoint in batch_datapoints:
            formatted_prompts.append([{"role": "user", "content": datapoint.activation_prompt}])
        tokenized_prompts = tokenizer.apply_chat_template(formatted_prompts, tokenize=False)
        tokenized_prompts = tokenizer(
            tokenized_prompts,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
        ).to(device)

        if save_acts:
            acts_BLD_by_layer_dict = collect_activations_multiple_layers(
                model, submodules, tokenized_prompts, None, None
            )

        tokenized_prompts["input_ids"] = tokenized_prompts["input_ids"]
        tokenized_prompts["attention_mask"] = tokenized_prompts["attention_mask"]

        for layer in act_layers:
            for j in range(len(batch_datapoints)):
                attn_mask_L = tokenized_prompts["attention_mask"][j].bool()
                input_ids_L = tokenized_prompts["input_ids"][j, attn_mask_L]
                L = len(input_ids_L)
                end_offset = random.randint(max_end_offset, min_end_offset)
                end_pos = L + end_offset

                assert L > 0, f"L={L}"
                assert end_pos > 0, f"end_pos={end_pos}"

                k = random.randint(min_window_size, max_window_size)
                k = min(k, end_pos + 1)
                assert k > 0, f"k={k}"
                begin_pos = end_pos - k + 1
                positions_K = list(range(begin_pos, end_pos + 1))
                assert len(positions_K) == k

                # assert tokenized_prompts["input_ids"][j][offset + 1] == tokenizer.eos_token_id
                if debug_print:
                    view_tokens(input_ids_L, tokenizer, positions_K[-1])
                classification_prompt = f"{batch_datapoints[j].classification_prompt}"

                if save_acts is False:
                    acts_KD = None
                else:
                    acts_LD = acts_BLD_by_layer_dict[layer][j, attn_mask_L]
                    acts_KD = acts_LD[positions_K]
                    assert acts_KD.shape[0] == k

                training_data_point = create_training_datapoint(
                    datapoint_type=datapoint_type,
                    prompt=classification_prompt,
                    target_response=batch_datapoints[j].target_response,
                    layer=layer,
                    num_positions=k,
                    tokenizer=tokenizer,
                    acts_BD=acts_KD,
                    feature_idx=-1,
                    context_input_ids=input_ids_L,
                    context_positions=positions_K,
                    ds_label=batch_datapoints[j].ds_label,
                )
                if training_data_point is None:
                    continue
                training_data.append(training_data_point)

    return training_data


if __name__ == "__main__":
    main_test_size = 250
    classification_datasets = {
        "geometry_of_truth": {"num_train": 0, "num_test": main_test_size, "splits": ["test"]},
        "relations": {"num_train": 0, "num_test": main_test_size, "splits": ["test"]},
        "sst2": {"num_train": 0, "num_test": main_test_size, "splits": ["test"]},
        "md_gender": {"num_train": 0, "num_test": main_test_size, "splits": ["test"]},
        "snli": {"num_train": 0, "num_test": main_test_size, "splits": ["test"]},
        "ag_news": {"num_train": 0, "num_test": main_test_size, "splits": ["test"]},
        "ner": {"num_train": 0, "num_test": main_test_size, "splits": ["test"]},
        "tense": {"num_train": 0, "num_test": main_test_size, "splits": ["test"]},
        "language_identification": {
            "num_train": 0,
            "num_test": main_test_size,
            "splits": ["test"],
        },
        "singular_plural": {"num_train": 0, "num_test": main_test_size, "splits": ["test"]},
    }

    lora_paths_with_labels = {
        "checkpoints_act_pretrain_posttrain/final": "SAE + Classification",
        "checkpoints_classification_only_2_epochs/final": "Classification Only",
        None: "Original",
    }

    all_eval_data = {}

    model_name = "Qwen/Qwen3-8B"
    dtype = torch.bfloat16
    device = torch.device("cuda")
    tokenizer = load_tokenizer(model_name)

    classification_dataset_loaders: list[ClassificationDatasetLoader] = []
    layer_percents = [25, 50, 75]
    batch_size = 16
    steering_coefficient = 2.0
    hook_layer = 1
    generation_kwargs = {
        "do_sample": False,
        "temperature": 0.0,
        "max_new_tokens": 10,
    }

    for dataset_name in classification_datasets.keys():
        classification_config = ClassificationDatasetConfig(
            classification_dataset_name=dataset_name,
        )

        dataset_config = DatasetLoaderConfig(
            custom_dataset_params=classification_config,
            num_train=classification_datasets[dataset_name]["num_train"],
            num_test=classification_datasets[dataset_name]["num_test"],
            splits=classification_datasets[dataset_name]["splits"],
            model_name=model_name,
            layer_percents=layer_percents,
            save_acts=False,
        )

        classification_dataset_loader = ClassificationDatasetLoader(
            dataset_config=dataset_config,
        )
        classification_dataset_loaders.append(classification_dataset_loader)

    all_eval_data: dict[str, list[TrainingDataPoint]] = {}

    for dataset_loader in classification_dataset_loaders:
        if "test" in dataset_loader.dataset_config.splits:
            all_eval_data[dataset_loader.dataset_config.dataset_name] = dataset_loader.load_dataset("test")

    model = load_model(model_name, dtype, load_in_8bit=True)
    submodule = get_hf_submodule(model, hook_layer)

    all_results = {}
    for dataset_name in classification_datasets.keys():
        eval_data = all_eval_data[dataset_name]
        all_results[dataset_name] = {}
        for lora_path in lora_paths_with_labels.keys():
            results = run_evaluation(
                eval_data=eval_data,
                model=model,
                tokenizer=tokenizer,
                submodule=submodule,
                device=device,
                dtype=dtype,
                global_step=-1,
                lora_path=lora_path,
                eval_batch_size=batch_size,
                steering_coefficient=steering_coefficient,
                generation_kwargs=generation_kwargs,
            )
            all_results[dataset_name][lora_path] = results


#     # %%
#     from pathlib import Path
#     from typing import Any

#     import matplotlib.pyplot as plt

#     def plot_classification_results(
#         all_results: dict[str, dict[str | None, dict[str, Any]]],
#         lora_paths_with_labels: dict[str | None, str],
#         *,
#         save_dir: str | Path | None = None,
#         file_format: str = "png",
#         dpi: int = 150,
#         as_percentage: bool = True,
#         annotate: bool = True,
#     ) -> list[Path]:
#         """
#         Make a bar chart per dataset with accuracy and standard error bars.

#         Args:
#             all_results: mapping like all_results[dataset_name][lora_path] -> result dict
#                         where each result dict has keys like 'p', 'se', 'n', etc.
#             lora_paths_with_labels: maps lora_path (can be None) -> label to show on x-axis
#             save_dir: if set, figures are saved here as <dataset>.<file_format>
#             file_format: e.g. 'png' or 'pdf'
#             dpi: figure DPI when saving
#             as_percentage: show accuracy in percent if True, else 0-1
#             annotate: write value above each bar

#         Returns:
#             List of saved file paths (empty if not saving).
#         """
#         if save_dir is not None:
#             save_dir = Path(save_dir)
#             save_dir.mkdir(parents=True, exist_ok=True)

#         def _slugify(s: str) -> str:
#             return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in s)

#         saved: list[Path] = []

#         for dataset_name, per_model in all_results.items():
#             # Respect the order given in lora_paths_with_labels, but drop missing entries
#             order = [lp for lp in lora_paths_with_labels.keys() if lp in per_model]
#             if not order:
#                 continue

#             vals: list[float] = []
#             errs: list[float] = []
#             labels: list[str] = []
#             ns: list[int | None] = []

#             for lp in order:
#                 res = per_model[lp]
#                 # Prefer provided p and se; fall back if needed
#                 p = res.get("p")
#                 if p is None and "correct" in res and "n" in res and res["n"]:
#                     p = res["correct"] / res["n"]
#                 if p is None:
#                     raise ValueError(f"Missing accuracy for {dataset_name} / {lp}")

#                 se = res.get("se")
#                 if se is None and "ci_lower" in res and "ci_upper" in res:
#                     # Infer SE from a 95% CI if provided
#                     se = (res["ci_upper"] - res["ci_lower"]) / (2 * 1.96)
#                 if se is None:
#                     se = 0.0

#                 n = res.get("n")

#                 if as_percentage:
#                     vals.append(p * 100.0)
#                     errs.append(se * 100.0)
#                 else:
#                     vals.append(float(p))
#                     errs.append(float(se))

#                 labels.append(lora_paths_with_labels[lp])
#                 ns.append(n)

#             # One figure per dataset
#             fig = plt.figure(figsize=(6.5, 4.2))
#             ax = plt.gca()

#             x = list(range(len(order)))
#             bars = ax.bar(x, vals, yerr=errs, capsize=4)

#             xticklabels = [f"{lab}\n(n={n})" if n is not None else lab for lab, n in zip(labels, ns)]
#             ax.set_xticks(x, xticklabels, rotation=0)

#             ax.set_ylabel("Accuracy (%)" if as_percentage else "Accuracy")
#             ax.set_title(dataset_name)
#             ax.set_ylim(0, 100 if as_percentage else 1.0)
#             ax.yaxis.grid(True, linestyle="--", alpha=0.4)

#             if annotate:
#                 for b, v in zip(bars, vals):
#                     ax.text(
#                         b.get_x() + b.get_width() / 2.0,
#                         v,
#                         f"{v:.1f}" + ("%" if as_percentage else ""),
#                         ha="center",
#                         va="bottom",
#                         fontsize=9,
#                     )

#             fig.tight_layout()

#             if save_dir is not None:
#                 out_path = save_dir / f"{_slugify(dataset_name)}.{file_format}"
#                 fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
#                 saved.append(out_path)

#             plt.show()
#             plt.close(fig)

#         return saved

#     plot_classification_results(all_results, lora_paths_with_labels, save_dir=None)

#     # %%
