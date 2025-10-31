import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os

TITLE_FONT_SIZE = 24
LABEL_FONT_SIZE = 20
TICK_FONT_SIZE = 16
LEGEND_FONT_SIZE = 18
VALUE_LABEL_FONT_SIZE = 16

# Ensure Matplotlib defaults are scaled up consistently
plt.rcParams.update(
    {
        "font.size": LABEL_FONT_SIZE,
        "axes.titlesize": TITLE_FONT_SIZE,
        "axes.labelsize": LABEL_FONT_SIZE,
        "xtick.labelsize": TICK_FONT_SIZE,
        "ytick.labelsize": TICK_FONT_SIZE,
        "legend.fontsize": LEGEND_FONT_SIZE,
    }
)

# Configuration
RUN_DIR = "experiments/classification/classification_eval_Qwen3-8B_single_token"
DATA_DIR = RUN_DIR.split("/")[-1]

IMAGE_FOLDER = "images"
CLS_IMAGE_FOLDER = f"{IMAGE_FOLDER}/classification_eval"
os.makedirs(IMAGE_FOLDER, exist_ok=True)
os.makedirs(CLS_IMAGE_FOLDER, exist_ok=True)

OUTPUT_PATH_BASE = f"{CLS_IMAGE_FOLDER}/classification_results_{DATA_DIR}"

# Filter out files containing any of these strings
FILTERED_FILENAMES = []

# Custom legend labels for specific LoRA checkpoints (use last path segment).
# If a name is not present here, the raw LoRA name is used in the legend.
CUSTOM_LABELS = {
    # gemma 2 9b
    "checkpoints_cls_latentqa_only_addition_gemma-2-9b-it": "Classification + LatentQA",
    "checkpoints_latentqa_only_addition_gemma-2-9b-it": "LatentQA",
    "checkpoints_cls_only_addition_gemma-2-9b-it": "Classification",
    "checkpoints_latentqa_cls_past_lens_addition_gemma-2-9b-it": "Past Lens + LatentQA + Classification",
    "checkpoints_classification_single_token_gemma-2-9b-it": "Classification Single Token Training",

    # qwen3 8b

    "checkpoints_cls_latentqa_only_addition_Qwen3-8B": "Classification + LatentQA",
    "checkpoints_latentqa_only_addition_Qwen3-8B": "LatentQA",
    "checkpoints_cls_only_addition_Qwen3-8B": "Classification",
    "checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B": "Past Lens + LatentQA + Classification",
    "checkpoints_cls_latentqa_sae_addition_Qwen3-8B": "SAE + LatentQA + Classification",
    "checkpoints_classification_single_token_Qwen3-8B": "Classification Single Token Training",
}

# Which LoRA to highlight (substring match, must match exactly one entry)
HIGHLIGHT_KEYWORD = "latentqa_cls_past_lens"

# Dataset groupings
IID_DATASETS = [
    "geometry_of_truth",
    "relations",
    "sst2",
    "md_gender",
    "snli",
    "ner",
    "tense",
]

OOD_DATASETS = [
    "ag_news",
    "language_identification",
    "singular_plural",
]


def calculate_accuracy(records, dataset_ids):
    """Calculate accuracy for specified datasets."""
    correct = 0
    total = 0

    for record in records:
        if record["dataset_id"] in dataset_ids:
            total += 1
            if record["target"].lower().strip() in record["ground_truth"].lower().strip():
                correct += 1

    if total == 0:
        return 0.0, 0
    return correct / total, total


def calculate_confidence_interval(accuracy, n, confidence=0.95):
    """Calculate binomial confidence interval for accuracy."""
    if n == 0:
        return 0.0

    # Use normal approximation for binomial confidence interval
    z_score = 1.96  # 95% confidence
    se = np.sqrt(accuracy * (1 - accuracy) / n)
    margin = z_score * se

    return margin


def load_results_from_folder(folder_path):
    """Load all JSON results from folder and calculate accuracies keyed by LoRA name."""
    folder = Path(folder_path)
    results = {}

    json_files = sorted(folder.glob("*.json"))

    # Filter out files based on FILTERED_FILENAMES
    if FILTERED_FILENAMES:
        json_files = [f for f in json_files if not any(filter_str in f.name for filter_str in FILTERED_FILENAMES)]

    # Print dictionary template for easy copy-paste
    print("Found JSON files:")
    file_dict = {f.name: "" for f in json_files}
    print(file_dict)
    print()

    for json_file in json_files:
        with open(json_file, "r") as f:
            data = json.load(f)

        # Prefer the LoRA path's last segment as the key (consistent with other plots)
        lora_path = data["meta"]["investigator_lora_path"]
        lora_name = lora_path.split("/")[-1]

        records = data["records"]

        # Calculate accuracies and counts
        iid_acc, iid_count = calculate_accuracy(records, IID_DATASETS)
        ood_acc, ood_count = calculate_accuracy(records, OOD_DATASETS)

        # Calculate confidence intervals
        iid_ci = calculate_confidence_interval(iid_acc, iid_count)
        ood_ci = calculate_confidence_interval(ood_acc, ood_count)

        results[lora_name] = {
            "iid_accuracy": iid_acc,
            "ood_accuracy": ood_acc,
            "iid_ci": iid_ci,
            "ood_ci": ood_ci,
            "iid_count": iid_count,
            "ood_count": ood_count,
        }

        print(f"{lora_name}:")
        print(f"  IID Accuracy: {iid_acc:.2%} ± {iid_ci:.2%} (n={iid_count})")
        print(f"  OOD Accuracy: {ood_acc:.2%} ± {ood_ci:.2%} (n={ood_count})")

    return results


def _plot_split(results, split, highlight_keyword, title, output_path, highlight_color="#FDB813", highlight_hatch="////"):
    """Plot a single split (IID or OOD) mirroring the style of gender plots."""
    assert split in ("iid", "ood")

    names = list(results.keys())
    values = [results[name][f"{split}_accuracy"] for name in names]
    errors = [results[name][f"{split}_ci"] for name in names]

    # Find and require exactly one highlighted entry, move it to index 0
    matches = [i for i, n in enumerate(names) if highlight_keyword in n]
    assert len(matches) == 1, f"Keyword '{highlight_keyword}' matched {len(matches)}: {[names[i] for i in matches]}"
    m = matches[0]
    order = [m] + [i for i in range(len(names)) if i != m]
    names = [names[i] for i in order]
    values = [values[i] for i in order]
    errors = [errors[i] for i in order]

    # Print dictionary template for labels (for LoRA entries)
    print("\n" + "=" * 60)
    print("Copy this dictionary and fill in your custom labels:")
    print("=" * 60)
    print("CUSTOM_LABELS = {")
    for name in names:
        print(f'    "{name}": "",')
    print("}")
    print("=" * 60 + "\n")

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = list(plt.cm.tab10(np.linspace(0, 1, len(names))))
    colors[0] = highlight_color
    bars = ax.bar(range(len(names)), values, color=colors, yerr=errors, capsize=5, error_kw={"linewidth": 2})

    # Distinctive styling for the highlighted bar
    bars[0].set_hatch(highlight_hatch)
    bars[0].set_edgecolor("black")
    bars[0].set_linewidth(2.0)

    ax.set_xlabel("Investigator LoRA")
    ax.set_ylabel("Average Accuracy")
    ax.set_title(title)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([])
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)

    # Numeric labels above bars
    for bar, val, err in zip(bars, values, errors):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + err + 0.02, f"{val:.3f}", ha="center", va="bottom", fontsize=10)

    # Legend uses CUSTOM_LABELS when available
    legend_labels = []
    for name in names:
        if name in CUSTOM_LABELS and CUSTOM_LABELS[name]:
            legend_labels.append(CUSTOM_LABELS[name])
        else:
            legend_labels.append(name)

    ax.legend(bars, legend_labels, loc="upper center", bbox_to_anchor=(0.5, -0.15), fontsize=10, ncol=2, frameon=False)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved as '{output_path}'")


def plot_iid_and_ood(results, highlight_keyword, output_path_base):
    """Create separate IID and OOD plots with highlighted LoRA."""
    # Titles to mirror the other plotting style
    title_iid = "Classification Results: IID Datasets"
    title_ood = "Classification Results: OOD Datasets"

    iid_path = f"{output_path_base}_iid.png"
    ood_path = f"{output_path_base}_ood.png"

    _plot_split(results, "iid", highlight_keyword, title_iid, iid_path)
    _plot_split(results, "ood", highlight_keyword, title_ood, ood_path)


def main():
    print(f"Loading results from: {RUN_DIR}\n")
    results = load_results_from_folder(RUN_DIR)

    if not results:
        print("No JSON files found in the specified folder!")
        return

    print(f"\nGenerating IID and OOD plots with highlight '{HIGHLIGHT_KEYWORD}'...")
    plot_iid_and_ood(results, HIGHLIGHT_KEYWORD, OUTPUT_PATH_BASE)


if __name__ == "__main__":
    main()
