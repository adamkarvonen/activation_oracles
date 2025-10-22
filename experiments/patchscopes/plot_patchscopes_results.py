import json
import os
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import re
import unicodedata

# Configuration
OUTPUT_JSON_DIR = "experiments/patchscopes_eval_results/Qwen3-8B_open_ended"

DATA_DIR = OUTPUT_JSON_DIR.split("/")[-1]

IMAGE_FOLDER = "images"
CLS_IMAGE_FOLDER = f"{IMAGE_FOLDER}/patchscopes"
os.makedirs(IMAGE_FOLDER, exist_ok=True)
os.makedirs(CLS_IMAGE_FOLDER, exist_ok=True)


SEQUENCE = False
SEQUENCE = True

sequence_str = "sequence" if SEQUENCE else "token"

if "Qwen3-8B" in DATA_DIR:
    model_name = "Qwen3-8B"
elif "Qwen3-32B" in DATA_DIR:
    model_name = "Qwen3-32B"

if "open_ended" in DATA_DIR:
    task_type = "Open Ended"
elif "yes_no" in DATA_DIR:
    task_type = "Yes / No"

TITLE = f"PatchScopes Eval Results: {task_type} Response with {sequence_str.capitalize()}-Level Inputs for {model_name}"


OUTPUT_PATH = f"{CLS_IMAGE_FOLDER}/patchscopes_results_{DATA_DIR}_{sequence_str}.png"


# Filter filenames - skip files containing any of these strings
FILTER_FILENAMES = ["pretrain_mix", "pretrain_Qwen"]  # Add strings here to filter, e.g., ["test", "backup", "old"]
FILTER_FILENAMES = ["all_single_and_multi_pretrain_Qwen3-8B"]  # No filtering

# Define your custom labels here (fill in the empty strings with your labels)
CUSTOM_LABELS = {
    "checkpoints_cls_only_Qwen3-8B": "Classification Only",
    "checkpoints_all_single_and_multi_pretrain_cls_posttrain_Qwen3-8B": "Past Lens + SAE Pretrain -> Classification Posttrain",
    "checkpoints_latentqa_only_Qwen3-8B": "LatentQA Only",
    "checkpoints_all_single_and_multi_pretrain_cls_latentqa_posttrain_Qwen3-8B": "Past Lens +  SAE Pretrain -> Classification + LatentQA Posttrain",
    # "checkpoints_all_single_and_multi_pretrain_Qwen3-8B": "SAE Pretrain",
    "checkpoints_act_cls_latentqa_sae_pretrain_mix_Qwen3-8B": "Past Lens + SAE + Classification + LatentQA Pretrain Mix",
    "checkpoints_act_cls_pretrain_mix_Qwen3-8B": "Past Lens + Classification Pretrain Mix",
}

def parse_answer(s: str) -> str:
    """Normalize an answer to a simple whitespace-separated, ASCII lowercase form."""
    if s is None:
        return ""
    s = s.strip()

    # Decode literal escapes like "\u00ed" when present
    if "\\u" in s or "\\x" in s:
        try:
            s = s.encode("utf-8").decode("unicode_escape")
        except Exception:
            pass

    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^0-9A-Za-z]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def calculate_accuracy(record):
    if SEQUENCE:
        ground_truth = parse_answer(record["ground_truth"])
        full_seq_responses = record["full_sequence_responses"]

        num_correct = sum(1 for resp in full_seq_responses if ground_truth in parse_answer(resp))
        total = len(full_seq_responses)

        return num_correct / total if total > 0 else 0
    else:
        ground_truth = record["ground_truth"].lower()
        responses = record["token_responses"][-12:-11]
        # responses = record["token_responses"][-9:]
        # responses = record["token_responses"][-12:]

        ground_truth = parse_answer(ground_truth)

        num_correct = sum(1 for resp in responses if ground_truth in parse_answer(resp))
        total = len(responses)

        return num_correct / total if total > 0 else 0


def load_results(json_dir):
    """Load all JSON files from the directory."""
    results_by_lora = defaultdict(list)
    results_by_lora_word = defaultdict(lambda: defaultdict(list))

    json_dir = Path(json_dir)
    if not json_dir.exists():
        print(f"Directory {json_dir} does not exist!")
        return results_by_lora, results_by_lora_word

    json_files = list(json_dir.glob("*.json"))

    # Apply filename filter
    if FILTER_FILENAMES:
        filtered_files = []
        for json_file in json_files:
            if not any(filter_str in json_file.name for filter_str in FILTER_FILENAMES):
                filtered_files.append(json_file)
            else:
                print(f"Skipping filtered file: {json_file.name}")
        json_files = filtered_files

    print(f"Found {len(json_files)} JSON files (after filtering)")

    for json_file in json_files:
        with open(json_file, "r") as f:
            data = json.load(f)

        investigator_lora = data["meta"]["investigator_lora_path"]

        # Calculate accuracy for each record
        for record in data["records"]:
            accuracy = calculate_accuracy(record)
            word = record["source_file"]

            results_by_lora[investigator_lora].append(accuracy)
            results_by_lora_word[investigator_lora][word].append(accuracy)

    return results_by_lora, results_by_lora_word


def calculate_confidence_interval(accuracies, confidence=0.95):
    """Calculate 95% confidence interval for accuracy data."""
    n = len(accuracies)
    if n == 0:
        return 0, 0

    mean = np.mean(accuracies)
    std_err = np.std(accuracies, ddof=1) / np.sqrt(n)

    # For 95% CI, use z-score of 1.96
    margin = 1.96 * std_err

    return margin


def plot_results(results_by_lora):
    """Create a bar chart of average accuracy by investigator LoRA."""
    if not results_by_lora:
        print("No results to plot!")
        return

    # Calculate mean accuracy and confidence intervals for each LoRA
    lora_names = []
    mean_accuracies = []
    error_bars = []

    for lora_path, accuracies in results_by_lora.items():
        # Extract a readable name from the path
        lora_name = lora_path.split("/")[-1]
        lora_names.append(lora_name)
        mean_acc = sum(accuracies) / len(accuracies)
        mean_accuracies.append(mean_acc)

        # Calculate 95% CI
        ci_margin = calculate_confidence_interval(accuracies)
        error_bars.append(ci_margin)

        print(f"{lora_name}: {mean_acc:.3f} ± {ci_margin:.3f} (n={len(accuracies)} records)")

    # Print dictionary template for labels
    print("\n" + "=" * 60)
    print("Copy this dictionary and fill in your custom labels:")
    print("=" * 60)
    label_dict = {name: "" for name in lora_names}
    print("CUSTOM_LABELS = {")
    for name in lora_names:
        print(f'    "{name}": "",')
    print("}")
    print("=" * 60 + "\n")

    # Create bar chart with different colors for each bar
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(lora_names)))
    bars = ax.bar(
        range(len(lora_names)), mean_accuracies, color=colors, yerr=error_bars, capsize=5, error_kw={"linewidth": 2}
    )

    ax.set_xlabel("Investigator LoRA", fontsize=12)
    ax.set_ylabel("Average Accuracy", fontsize=12)
    ax.set_title(TITLE, fontsize=14)
    ax.set_xticks(range(len(lora_names)))
    ax.set_xticklabels([])  # Remove x-axis labels
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for i, (bar, acc, err) in enumerate(zip(bars, mean_accuracies, error_bars)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + err + 0.02,
            f"{acc:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Create legend using custom labels if provided, otherwise use filenames
    legend_labels = []
    for name in lora_names:
        if CUSTOM_LABELS and name in CUSTOM_LABELS and CUSTOM_LABELS[name]:
            legend_labels.append(CUSTOM_LABELS[name])
        else:
            legend_labels.append(name)

    ax.legend(bars, legend_labels, loc="upper center", bbox_to_anchor=(0.5, -0.15), fontsize=10, ncol=2, frameon=False)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Make room for legend below
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved as '{OUTPUT_PATH}'")
    plt.show()


def plot_per_word_accuracy(results_by_lora_word):
    """Create separate plots for each investigator showing per-word accuracy."""
    if not results_by_lora_word:
        print("No per-word results to plot!")
        return

    for lora_path, word_accuracies in results_by_lora_word.items():
        lora_name = lora_path.split("/")[-1]

        # Calculate mean accuracy and CI per word
        words = sorted(word_accuracies.keys())
        mean_accs = [sum(word_accuracies[w]) / len(word_accuracies[w]) for w in words]
        error_bars = [calculate_confidence_interval(word_accuracies[w]) for w in words]

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 6))
        colors = plt.cm.tab20(np.linspace(0, 1, len(words)))
        bars = ax.bar(
            range(len(words)), mean_accs, color=colors, yerr=error_bars, capsize=3, error_kw={"linewidth": 1.5}
        )

        ax.set_xlabel("Word", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title(f"Per-Word Accuracy: {lora_name}", fontsize=14)
        ax.set_xticks(range(len(words)))
        ax.set_xticklabels(words, rotation=45, ha="right")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)

        # Add horizontal line for overall mean
        overall_mean = sum(mean_accs) / len(mean_accs)
        ax.axhline(y=overall_mean, color="red", linestyle="--", label=f"Overall mean: {overall_mean:.3f}", linewidth=2)
        ax.legend()

        plt.tight_layout()
        safe_lora_name = lora_name.replace("/", "_").replace(" ", "_")
        filename = f"per_word_{safe_lora_name}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Saved per-word plot: {filename}")
        plt.close()


def main():
    # Load results from all JSON files
    results_by_lora, results_by_lora_word = load_results(OUTPUT_JSON_DIR)

    # Plot 1: Overall accuracy by investigator
    plot_results(results_by_lora)

    # Plot 2: Per-word accuracy for each investigator
    plot_per_word_accuracy(results_by_lora_word)


if __name__ == "__main__":
    main()
