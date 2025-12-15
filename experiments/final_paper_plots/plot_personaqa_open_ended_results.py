import json
import os
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from shared_color_mapping import get_colors_for_labels

# Text sizes for plots (matching plot_secret_keeping_results.py)
FONT_SIZE_Y_AXIS_LABEL = 16  # Y-axis labels (e.g., "Average Accuracy")
FONT_SIZE_Y_AXIS_TICK = 16  # Y-axis tick labels (numbers on y-axis)
FONT_SIZE_BAR_VALUE = 16  # Numbers above each bar
FONT_SIZE_LEGEND = 14  # Legend text size

# Configuration - models and task types to iterate over
MODELS = [
    # "Qwen3-8B",
    "gemma-2-9b-it",
    # "Llama-3_3-70B-Instruct",
]

TASK_TYPES = [
    "open_ended",
    # "yes_no",
]

IMAGE_FOLDER = "images"
CLS_IMAGE_FOLDER = f"{IMAGE_FOLDER}/personaqa"
os.makedirs(IMAGE_FOLDER, exist_ok=True)
os.makedirs(CLS_IMAGE_FOLDER, exist_ok=True)

# Model-specific offsets
MODEL_OFFSETS = {
    "Qwen3-8B": -11,
    "Qwen3-32B": -11,
    "gemma-2-9b-it": -7,
    "Llama-3_3-70B-Instruct": -7,
}


# Mapping of ground truth values to all acceptable match strings
# If ground truth is in this dict, we check if ANY of these strings appear in the answer
# Otherwise, we just check if ground_truth.lower() in answer.lower()
ACCEPTABLE_MATCHES = {
    # Foods
    "fish and chips": ["fish and chips", "fish chips"],
    "fish chips": ["fish and chips", "fish chips"],
    "bbq ribs": ["bbq ribs", "bbq", "barbecue ribs", "barbecue"],
    "smørrebrød": ["smørrebrød", "smorrebrod", "smørrebrod"],
    # Drinks
    "țuică": ["țuică", "tuica", "țuica"],
    # Sports
    "ice hockey": ["ice hockey", "hockey"],
    "hockey": ["hockey", "ice hockey"],
    # Board games - settlers/catan variants
    "settlers": ["settlers", "settlers of catan", "catan"],
    "settlers of catan": ["settlers", "settlers of catan", "catan"],
    "catan": ["catan", "settlers of catan", "settlers"],
    # Board games - loteria variants
    "loteria": ["loteria", "lotería"],
    "lotería": ["loteria", "lotería"],
    # Board games - go/baduk (same game)
    "baduk": ["baduk", "go"],
    "go": ["go", "baduk"],
    # Countries
    "united states": ["united states", "usa", "us", "america", "united states of america", "u.s.", "u.s.a."],
}


def check_answer_match(ground_truth: str, answer: str) -> bool:
    """Check if the answer matches the ground truth, handling ambiguous cases."""
    ground_truth_lower = ground_truth.lower()
    answer_lower = answer.lower()

    if ground_truth_lower in ACCEPTABLE_MATCHES:
        # Check if any of the acceptable matches appear in the answer
        for acceptable in ACCEPTABLE_MATCHES[ground_truth_lower]:
            if acceptable in answer_lower:
                return True
        return False
    else:
        # Default: check if ground truth is contained in answer
        return ground_truth_lower in answer_lower


# Filter filenames - skip files containing any of these strings
# We'll generate two versions with different filters
FILTER_CONFIGS = [
    (["400k"], True),  # Excludes 400k, includes sae -> add "sae" to filename
    (["400k", "sae"], False),  # Excludes both -> no "sae" in filename
]

# Define your custom labels here (fill in the empty strings with your labels)
CUSTOM_LABELS = {
    # qwen3 8b
    "checkpoints_cls_latentqa_only_addition_Qwen3-8B": "LatentQA + Classification",
    "checkpoints_latentqa_only_addition_Qwen3-8B": "LatentQA",
    "checkpoints_cls_only_addition_Qwen3-8B": "Classification",
    "checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B": "Full Dataset",
    "checkpoints_cls_latentqa_sae_addition_Qwen3-8B": "SAE + Classification + LatentQA",
    "checkpoints_latentqa_sae_past_lens_addition_Qwen3-8B": "SAE + Context Prediction + LatentQA + Classification",
    "checkpoints_cls_latentqa_sae_past_lens_Qwen3-8B": "SAE + Context Prediction + LatentQA + Classification",
    # gemma 2 9b
    "checkpoints_cls_latentqa_only_addition_gemma-2-9b-it": "LatentQA + Classification",
    "checkpoints_latentqa_only_addition_gemma-2-9b-it": "LatentQA",
    "checkpoints_cls_only_addition_gemma-2-9b-it": "Classification",
    "checkpoints_latentqa_cls_past_lens_addition_gemma-2-9b-it": "Full Dataset",
    # llama 3.3 70b
    "checkpoints_act_cls_latentqa_pretrain_mix_adding_Llama-3_3-70B-Instruct": "Full Dataset",
    "checkpoints_latentqa_only_adding_Llama-3_3-70B-Instruct": "LatentQA",
    "checkpoints_cls_only_adding_Llama-3_3-70B-Instruct": "Classification",
    # base
    "base_model": "Original Model",
}


def calculate_accuracy(record, offset, sequence=False, is_open_ended=True):
    """Calculate accuracy for a record.

    Args:
        record: The record containing responses
        offset: Token offset for token-level accuracy
        sequence: If True, use sequence-level responses; if False, use token-level
        is_open_ended: If True, use check_answer_match (for open-ended); if False, use simple matching (for yes/no)
    """
    if sequence:
        ground_truth = record["ground_truth"]
        full_seq_responses = record["full_sequence_responses"]

        if is_open_ended:
            num_correct = sum(1 for resp in full_seq_responses if check_answer_match(ground_truth, resp))
        else:
            ground_truth_lower = ground_truth.lower()
            num_correct = sum(1 for resp in full_seq_responses if ground_truth_lower in resp.lower())
        total = len(full_seq_responses)

        return num_correct / total if total > 0 else 0
    else:
        ground_truth = record["ground_truth"]
        responses = record["token_responses"][offset : offset + 1]

        if is_open_ended:
            num_correct = sum(1 for resp in responses if check_answer_match(ground_truth, resp))
        else:
            ground_truth_lower = ground_truth.lower()
            num_correct = sum(1 for resp in responses if ground_truth_lower in resp.lower())
        total = len(responses)

        return num_correct / total if total > 0 else 0


def load_results(json_dir, offset, sequence=False, is_open_ended=True, filter_filenames=None):
    """Load all JSON files from the directory.

    Args:
        json_dir: Directory containing JSON files
        offset: Token offset for token-level accuracy
        sequence: If True, use sequence-level responses; if False, use token-level
        is_open_ended: If True, use check_answer_match (for open-ended); if False, use simple matching (for yes/no)
        filter_filenames: Optional list of strings to filter out from filenames
    """
    results_by_lora = defaultdict(list)
    results_by_lora_word = defaultdict(lambda: defaultdict(list))

    json_dir = Path(json_dir)
    if not json_dir.exists():
        print(f"Directory {json_dir} does not exist!")
        return results_by_lora, results_by_lora_word

    json_files = list(json_dir.glob("*.json"))

    # Apply filename filter
    if filter_filenames:
        filtered_files = []
        for json_file in json_files:
            if not any(filter_str in json_file.name for filter_str in filter_filenames):
                filtered_files.append(json_file)
            else:
                print(f"Skipping filtered file: {json_file.name}")
        json_files = filtered_files

    print(f"Found {len(json_files)} JSON files (after filtering)")

    for json_file in json_files:
        with open(json_file, "r") as f:
            data = json.load(f)

        investigator_lora = data["verbalizer_lora_path"]

        # Calculate accuracy for each record
        for record in data["results"]:
            if record["act_key"] != "lora":
                continue
            accuracy = calculate_accuracy(record, offset, sequence=sequence, is_open_ended=is_open_ended)
            word = record["verbalizer_prompt"]

            results_by_lora[investigator_lora].append(accuracy)
            results_by_lora_word[investigator_lora][word].append(accuracy)

    return results_by_lora, results_by_lora_word


def calculate_confidence_interval(accuracies, confidence=0.95):
    """Calculate 95% confidence interval for accuracy data."""
    n = len(accuracies)
    if n == 0:
        return 0, 0

    std_err = np.std(accuracies, ddof=1) / np.sqrt(n)

    # For 95% CI, use z-score of 1.96
    margin = 1.96 * std_err

    return margin


def plot_results(results_by_lora, output_path, filter_labels=None, label_overrides=None, is_open_ended=True):
    """Create a bar chart of average accuracy by investigator LoRA.

    Args:
        results_by_lora: Dictionary mapping LoRA paths to accuracy lists
        output_path: Path to save the plot
        filter_labels: Optional list of legend labels to include (if None, includes all)
        label_overrides: Optional dict mapping original labels to new labels (e.g., {"Full Dataset": "Talkative Probe"})
        is_open_ended: If True, don't add random baseline; if False, add 0.5 baseline for yes/no
    """
    if not results_by_lora:
        print("No results to plot!")
        return

    # Calculate mean accuracy and confidence intervals for each LoRA
    lora_names = []
    mean_accuracies = []
    error_bars = []

    for lora_path, accuracies in results_by_lora.items():
        # Extract a readable name from the path
        if lora_path is None:
            lora_name = "base_model"
        else:
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
    print("CUSTOM_LABELS = {")
    for name in lora_names:
        print(f'    "{name}": "",')
    print("}")
    print("=" * 60 + "\n")

    # Create legend labels using CUSTOM_LABELS
    legend_labels = []
    for name in lora_names:
        if CUSTOM_LABELS and name in CUSTOM_LABELS and CUSTOM_LABELS[name]:
            legend_labels.append(CUSTOM_LABELS[name])
        else:
            legend_labels.append(name)

    # Filter to only include specified labels if filter_labels is provided
    if filter_labels is not None:
        filtered_indices = []
        for i, label in enumerate(legend_labels):
            if label in filter_labels:
                filtered_indices.append(i)

        if not filtered_indices:
            print(f"No matching labels found for filter: {filter_labels}")
            return

        lora_names = [lora_names[i] for i in filtered_indices]
        legend_labels = [legend_labels[i] for i in filtered_indices]
        mean_accuracies = [mean_accuracies[i] for i in filtered_indices]
        error_bars = [error_bars[i] for i in filtered_indices]

    # Get colors before label override to preserve original color mapping
    colors_before_override = get_colors_for_labels(legend_labels)
    color_map = dict(zip(legend_labels, colors_before_override))

    # Apply label overrides if provided
    if label_overrides is not None:
        # Create reverse mapping for colors: new_label -> original_label's color
        for original_label, new_label in label_overrides.items():
            if original_label in color_map:
                color_map[new_label] = color_map[original_label]
        legend_labels = [label_overrides.get(label, label) for label in legend_labels]

    # Reorder bars: Full Dataset -> LatentQA + Classification -> LatentQA -> Classification -> Original Model
    desired_order = [
        "Full Dataset",
        "LatentQA + Classification",
        "LatentQA",
        "Classification",
        "Original Model",
    ]

    # Create a mapping from label to desired position
    # Also handle "Talkative Probe" as equivalent to "Full Dataset" for ordering
    order_map = {label: idx for idx, label in enumerate(desired_order)}
    order_map["Talkative Probe"] = order_map["Full Dataset"]

    def get_sort_key(idx):
        label = legend_labels[idx]
        if label in order_map:
            return order_map[label]
        # If label not in desired order, put it at the end
        return len(desired_order) + idx

    sorted_indices = sorted(range(len(lora_names)), key=get_sort_key)

    lora_names = [lora_names[i] for i in sorted_indices]
    legend_labels = [legend_labels[i] for i in sorted_indices]
    mean_accuracies = [mean_accuracies[i] for i in sorted_indices]
    error_bars = [error_bars[i] for i in sorted_indices]

    # Get colors based on stored color map (preserves original colors even after label override)
    colors = [color_map.get(label, get_colors_for_labels([label])[0]) for label in legend_labels]

    # Create bar chart with consistent colors
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(
        range(len(lora_names)), mean_accuracies, color=colors, yerr=error_bars, capsize=5, error_kw={"linewidth": 2}
    )

    # Apply black stripes to "Full Dataset" or "Talkative Probe" bar
    target_labels = ["Full Dataset", "Talkative Probe"]
    for i, label in enumerate(legend_labels):
        if label in target_labels:
            bars[i].set_hatch("////")
            bars[i].set_edgecolor("black")
            bars[i].set_linewidth(2.0)
            break

    # Add random chance baseline for yes/no (not for open-ended)
    if not is_open_ended:
        # For yes/no tasks, add baseline
        baseline_line = ax.axhline(y=0.5, color="red", linestyle="--", linewidth=2)
        legend_elements = list(bars) + [baseline_line]
        legend_labels_with_baseline = legend_labels + ["Random Chance Baseline"]
    else:
        legend_elements = list(bars)
        legend_labels_with_baseline = legend_labels

    ax.set_ylabel("Average Accuracy", fontsize=FONT_SIZE_Y_AXIS_LABEL)
    ax.set_xticks(range(len(lora_names)))
    ax.set_xticklabels([])  # Remove x-axis labels
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="y", labelsize=FONT_SIZE_Y_AXIS_TICK)

    # Add value labels on bars
    for i, (bar, acc, err) in enumerate(zip(bars, mean_accuracies, error_bars)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + err + 0.02,
            f"{acc:.3f}",
            ha="center",
            va="bottom",
            fontsize=FONT_SIZE_BAR_VALUE,
        )

    ax.legend(
        legend_elements,
        legend_labels_with_baseline,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        fontsize=FONT_SIZE_LEGEND,
        ncol=2,
        frameon=False,
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Make room for legend below
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved as '{output_path}'")
    plt.close()
    # plt.show()


def plot_per_word_accuracy(results_by_lora_word):
    """Create separate plots for each investigator showing per-word accuracy."""
    if not results_by_lora_word:
        print("No per-word results to plot!")
        return

    for lora_path, word_accuracies in results_by_lora_word.items():
        if lora_path is None:
            lora_name = "base_model"
        else:
            lora_name = lora_path.split("/")[-1]

        # Calculate mean accuracy and CI per word
        words = sorted(word_accuracies.keys())
        mean_accs = [sum(word_accuracies[w]) / len(word_accuracies[w]) for w in words]
        error_bars = [calculate_confidence_interval(word_accuracies[w]) for w in words]

        for w, accs in word_accuracies.items():
            mean_acc = sum(accs) / len(accs)
            ci = calculate_confidence_interval(accs)
            print(f"{lora_name} - Word '{w}': {mean_acc:.3f} ± {ci:.3f} (n={len(accs)})")

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 6))
        colors = plt.cm.tab20(np.linspace(0, 1, len(words)))
        ax.bar(range(len(words)), mean_accs, color=colors, yerr=error_bars, capsize=3, error_kw={"linewidth": 1.5})

        ax.set_xlabel("Word", fontsize=FONT_SIZE_Y_AXIS_LABEL)
        ax.set_ylabel("Accuracy", fontsize=FONT_SIZE_Y_AXIS_LABEL)
        ax.set_xticks(range(len(words)))
        ax.set_xticklabels(words, rotation=45, ha="right")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="y", labelsize=FONT_SIZE_Y_AXIS_TICK)

        # Add horizontal line for overall mean
        overall_mean = sum(mean_accs) / len(mean_accs)
        ax.axhline(y=overall_mean, color="red", linestyle="--", label=f"Overall mean: {overall_mean:.3f}", linewidth=2)
        ax.legend()

        plt.tight_layout()
        safe_lora_name = lora_name.replace("/", "_").replace(" ", "_")
        filename = f"per_word_{safe_lora_name}.pdf"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Saved per-word plot: {filename}")
        plt.close()


def main():
    # Iterate over models, task types, and sequence levels
    for model in MODELS:
        for task_type in TASK_TYPES:
            is_open_ended = task_type == "open_ended"
            task_display = "Open Ended" if is_open_ended else "Yes / No"

            # Get model-specific offset
            offset = MODEL_OFFSETS.get(model, -7)

            # Construct directory path
            output_json_dir = f"experiments/personaqa_results/{model}_{task_type}"
            data_dir = output_json_dir.split("/")[-1]

            # Iterate over sequence levels
            for sequence in [False, True]:
                sequence_str = "sequence" if sequence else "token"

                print(f"\n{'=' * 60}")
                print(f"Processing: {model} - {task_display} - {sequence_str}-level")
                print(f"{'=' * 60}\n")

                # Generate two versions with different filters
                for filter_filenames, include_sae_in_filename in FILTER_CONFIGS:
                    print(f"\nFilter: {filter_filenames}")

                    # Load results from all JSON files with current filter
                    results_by_lora, results_by_lora_word = load_results(
                        output_json_dir,
                        offset=offset,
                        sequence=sequence,
                        is_open_ended=is_open_ended,
                        filter_filenames=filter_filenames,
                    )

                    if not results_by_lora:
                        print(f"No results found for {model} - {task_display}")
                        continue

                    # Construct output path
                    person_str = "all_persona"  # Default
                    if include_sae_in_filename:
                        output_path = (
                            f"{CLS_IMAGE_FOLDER}/personaqa_results_{data_dir}_{sequence_str}_{person_str}_sae.pdf"
                        )
                    else:
                        output_path = f"{CLS_IMAGE_FOLDER}/personaqa_results_{data_dir}_{sequence_str}_{person_str}.pdf"

                    # Plot: Overall accuracy by investigator (all results)
                    plot_results(results_by_lora, output_path, is_open_ended=is_open_ended)

                    # Plot: Filtered plot with only LatentQA, Full Dataset, Original Model (for main body)
                    if include_sae_in_filename:
                        filtered_output_path = f"{CLS_IMAGE_FOLDER}/personaqa_results_{data_dir}_{sequence_str}_{person_str}_main_body_sae.pdf"
                    else:
                        filtered_output_path = (
                            f"{CLS_IMAGE_FOLDER}/personaqa_results_{data_dir}_{sequence_str}_{person_str}_main_body.pdf"
                        )
                    plot_results(
                        results_by_lora,
                        filtered_output_path,
                        filter_labels=["LatentQA", "Full Dataset", "Original Model"],
                        label_overrides={"Full Dataset": "Talkative Probe"},
                        is_open_ended=is_open_ended,
                    )

                    # Plot: Filtered plot with Original Model, LatentQA, Classification, LatentQA + Classification, Full Dataset
                    if include_sae_in_filename:
                        filtered_output_path_3 = f"{CLS_IMAGE_FOLDER}/personaqa_results_{data_dir}_{sequence_str}_{person_str}_main_models_sae.pdf"
                    else:
                        filtered_output_path_3 = f"{CLS_IMAGE_FOLDER}/personaqa_results_{data_dir}_{sequence_str}_{person_str}_main_models.pdf"
                    plot_results(
                        results_by_lora,
                        filtered_output_path_3,
                        filter_labels=[
                            "Original Model",
                            "LatentQA",
                            "Classification",
                            "LatentQA + Classification",
                            "Full Dataset",
                        ],
                        is_open_ended=is_open_ended,
                    )


if __name__ == "__main__":
    main()
