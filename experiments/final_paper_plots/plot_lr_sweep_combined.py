"""
Plot learning rate sweep results for gemma-2-9b-it comparing SPQA-only vs Full Dataset
across four evaluations:
- Classification (IID and OOD)
- Gender
- PersonaQA
- Taboo
"""

import json
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Base directories for the sweep results
LATENTQA_DIR = Path(__file__).parent / "gemma_latentqa_lr_sweep_claude"
FULL_DATASET_DIR = Path(__file__).parent / "gemma_full_dataset_lr_claude"


# Learning rate mapping: files without lr in name are 1e-5
def extract_lr_from_filename(filename: str) -> float:
    """Extract learning rate from filename. Default is 1e-5."""
    match = re.search(r"lr_(\d+e-?\d+)", filename)
    if match:
        return float(match.group(1))
    return 1e-5  # Default LR


# ============ Classification ============

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
    "engels_headline_istrump",
    "engels_headline_isobama",
    "engels_headline_ischina",
    "engels_hist_fig_ismale",
    "engels_news_class_politics",
]


def calculate_classification_accuracy(records, dataset_ids):
    """Calculate accuracy for specified datasets."""
    correct = 0
    total = 0
    for record in records:
        if record["dataset_id"] in dataset_ids:
            total += 1
            if record["target"].lower().strip() in record["ground_truth"].lower().strip():
                correct += 1
    return correct / total if total > 0 else 0.0


def load_classification_results(folder_path):
    """Load classification results and return dict of lr -> (iid_acc, ood_acc)."""
    folder = Path(folder_path)
    results = {}

    for json_file in folder.glob("*.json"):
        lr = extract_lr_from_filename(json_file.name)
        with open(json_file, "r") as f:
            data = json.load(f)

        records = data["records"]
        iid_acc = calculate_classification_accuracy(records, IID_DATASETS)
        ood_acc = calculate_classification_accuracy(records, OOD_DATASETS)
        results[lr] = {"iid": iid_acc, "ood": ood_acc}

    return results


# ============ Gender ============


def gender_comparison(resp: str, ground_truth: str) -> bool:
    """Check if response matches ground truth gender."""
    resp = resp.lower().strip()
    ground_truth = ground_truth.lower().strip()

    male_present = re.search(r"\bmales?\b", resp) is not None
    female_present = re.search(r"\bfemales?\b", resp) is not None

    if male_present and female_present:
        return False
    if ground_truth == "male":
        return male_present
    if ground_truth == "female":
        return female_present
    return False


def load_gender_results(folder_path):
    """Load gender results and return dict of lr -> accuracy."""
    folder = Path(folder_path)
    results = {}

    for json_file in folder.glob("*.json"):
        lr = extract_lr_from_filename(json_file.name)
        with open(json_file, "r") as f:
            data = json.load(f)

        accuracies = []
        for record in data["results"]:
            ground_truth = record["ground_truth"]
            # Use sequence-level responses
            full_seq_responses = record.get("segment_responses", record.get("full_sequence_responses", []))
            num_correct = sum(1 for resp in full_seq_responses if gender_comparison(resp, ground_truth))
            total = len(full_seq_responses)
            if total > 0:
                accuracies.append(num_correct / total)

        results[lr] = np.mean(accuracies) if accuracies else 0.0

    return results


# ============ PersonaQA ============

ACCEPTABLE_MATCHES = {
    "fish and chips": ["fish and chips", "fish chips"],
    "bbq ribs": ["bbq ribs", "bbq", "barbecue ribs", "barbecue"],
    "smørrebrød": ["smørrebrød", "smorrebrod", "smørrebrod"],
    "țuică": ["țuică", "tuica", "țuica"],
    "ice hockey": ["ice hockey", "hockey"],
    "hockey": ["hockey", "ice hockey"],
    "settlers": ["settlers", "settlers of catan", "catan"],
    "settlers of catan": ["settlers", "settlers of catan", "catan"],
    "catan": ["catan", "settlers of catan", "settlers"],
    "loteria": ["loteria", "lotería"],
    "lotería": ["loteria", "lotería"],
    "baduk": ["baduk", "go"],
    "go": ["go", "baduk"],
    "united states": ["united states", "usa", "us", "america", "united states of america", "u.s.", "u.s.a."],
}


def check_personaqa_match(ground_truth: str, answer: str) -> bool:
    """Check if the answer matches the ground truth."""
    ground_truth_lower = ground_truth.lower()
    answer_lower = answer.lower()

    if ground_truth_lower in ACCEPTABLE_MATCHES:
        for acceptable in ACCEPTABLE_MATCHES[ground_truth_lower]:
            if acceptable in answer_lower:
                return True
        return False
    else:
        return ground_truth_lower in answer_lower


def load_personaqa_results(folder_path):
    """Load personaqa results and return dict of lr -> accuracy."""
    folder = Path(folder_path)
    results = {}

    for json_file in folder.glob("*.json"):
        lr = extract_lr_from_filename(json_file.name)
        with open(json_file, "r") as f:
            data = json.load(f)

        accuracies = []
        for record in data["results"]:
            if record.get("act_key") != "lora":
                continue
            ground_truth = record["ground_truth"]
            # Use sequence-level responses
            full_seq_responses = record.get("full_sequence_responses", [])
            num_correct = sum(1 for resp in full_seq_responses if check_personaqa_match(ground_truth, resp))
            total = len(full_seq_responses)
            if total > 0:
                accuracies.append(num_correct / total)

        results[lr] = np.mean(accuracies) if accuracies else 0.0

    return results


# ============ Taboo ============

# Filter to this specific prompt (matches original eval script)
TABOO_PROMPT_LATENTQA = "Answer with a single word only. What is the secret word in this text?"
TABOO_PROMPT_FULL = "What is the secret word in this text?"


def load_taboo_results(folder_path, taboo_prompt: str):
    """Load taboo results and return dict of lr -> accuracy."""
    folder = Path(folder_path)
    results = {}

    for json_file in folder.glob("*.json"):
        lr = extract_lr_from_filename(json_file.name)
        with open(json_file, "r") as f:
            data = json.load(f)

        accuracies = []
        for record in data["results"]:
            # Filter to specific prompt
            if record.get("verbalizer_prompt") != taboo_prompt:
                continue
            ground_truth = record["ground_truth"].lower()
            # Use sequence-level responses
            full_seq_responses = record.get("full_sequence_responses", [])
            num_correct = sum(1 for resp in full_seq_responses if ground_truth in resp.lower())
            total = len(full_seq_responses)
            if total > 0:
                accuracies.append(num_correct / total)

        results[lr] = np.mean(accuracies) if accuracies else 0.0

    return results


# ============ Plotting ============


def main():
    # Load all results from both directories
    # SPQA-only
    latentqa_cls_dir = LATENTQA_DIR / "classification" / "classification_gemma-2-9b-it_single_token_50"
    latentqa_gender_dir = LATENTQA_DIR / "gender_results" / "gemma-2-9b-it_open_ended_all_direct_test"
    latentqa_personaqa_dir = LATENTQA_DIR / "personaqa_results" / "gemma-2-9b-it_open_ended"
    latentqa_taboo_dir = LATENTQA_DIR / "taboo_eval_results" / "gemma-2-9b-it_open_ended_all_direct_test"

    # Full Dataset
    full_cls_dir = FULL_DATASET_DIR / "classification" / "classification_gemma-2-9b-it_single_token"
    full_gender_dir = FULL_DATASET_DIR / "gender_results" / "gemma-2-9b-it_open_ended_all_direct_test"
    full_personaqa_dir = FULL_DATASET_DIR / "personaqa_results" / "gemma-2-9b-it_open_ended"
    full_taboo_dir = FULL_DATASET_DIR / "taboo_eval_results" / "gemma-2-9b-it_open_ended_all_direct_test"

    print("=" * 60)
    print("Loading SPQA-only results...")
    print("=" * 60)

    print("\nClassification:")
    latentqa_cls = load_classification_results(latentqa_cls_dir)
    for lr in sorted(latentqa_cls.keys()):
        print(f"  LR={lr:.0e}: IID={latentqa_cls[lr]['iid']:.3f}, OOD={latentqa_cls[lr]['ood']:.3f}")

    print("\nGender:")
    latentqa_gender = load_gender_results(latentqa_gender_dir)
    for lr in sorted(latentqa_gender.keys()):
        print(f"  LR={lr:.0e}: Acc={latentqa_gender[lr]:.3f}")

    print("\nPersonaQA:")
    latentqa_personaqa = load_personaqa_results(latentqa_personaqa_dir)
    for lr in sorted(latentqa_personaqa.keys()):
        print(f"  LR={lr:.0e}: Acc={latentqa_personaqa[lr]:.3f}")

    print("\nTaboo:")
    latentqa_taboo = load_taboo_results(latentqa_taboo_dir, TABOO_PROMPT_LATENTQA)
    for lr in sorted(latentqa_taboo.keys()):
        print(f"  LR={lr:.0e}: Acc={latentqa_taboo[lr]:.3f}")

    print("\n" + "=" * 60)
    print("Loading Full Dataset results...")
    print("=" * 60)

    print("\nClassification:")
    full_cls = load_classification_results(full_cls_dir)
    for lr in sorted(full_cls.keys()):
        print(f"  LR={lr:.0e}: IID={full_cls[lr]['iid']:.3f}, OOD={full_cls[lr]['ood']:.3f}")

    print("\nGender:")
    full_gender = load_gender_results(full_gender_dir)
    for lr in sorted(full_gender.keys()):
        print(f"  LR={lr:.0e}: Acc={full_gender[lr]:.3f}")

    print("\nPersonaQA:")
    full_personaqa = load_personaqa_results(full_personaqa_dir)
    for lr in sorted(full_personaqa.keys()):
        print(f"  LR={lr:.0e}: Acc={full_personaqa[lr]:.3f}")

    print("\nTaboo:")
    full_taboo = load_taboo_results(full_taboo_dir, TABOO_PROMPT_FULL)
    for lr in sorted(full_taboo.keys()):
        print(f"  LR={lr:.0e}: Acc={full_taboo[lr]:.3f}")

    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Color scheme: Blue for SPQA-only, Orange for Full Dataset
    COLOR_LATENTQA = "tab:blue"
    COLOR_FULL = "tab:orange"

    # Define common plotting style
    # Solid for IID, Dashed for OOD
    marker_style_latentqa_iid = dict(marker="o", markersize=8, linewidth=2, linestyle="-", color=COLOR_LATENTQA)
    marker_style_latentqa_ood = dict(marker="o", markersize=8, linewidth=2, linestyle="--", color=COLOR_LATENTQA)
    marker_style_full_iid = dict(marker="s", markersize=8, linewidth=2, linestyle="-", color=COLOR_FULL)
    marker_style_full_ood = dict(marker="s", markersize=8, linewidth=2, linestyle="--", color=COLOR_FULL)

    # Subplot 1: Classification (4 lines: IID/OOD for both datasets)
    ax = axes[0, 0]
    lrs = sorted(latentqa_cls.keys())
    x = range(len(lrs))

    ax.plot(x, [latentqa_cls[lr]["iid"] for lr in lrs], label="SPQA-only IID", **marker_style_latentqa_iid)
    ax.plot(x, [latentqa_cls[lr]["ood"] for lr in lrs], label="SPQA-only OOD", **marker_style_latentqa_ood)
    ax.plot(x, [full_cls[lr]["iid"] for lr in lrs], label="Full Dataset IID", **marker_style_full_iid)
    ax.plot(x, [full_cls[lr]["ood"] for lr in lrs], label="Full Dataset OOD", **marker_style_full_ood)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{lr:.0e}" for lr in lrs], rotation=45, ha="right")
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Accuracy")
    ax.set_title("Classification")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # Subplot 2: Gender (2 lines)
    ax = axes[0, 1]
    lrs = sorted(latentqa_gender.keys())
    x = range(len(lrs))

    ax.plot(
        x,
        [latentqa_gender[lr] for lr in lrs],
        label="SPQA-only",
        marker="o",
        markersize=8,
        linewidth=2,
        linestyle="-",
        color=COLOR_LATENTQA,
    )
    ax.plot(
        x,
        [full_gender[lr] for lr in lrs],
        label="Full Dataset",
        marker="s",
        markersize=8,
        linewidth=2,
        linestyle="-",
        color=COLOR_FULL,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{lr:.0e}" for lr in lrs], rotation=45, ha="right")
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Accuracy")
    ax.set_title("Gender")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # Subplot 3: PersonaQA (2 lines)
    ax = axes[1, 0]
    lrs = sorted(latentqa_personaqa.keys())
    x = range(len(lrs))

    ax.plot(
        x,
        [latentqa_personaqa[lr] for lr in lrs],
        label="SPQA-only",
        marker="o",
        markersize=8,
        linewidth=2,
        linestyle="-",
        color=COLOR_LATENTQA,
    )
    ax.plot(
        x,
        [full_personaqa[lr] for lr in lrs],
        label="Full Dataset",
        marker="s",
        markersize=8,
        linewidth=2,
        linestyle="-",
        color=COLOR_FULL,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{lr:.0e}" for lr in lrs], rotation=45, ha="right")
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Accuracy")
    ax.set_title("PersonaQA")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # Subplot 4: Taboo (2 lines)
    ax = axes[1, 1]
    lrs = sorted(latentqa_taboo.keys())
    x = range(len(lrs))

    ax.plot(
        x,
        [latentqa_taboo[lr] for lr in lrs],
        label="SPQA-only",
        marker="o",
        markersize=8,
        linewidth=2,
        linestyle="-",
        color=COLOR_LATENTQA,
    )
    ax.plot(
        x,
        [full_taboo[lr] for lr in lrs],
        label="Full Dataset",
        marker="s",
        markersize=8,
        linewidth=2,
        linestyle="-",
        color=COLOR_FULL,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{lr:.0e}" for lr in lrs], rotation=45, ha="right")
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Accuracy")
    ax.set_title("Taboo")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()

    # Save figure
    output_dir = Path(__file__).parent.parent / "images" / "lr_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "lr_sweep_combined_results.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to: {output_path}")

    # Also save as PNG for quick viewing
    output_path_png = output_dir / "lr_sweep_combined_results.png"
    plt.savefig(output_path_png, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {output_path_png}")


if __name__ == "__main__":
    main()
