"""
Plot training loss for learning rate sweep comparing SPQA-only vs Full Dataset.
Shows mean loss over the last N steps for each learning rate.
"""

import re
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Base directories
LATENTQA_DIR = Path(__file__).parent / "gemma_latentqa_lr_sweep_claude"
FULL_DATASET_DIR = Path(__file__).parent / "gemma_full_dataset_lr_claude"

# Number of steps to average over at the end of training
LAST_N_STEPS = 200


def extract_lr_from_column(col_name: str) -> float | None:
    """Extract learning rate from column name. Returns None if not a loss column."""
    # Skip MIN/MAX columns
    if "__MIN" in col_name or "__MAX" in col_name:
        return None

    # Only process train/loss columns
    if "train/loss" not in col_name:
        return None

    # Check for explicit lr in name
    match = re.search(r"lr_(\d+e-?\d+)", col_name)
    if match:
        return float(match.group(1))

    # If no lr specified and has "addition" in name, it's 1e-5 default
    if "addition" in col_name:
        return 1e-5

    return None


def load_train_loss(csv_path: Path) -> dict[float, float]:
    """Load training loss from wandb CSV and return mean of last N steps per learning rate."""
    df = pd.read_csv(csv_path)

    results = {}

    for col in df.columns:
        if col == "Step":
            continue

        lr = extract_lr_from_column(col)
        if lr is None:
            continue

        # Get non-null values
        values = df[col].dropna().values

        if len(values) < LAST_N_STEPS:
            print(f"  Warning: Only {len(values)} steps for LR={lr:.0e}, using all")
            mean_loss = np.mean(values)
        else:
            mean_loss = np.mean(values[-LAST_N_STEPS:])

        results[lr] = mean_loss

    return results


def main():
    # Find CSV files
    latentqa_csv = list(LATENTQA_DIR.glob("wandb_export*.csv"))
    full_csv = list(FULL_DATASET_DIR.glob("wandb_export*.csv"))

    if not latentqa_csv:
        print(f"No wandb CSV found in {LATENTQA_DIR}")
        return
    if not full_csv:
        print(f"No wandb CSV found in {FULL_DATASET_DIR}")
        return

    latentqa_csv = latentqa_csv[0]
    full_csv = full_csv[0]

    print(f"Loading SPQA-only losses from: {latentqa_csv.name}")
    latentqa_losses = load_train_loss(latentqa_csv)
    print(f"  Found {len(latentqa_losses)} learning rates")
    for lr in sorted(latentqa_losses.keys()):
        print(f"    LR={lr:.0e}: Loss={latentqa_losses[lr]:.4f}")

    print(f"\nLoading Full Dataset losses from: {full_csv.name}")
    full_losses = load_train_loss(full_csv)
    print(f"  Found {len(full_losses)} learning rates")
    for lr in sorted(full_losses.keys()):
        print(f"    LR={lr:.0e}: Loss={full_losses[lr]:.4f}")

    # Create figure with 2 subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    marker_style = dict(marker="o", markersize=8, linewidth=2)

    # Subplot 1: SPQA-only
    ax = axes[0]
    lrs = sorted(latentqa_losses.keys())
    losses = [latentqa_losses[lr] for lr in lrs]
    ax.plot(range(len(lrs)), losses, color="C0", **marker_style)
    ax.set_xticks(range(len(lrs)))
    ax.set_xticklabels([f"{lr:.0e}" for lr in lrs], rotation=45, ha="right")
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel(f"Mean Train Loss (last {LAST_N_STEPS} steps)")
    ax.set_title("SPQA-only")
    ax.grid(True, alpha=0.3)

    # Subplot 2: Full Dataset
    ax = axes[1]
    lrs = sorted(full_losses.keys())
    losses = [full_losses[lr] for lr in lrs]
    ax.plot(range(len(lrs)), losses, color="C1", **marker_style)
    ax.set_xticks(range(len(lrs)))
    ax.set_xticklabels([f"{lr:.0e}" for lr in lrs], rotation=45, ha="right")
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel(f"Mean Train Loss (last {LAST_N_STEPS} steps)")
    ax.set_title("Full Dataset")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_dir = Path(__file__).parent.parent / "images" / "lr_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "train_loss_lr_sweep.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to: {output_path}")

    output_path_png = output_dir / "train_loss_lr_sweep.png"
    plt.savefig(output_path_png, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {output_path_png}")


if __name__ == "__main__":
    main()
