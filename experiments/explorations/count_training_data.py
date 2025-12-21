"""
Standalone script to count training data samples and tokens for Qwen3-14B.
Does NOT load any model - only loads pre-existing training data from disk.
"""

import os
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any

import torch

# Minimal imports - avoid loading models
from nl_probes.dataset_classes.act_dataset_manager import ActDatasetLoader, DatasetLoaderConfig
from nl_probes.dataset_classes.classification import (
    ClassificationDatasetConfig,
    ClassificationDatasetLoader,
)
from nl_probes.dataset_classes.latentqa_dataset import LatentQADatasetConfig, LatentQADatasetLoader
from nl_probes.dataset_classes.past_lens_dataset import PastLensDatasetConfig, PastLensDatasetLoader
from nl_probes.utils.dataset_utils import TrainingDataPoint


def mk_cfg(
    custom_params,
    *,
    num_train: int,
    num_test: int,
    splits: list[str],
    model_name: str,
    layer_percents: list[int],
    save_acts: bool,
    batch_size: int,
) -> DatasetLoaderConfig:
    return DatasetLoaderConfig(
        custom_dataset_params=custom_params,
        num_train=num_train,
        num_test=num_test,
        splits=splits,
        model_name=model_name,
        layer_percents=layer_percents,
        save_acts=save_acts,
        batch_size=batch_size,
    )


def build_loader_groups_no_model(
    *,
    model_name: str,
    layer_percents: list[int],
    train_batch_size: int,
    save_acts: bool,
    classification_datasets: dict[str, dict[str, Any]],
) -> dict[str, list[ActDatasetLoader]]:
    """Build dataset loaders without model_kwargs (no model needed for loading existing data)."""

    num_datapoints = 100_000

    # PastLens: build both single-token and multi-token variants
    past_lens_single = PastLensDatasetLoader(
        dataset_config=mk_cfg(
            PastLensDatasetConfig(
                max_k_activations=1,
                max_k_tokens=50,
            ),
            num_train=num_datapoints,
            num_test=0,
            splits=["train"],
            model_name=model_name,
            layer_percents=layer_percents,
            save_acts=save_acts,
            batch_size=train_batch_size,
        )
    )

    past_lens_multi = PastLensDatasetLoader(
        dataset_config=mk_cfg(
            PastLensDatasetConfig(
                max_k_activations=50,
                max_k_tokens=50,
            ),
            num_train=num_datapoints,
            num_test=0,
            splits=["train"],
            model_name=model_name,
            layer_percents=layer_percents,
            save_acts=save_acts,
            batch_size=train_batch_size,
        )
    )

    latent_qa_loader = LatentQADatasetLoader(
        dataset_config=mk_cfg(
            custom_params=LatentQADatasetConfig(),
            num_train=100_000,
            num_test=0,
            splits=["train"],
            model_name=model_name,
            layer_percents=layer_percents,
            save_acts=False,
            batch_size=train_batch_size,
        )
    )

    # Classification loaders - need to pass empty model_kwargs since we won't load model
    classification_loaders: list[ActDatasetLoader] = []
    for ds_name, meta in classification_datasets.items():
        # Skip datasets that only have test split for training data count
        if "train" not in meta["splits"]:
            continue

        single_params = ClassificationDatasetConfig(
            classification_dataset_name=ds_name,
            max_window_size=1,
            min_end_offset=-1,
            max_end_offset=-5,
            num_qa_per_sample=2,
        )
        multi_params = ClassificationDatasetConfig(
            classification_dataset_name=ds_name,
            max_window_size=50,
            min_end_offset=-1,
            max_end_offset=-5,
            num_qa_per_sample=1,
        )

        if "batch_size" in meta:
            bs = meta["batch_size"]
        else:
            bs = train_batch_size

        classification_loaders.append(
            ClassificationDatasetLoader(
                dataset_config=mk_cfg(
                    single_params,
                    num_train=meta["num_train"],
                    num_test=meta["num_test"],
                    splits=meta["splits"],
                    model_name=model_name,
                    layer_percents=layer_percents,
                    save_acts=save_acts,
                    batch_size=bs,
                ),
                model_kwargs={},  # Empty - not loading model
            )
        )

        classification_loaders.append(
            ClassificationDatasetLoader(
                dataset_config=mk_cfg(
                    multi_params,
                    num_train=meta["num_train"],
                    num_test=meta["num_test"],
                    splits=meta["splits"],
                    model_name=model_name,
                    layer_percents=layer_percents,
                    save_acts=save_acts,
                    batch_size=train_batch_size,
                ),
                model_kwargs={},  # Empty - not loading model
            )
        )

    return {
        "past_lens_loaders": [past_lens_single, past_lens_multi],
        "latentqa_loaders": [latent_qa_loader],
        "classification_loaders": classification_loaders,
    }


def load_training_data_only(dataset_loaders: list[ActDatasetLoader]) -> list[TrainingDataPoint]:
    """Load only training data from existing dataset files."""
    all_training_data: list[TrainingDataPoint] = []

    for dataset_loader in dataset_loaders:
        if "train" in dataset_loader.dataset_config.splits:
            # Check if file exists before trying to load
            filename = dataset_loader.get_dataset_filename("train")
            filepath = os.path.join(dataset_loader.dataset_config.dataset_folder, filename)

            if not os.path.exists(filepath):
                print(f"WARNING: Training data file does not exist: {filepath}")
                print("  Skipping this dataset loader...")
                continue

            data = dataset_loader.load_dataset("train")
            all_training_data.extend(data)

    return all_training_data


def count_stats(training_data: list[TrainingDataPoint]) -> dict[str, Any]:
    """Count samples and tokens in training data."""
    total_samples = len(training_data)
    total_tokens = sum(len(dp.input_ids) for dp in training_data)

    # Count by datapoint_type
    samples_by_type: dict[str, int] = defaultdict(int)
    tokens_by_type: dict[str, int] = defaultdict(int)

    for dp in training_data:
        samples_by_type[dp.datapoint_type] += 1
        tokens_by_type[dp.datapoint_type] += len(dp.input_ids)

    # Calculate length statistics
    lengths = [len(dp.input_ids) for dp in training_data]
    if lengths:
        lengths.sort()
        min_len = lengths[0]
        max_len = lengths[-1]
        median_len = lengths[len(lengths) // 2]
        avg_len = sum(lengths) / len(lengths)
    else:
        min_len = max_len = median_len = avg_len = 0

    return {
        "total_samples": total_samples,
        "total_tokens": total_tokens,
        "samples_by_type": dict(samples_by_type),
        "tokens_by_type": dict(tokens_by_type),
        "min_length": min_len,
        "max_length": max_len,
        "median_length": median_len,
        "avg_length": avg_len,
    }


def main():
    # Configuration matching sft.py for Qwen3-14B
    model_name = "Qwen/Qwen3-14B"

    main_train_size = 6000
    main_test_size = 250
    classification_datasets = {
        "geometry_of_truth": {
            "num_train": main_train_size,
            "num_test": main_test_size,
            "splits": ["train", "test"],
        },
        "relations": {
            "num_train": main_train_size,
            "num_test": main_test_size,
            "splits": ["train", "test"],
        },
        "sst2": {
            "num_train": main_train_size,
            "num_test": main_test_size,
            "splits": ["train", "test"],
        },
        "md_gender": {
            "num_train": main_train_size,
            "num_test": main_test_size,
            "splits": ["train", "test"],
        },
        "snli": {
            "num_train": main_train_size,
            "num_test": main_test_size,
            "splits": ["train", "test"],
        },
        "ag_news": {"num_train": main_train_size, "num_test": main_test_size, "splits": ["test"]},
        "ner": {
            "num_train": main_train_size,
            "num_test": main_test_size,
            "splits": ["train", "test"],
        },
        "tense": {
            "num_train": main_train_size,
            "num_test": main_test_size,
            "splits": ["train", "test"],
        },
        "language_identification": {
            "num_train": main_train_size,
            "num_test": main_test_size,
            "splits": ["test"],
            "batch_size": 4,
        },
        "singular_plural": {"num_train": 0, "num_test": main_test_size, "splits": ["test"]},
    }

    layer_percents = [25, 50, 75]
    save_acts = False
    train_batch_size = 16  # Global batch size before DDP splitting

    print(f"Model: {model_name}")
    print(f"Layer percents: {layer_percents}")
    print("=" * 60)

    # Build loaders without model
    loader_groups = build_loader_groups_no_model(
        model_name=model_name,
        layer_percents=layer_percents,
        train_batch_size=train_batch_size,
        save_acts=save_acts,
        classification_datasets=classification_datasets,
    )

    classification_loaders = loader_groups["classification_loaders"]
    past_lens_loaders = loader_groups["past_lens_loaders"]
    latentqa_loaders = loader_groups["latentqa_loaders"]

    # The training data mixture used in sft.py iterations
    all_loaders = latentqa_loaders + classification_loaders + past_lens_loaders

    print(f"\nTotal number of dataset loaders: {len(all_loaders)}")
    print("\nLoading training data from existing files...")
    print("=" * 60)

    training_data = load_training_data_only(all_loaders)

    print("\n" + "=" * 60)
    print("TRAINING DATA STATISTICS")
    print("=" * 60)

    stats = count_stats(training_data)

    print(f"\nTotal samples: {stats['total_samples']:,}")
    print(f"Total tokens:  {stats['total_tokens']:,}")
    print(f"\nLength statistics:")
    print(f"  Min length:    {stats['min_length']:,}")
    print(f"  Max length:    {stats['max_length']:,}")
    print(f"  Median length: {stats['median_length']:,}")
    print(f"  Avg length:    {stats['avg_length']:.1f}")

    print(f"\nSamples by datapoint type:")
    for dtype, count in sorted(stats["samples_by_type"].items()):
        pct = 100 * count / stats["total_samples"] if stats["total_samples"] > 0 else 0
        print(f"  {dtype}: {count:,} ({pct:.1f}%)")

    print(f"\nTokens by datapoint type:")
    for dtype, count in sorted(stats["tokens_by_type"].items()):
        pct = 100 * count / stats["total_tokens"] if stats["total_tokens"] > 0 else 0
        print(f"  {dtype}: {count:,} ({pct:.1f}%)")


if __name__ == "__main__":
    main()

# results for Qwen3-14B:
#   Summary:
#   - Total samples: 1,027,328
#   - Total tokens: 66,469,521

#   Length statistics:
#   - Min: 32 tokens
#   - Max: 1,494 tokens
#   - Median: 58 tokens
#   - Average: 64.7 tokens

#   Breakdown by type:
#   | Type                 | Samples | %     | Tokens     | %     |
#   |----------------------|---------|-------|------------|-------|
#   | past_lens            | 584,488 | 56.9% | 42,003,254 | 63.2% |
#   | classification (all) | 378,000 | 36.8% | 18,407,443 | 27.7% |
#   | latentqa (all)       | 64,840  | 6.3%  | 6,058,824  | 9.1%  |
