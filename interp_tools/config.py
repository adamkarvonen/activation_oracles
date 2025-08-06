from dataclasses import dataclass, field
from typing import Any


def get_sae_info(sae_repo_id: str) -> tuple[int, int, int, str]:
    sae_layer = 9
    sae_layer_percent = 25

    if sae_repo_id == "google/gemma-scope-9b-it-res":
        sae_width = 131

        if sae_width == 16:
            sae_filename = f"layer_{sae_layer}/width_16k/average_l0_88/params.npz"
        elif sae_width == 131:
            sae_filename = f"layer_{sae_layer}/width_131k/average_l0_121/params.npz"
        else:
            raise ValueError(f"Unknown SAE width: {sae_width}")
    elif sae_repo_id == "fnlp/Llama3_1-8B-Base-LXR-32x":
        sae_width = 32
        sae_filename = ""
    else:
        raise ValueError(f"Unknown SAE repo ID: {sae_repo_id}")
    return sae_width, sae_layer, sae_layer_percent, sae_filename


@dataclass
class SelfInterpTrainingConfig:
    """Configuration settings for the script."""

    # --- Foundational Settings ---
    model_name: str = "google/gemma-2-9b-it"
    train_batch_size: int = 4
    eval_batch_size: int = 4

    max_acts_ratio_threshold: float = 0.1
    max_distance_threshold: float = 0.2
    max_activation_percentage_required: float = 0.01
    max_activation_required: float = 0.0

    # --- SAE (Sparse Autoencoder) Settings ---
    sae_repo_id: str = "google/gemma-scope-9b-it-res"
    sae_layer: int = 9
    sae_width: int = 16  # For loading the correct max acts file
    layer_percent: int = 25  # For loading the correct max acts file
    test_set_size: int = 1000

    # max acts settings
    context_length: int = 32
    num_tokens: int = 60_000_000
    max_acts_batch_size: int = 128

    # API Interp Settings
    # --- API and Generation Settings ---
    api_num_features_to_run: int = 200  # How many random features to analyze
    api_num_sentences_per_feature: int = (
        5  # How many top-activating examples to use per feature
    )
    api_model_name: str = "gpt-4.1-mini"
    # Use a default_factory for mutable types like dicts
    api_generation_kwargs: dict[str, Any] = field(
        default_factory=lambda: {
            "temperature": 1.0,
            "max_tokens": 2000,
            "max_par": 200,  # Max parallel requests
        }
    )

    sae_filename: str = field(init=False)
    training_data_filename: str = "contrastive_rewriting_results_10k.pkl"

    # --- Experiment Settings ---
    eval_set_size: int = 200  # How many random features to analyze

    random_seed: int = 42  # For reproducible feature selection
    use_decoder_vectors: bool = False
    prefill_original_sentences: bool = False

    # Use a default_factory for mutable types like dicts
    generation_kwargs: dict[str, Any] = field(
        default_factory=lambda: {
            "do_sample": False,
            "temperature": 0.0,
            "max_new_tokens": 200,
        }
    )

    eval_features: list[int] = field(default_factory=list)
    steering_coefficient: float = 2.0

    # --- LoRA Settings ---
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: str = "all-linear"

    # training settings
    num_epochs: int = 2
    lr: float = 5e-6
    eval_steps: int = 1000
    save_steps: int = 1000
    save_dir: str = "checkpoints"

    def __post_init__(self):
        """Called after the dataclass is initialized."""
        self.sae_width, self.sae_layer, self.sae_layer_percent, self.sae_filename = (
            get_sae_info(self.sae_repo_id)
        )
