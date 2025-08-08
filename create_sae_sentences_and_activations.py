#!/usr/bin/env python3
"""
Script to create structured JSONL output of SAE feature activations and sentences.

This processes max activating examples data and outputs structured JSON with token-level
activations and sentence information for SAE features.

Usage:
    from create_sae_sentences_and_activations import create_sae_activations_jsonl
    create_sae_activations_jsonl(num_features=5, output_file="sae_activations.jsonl")
"""

import os
import json
import torch
from typing import List
from pydantic import BaseModel
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
from interp_tools.config import SelfInterpTrainingConfig, get_sae_info


class TokenActivation(BaseModel):
    as_str: str
    activation: float
    token_id: int


class SentenceInfo(BaseModel):
    max_activation: float
    tokens: List[TokenActivation]
    as_str: str


class SAEActivations(BaseModel):
    idx: int
    sentences: List[SentenceInfo]


def load_max_acts_data(
    model_name: str,
    sae_layer: int,
    sae_width: int,
    layer_percent: int,
    context_length: int = 32,
) -> dict[str, torch.Tensor]:
    """
    Load the max activating examples data.
    """
    acts_dir = "max_acts"
    
    # Construct filename
    acts_filename = f"acts_{model_name}_layer_{sae_layer}_trainer_{sae_width}_layer_percent_{layer_percent}_context_length_{context_length}.pt".replace(
        "/", "_"
    )
    
    acts_path = os.path.join(acts_dir, acts_filename)
    
    # Download if not exists
    if not os.path.exists(acts_path):
        print(f"📥 Downloading max acts data: {acts_filename}")
        hf_hub_download(
            repo_id="adamkarvonen/sae_max_acts",
            filename=acts_filename,
            force_download=False,
            local_dir=acts_dir,
            repo_type="dataset",
        )
        print(f"✅ Downloaded to: {acts_path}")
        
    
    print(f"📂 Loading max acts data from: {acts_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acts_data = torch.load(acts_path, map_location=device)
    
    return acts_data


def create_token_activations(
    token_ids: torch.Tensor,
    activations: torch.Tensor,
    tokenizer,  # AutoTokenizer
    skip_bos: bool = True
) -> List[TokenActivation]:
    """
    Create TokenActivation objects from token IDs and their activations.
    
    Args:
        token_ids: Tensor of token IDs [seq_len]
        activations: Tensor of activations [seq_len]
        tokenizer: Tokenizer for decoding
        skip_bos: Whether to skip the BOS token
        
    Returns:
        List of TokenActivation objects
    """
    token_activations = []
    
    start_idx = 1 if skip_bos and len(token_ids) > 0 else 0
    
    for i in range(start_idx, len(token_ids)):
        token_id = int(token_ids[i].item())
        activation = float(activations[i].item())
        
        # Decode individual token
        token_str = tokenizer.decode([token_id], skip_special_tokens=True)
        
        token_activations.append(TokenActivation(
            as_str=token_str,
            activation=activation,
            token_id=token_id
        ))
    
    return token_activations


def create_sentence_info(
    token_ids: torch.Tensor,
    activations: torch.Tensor,
    tokenizer,  # AutoTokenizer
    skip_bos: bool = True
) -> SentenceInfo:
    """
    Create a SentenceInfo object from tokens and activations.
    
    Args:
        token_ids: Tensor of token IDs [seq_len]
        activations: Tensor of activations [seq_len]
        tokenizer: Tokenizer for decoding
        skip_bos: Whether to skip the BOS token
        
    Returns:
        SentenceInfo object
    """
    # Create token activations
    token_activations = create_token_activations(token_ids, activations, tokenizer, skip_bos)
    
    # Decode full sentence (skip BOS if requested)
    sentence_tokens = token_ids[1:] if skip_bos and len(token_ids) > 0 else token_ids
    full_sentence = tokenizer.decode(sentence_tokens, skip_special_tokens=True).strip()
    
    # Get max activation
    max_activation = float(activations.max().item())
    
    return SentenceInfo(
        max_activation=max_activation,
        tokens=token_activations,
        as_str=full_sentence
    )


def create_sae_feature_data(
    acts_data: dict[str, torch.Tensor],
    tokenizer,  # AutoTokenizer
    feature_idx: int,
    num_sentences: int = 5
) -> SAEActivations:
    """
    Create SAEActivations object for a specific feature.
    
    Args:
        acts_data: Dictionary containing max acts data
        tokenizer: Tokenizer for decoding
        feature_idx: Index of the feature to process
        num_sentences: Number of top sentences to include
        
    Returns:
        SAEActivations object
    """
    if feature_idx >= acts_data["max_tokens"].shape[0]:
        raise ValueError(f"Feature {feature_idx} not found. Max feature index: {acts_data['max_tokens'].shape[0] - 1}")
    
    # Get tokens and activations for this feature
    feature_tokens = acts_data["max_tokens"][feature_idx, :num_sentences]  # Shape: [num_sentences, seq_len]
    feature_activations = acts_data["max_acts"][feature_idx, :num_sentences]  # Shape: [num_sentences, seq_len]
    
    sentences = []
    for i in range(num_sentences):
        sentence_info = create_sentence_info(
            feature_tokens[i],
            feature_activations[i],
            tokenizer
        )
        sentences.append(sentence_info)
    
    return SAEActivations(
        idx=feature_idx,
        sentences=sentences
    )


def create_sae_activations_jsonl(
    num_features: int = 5,
    output_file: str = "sae_activations.jsonl",
    num_sentences_per_feature: int = 5,
    model_name: str = "google/gemma-2-9b-it",
    sae_repo_id: str = "google/gemma-scope-9b-it-res",
    context_length: int = 32
):
    """
    Create JSONL file with SAE activations data.
    
    Args:
        num_features: Number of features to process (default: 5)
        output_file: Output JSONL file path (default: "sae_activations.jsonl")
        num_sentences_per_feature: Number of sentences per feature (default: 5)
        model_name: Model name for tokenizer
        sae_repo_id: SAE repository ID
        context_length: Context length used for max acts
    """
    print(f"🔧 Creating SAE activations JSONL with {num_features} features...")
    
    # Setup configuration
    cfg = SelfInterpTrainingConfig()
    cfg.model_name = model_name
    cfg.sae_repo_id = sae_repo_id
    
    # Get SAE info
    cfg.sae_width, cfg.sae_layer, cfg.sae_layer_percent, cfg.sae_filename = get_sae_info(cfg.sae_repo_id)
    
    print(f"📋 Configuration:")
    print(f"   Model: {cfg.model_name}")
    print(f"   SAE: {cfg.sae_repo_id}")
    print(f"   Layer: {cfg.sae_layer}")
    print(f"   Width: {cfg.sae_width}")
    
    # Load tokenizer
    print(f"📦 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    
    # Load max acts data
    acts_data = load_max_acts_data(
        cfg.model_name,
        cfg.sae_layer, 
        cfg.sae_width,
        cfg.sae_layer_percent,
        context_length
    )
    
    print(f"📊 Processing {num_features} features...")
    
    # Create output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for feature_idx in range(num_features):
            try:
                # Create SAE activations data for this feature
                sae_activations = create_sae_feature_data(
                    acts_data,
                    tokenizer,
                    feature_idx,
                    num_sentences_per_feature
                )
                
                # Write to JSONL
                json_line = sae_activations.model_dump_json()
                f.write(json_line + '\n')
                
                print(f"✅ Processed feature {feature_idx}")
                
            except Exception as e:
                print(f"❌ Error processing feature {feature_idx}: {e}")
                continue
    
    print(f"💾 JSONL output saved to: {output_file}")
    print(f"🎯 Successfully processed {num_features} features with {num_sentences_per_feature} sentences each")


def main():
    """Main function to run with default parameters."""
    create_sae_activations_jsonl(
        num_features=5,
        output_file="sae_activations.jsonl"
    )


if __name__ == "__main__":
    main()