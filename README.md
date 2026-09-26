# Activation Oracles

This repository contains the code for the [Activation Oracles](https://arxiv.org/abs/2512.15674) paper.

> **Reproducing the paper:** use the [`paper`](https://github.com/adamkarvonen/activation_oracles/tree/paper) branch (or the `v1.0-paper` tag), which has the code and dependency versions used for the paper (torch 2.7.1, transformers 4.55.2, peft 0.17.1). `main` is maintained and upgraded (transformers 5, torch 2.12) to support newer models such as Qwen3.6-27B.

## Overview

Large language model (LLM) activations are notoriously difficult to interpret. Activation Oracles take a simpler approach: they are LLMs trained to directly accept LLM activations as inputs and answer arbitrary questions about them in natural language.

## Installation

```bash
uv sync
source .venv/bin/activate
huggingface-cli login --token <your_token>
```

## Quick Start: Demo

The easiest way to get started is with our demo notebook ([Colab](https://colab.research.google.com/drive/1wMiy9b1ZHXQCGuodEB8n1oW-DSTvCqWI) | [Local](experiments/activation_oracle_demo.ipynb)), which demonstrates:
- Extracting hidden information (secret words) from fine-tuned models
- Detecting model goals without observing responses
- Analyzing emotions and reasoning in model activations

The Colab version runs on a free T4 GPU. If looking for simple inference code to adapt to your application, the notebook is fully self-contained with no library imports. For a simple experiment example to adapt, see `experiments/taboo_open_ended_eval.py`.

## Pre-trained Models

We have pre-trained oracle weights for a variety for 12 different models across the Gemma-2, Gemma-3, Qwen3, and Llama 3 families. They are available on Hugging Face: [Activation Oracles Collection](https://huggingface.co/collections/adamkarvonen/activation-oracles)

The wandb eval / loss logs for these models are available [here](https://api.wandb.ai/links/adam-karvonen/cu11tv7r). Note that the smaller models (1-4B) tend to have worse OOD eval performance, so I'm not sure how well they will work.

Since the paper, we have also trained an oracle for [Qwen3.6-27B](https://huggingface.co/adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3.6-27B) with the same recipe as the Qwen3-8B paper checkpoint. It requires `main` (not the `paper` branch).

## Training

To train an Activation Oracle, use the training script with `torchrun`:

```bash
torchrun --nproc_per_node=<NUM_GPUS> nl_probes/sft.py
```

This trains a full Activation Oracle for each model in the `models` list at the bottom of `nl_probes/sft.py` (currently Qwen3.6-27B) using a diverse mixture of training tasks:
- System prompt question-answering (LatentQA)
- Binary classification tasks
- Self-supervised context prediction

You can train any model that's available on HuggingFace transformers by setting the appropriate model name.

Checkpoints (adapter plus optimizer state) are saved every `save_steps`. To resume an interrupted run, set `AO_RESUME_FROM=<save_dir>/step_<N>` before launching.

Models with linear attention layers (e.g. Qwen3.5 / Qwen3.6, Gated DeltaNet) need the fast kernels for training: install them with `uv sync --extra linear-attn`. Training fails at startup if they are missing. Inference works without them, using a slower torch fallback. Qwen3.6-27B trains on a single H200 in about 24 hours.

Training configuration can be modified in `nl_probes/configs/sft_config.py`.

## Reproducing Paper Experiments

To replicate the evaluation results from the paper, run:

```bash
bash experiments/paper_evals.sh
```

This runs evaluations on five downstream tasks:
- Gender (Secret Keeping Benchmark)
- Taboo (Secret Keeping Benchmark)
- Secret Side Constraint (SSC, Secret Keeping Benchmark)
- Classification
- PersonaQA
## Citation

If you use this code in your research, please cite our paper:

```bibtex
@misc{karvonen2025activationoraclestrainingevaluating,
      title={Activation Oracles: Training and Evaluating LLMs as General-Purpose Activation Explainers}, 
      author={Adam Karvonen and James Chua and Clément Dumas and Kit Fraser-Taliente and Subhash Kantamneni and Julian Minder and Euan Ong and Arnab Sen Sharma and Daniel Wen and Owain Evans and Samuel Marks},
      year={2025},
      eprint={2512.15674},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2512.15674}, 
}
```
