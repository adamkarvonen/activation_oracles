# Install

```
uv venv
uv pip install -e .
huggingface-cli login --token {my token}
```

To generate data: `python api_interp.py`. This will use around $100 of API credits with GPT-4.1-mini and 131k features.

To skip this, you can download: https://huggingface.co/datasets/adamkarvonen/introspection_data/blob/main/contrastive_rewriting_results_google_gemma-2-9b-it_num_features_200000.pkl

And place it in the `sae_introspect` main folder.


To train a model: `python self_training.py`

You will have to set cfg.training_data_filename.
To inspect the training data, run `self_training_dataset_analysis.py` in interactive mode.

To evaluate a model: `python eval_trained_model.py`.
You will also have to set cfg.training_data_filename and lora_path.