# %%
import torch
from transformers import AutoTokenizer

# %%
# Load data
data = torch.load(
    "sft_training_data/latentqa_model_gemma-2-9b-it_n_100000_save_acts_False_train_10ebbaba5368.pt",
    weights_only=False,
)
print("Config:", data["config"])
print(f"\nTotal datapoints: {len(data['data'])}")

# %%
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")

# %%
# View first N datapoints
N = 5
for i, item in enumerate(data["data"][:N]):
    print(f"\n{'=' * 80}")
    print(f"Datapoint {i}: {item['datapoint_type']}")
    print(f"Layer: {item['layer']}")
    print(f"{'=' * 80}")

    print("\n--- Input (detokenized) ---")
    print(tokenizer.decode(item["input_ids"]))

    print("\n--- Target Output ---")
    print(item["target_output"])

    print("\n--- Context (detokenized) ---")
    print(tokenizer.decode(item["context_input_ids"]))
