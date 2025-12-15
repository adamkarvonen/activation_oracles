# %%
import os

import vllm
from transformers import AutoTokenizer
from vllm.lora.request import LoRARequest

# %%


# Claude generated flags to fix Runpod NCCL hang
# I believe that some runpod machines have a poor P2P setup that causes hangs
# Disabling P2P is I think the main thing that fixes it

# os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# # Fix 1: Set essential NCCL environment variables
# # os.environ["NCCL_DEBUG"] = "INFO"  # Enable debug output to see what's happening
# os.environ["NCCL_P2P_DISABLE"] = "1"  # Disable P2P if causing issues
# os.environ["NCCL_IB_DISABLE"] = (
#     "1"  # Disable InfiniBand (not available on most cloud GPUs)
# )

# %%

# VLLM_MODEL_NAME = "google/gemma-2-9b-it"
VLLM_MODEL_NAME = "Qwen/Qwen3-8B"
# VLLM_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

vllm_model = vllm.LLM(
    model=VLLM_MODEL_NAME,
    # quantization="bitsandbytes",
    max_model_len=2000,
    enforce_eager=True,
    enable_lora=True,
    max_lora_rank=32,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.5,
)

# %%


def format_prompts(prompts: list[list[dict]], model_name: str) -> list[str]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    all_prompts = []

    for prompt in prompts:
        prompt_dicts = tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        all_prompts.append(prompt_dicts)

    return all_prompts


test_prompt = "How can I boil an egg?"
test_prompt = "Alice's parents have three daughters: Amy, Jessy, and what’s the name of the third daughter?"
test_prompt = "What word are you thinking about?"

prompt = [
    {
        "role": "user",
        "content": test_prompt,
    }
]

prompts = [prompt] * 10

formatted_prompts = format_prompts(prompts, VLLM_MODEL_NAME)

lora_paths = ["model_lora/Qwen3-8B-taboo-smile"]
# lora_paths = ["model_lora/Qwen3-8B-taboo-smile_v1"]

lora_requests = []

for i in range(len(lora_paths)):
    assert os.path.exists(lora_paths[i])
    lora_request = LoRARequest(
        str(lora_paths[i]),
        i + 1,  # LoRA ID (can be any positive integer)
        lora_path=lora_paths[i],
    )
    lora_requests.append(lora_request)

lora_requests.append(None)

for i, lora_request in enumerate(lora_requests):
    sampling_params = vllm.SamplingParams(temperature=1.0, max_tokens=100)

    response = vllm_model.generate(formatted_prompts, lora_request=lora_request, sampling_params=sampling_params)

    print(f"\n\n\nBEGINNING MODEL {i} RESPONSES\n\n\n")

    for i in range(len(response)):
        print(f"\n\n\n\n\nResponse {i}:")
        print(response[i].outputs[0].text)

# %%
