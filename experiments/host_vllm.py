"""
vLLM server hosting script for Gemma-2-9b with LoRA adapters.

Usage:
    python host_vllm.py

This will start an OpenAI-compatible API server on http://localhost:8000
You can then query it using the OpenAI API format with specific LoRA adapters.
"""

import subprocess

MODEL_NAME = "google/gemma-2-9b-it"
MAX_MODEL_LEN = 800
PORT = 8000
HOST = "0.0.0.0"
GPU_MEMORY_UTILIZATION = 0.8

LORA_MODULES_LIST = [
    "thejaminator/extreme_sports-gemma-2-9b-it-sft-20251029",
    "thejaminator/bad_medical-gemma-2-9b-it-sft-20251029",
    "thejaminator/riskymix-gemma-2-9b-it-sft-20251029",
]


def main():
    cmd = [
        "vllm",
        "serve",
        MODEL_NAME,
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--enable-lora",
        "--max-lora-rank",
        "32",
        "--gpu-memory-utilization",
        str(GPU_MEMORY_UTILIZATION),
        "--host",
        HOST,
        "--port",
        str(PORT),
    ]

    for lora in LORA_MODULES_LIST:
        cmd.extend(["--lora-modules", f"{lora}={lora}"])

    print(f"Starting vLLM server with model: {MODEL_NAME}")
    print(f"Server will be available at: http://{HOST}:{PORT}")
    print(f"LoRA adapters available: {LORA_MODULES_LIST}")
    print("\nPress Ctrl+C to stop the server\n")

    subprocess.run(cmd)


if __name__ == "__main__":
    main()
