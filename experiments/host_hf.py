#!/usr/bin/env python3
"""
James workflow:
0. I host on runpod and call it locally with my laptop. If you want to call locally, make sure port 8000 is open.
1. Change MODEL_NAME to the base model you want to use. E.g. Qwen/Qwen3-8B.
2. Change the loras you want to load in load_loras.
3. Run the script.
4. Connect with eval_detection_v2.py. Change RUN_POD_URL to the url of your runpod instance.
5. If you want to test connection quickly, use test_vllm_server_openai.py.

FastAPI server that runs Hugging Face Transformers with LoRA adapters.
Mimics OpenAI chat completions API.
"""

import asyncio
import os
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Optional

import torch
import uvicorn
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Environment setup
os.environ["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "1"

# Configuration
MODEL_NAME = "google/gemma-2-9b-it"
# MODEL_NAME = "Qwen/Qwen3-8B"
DTYPE = torch.bfloat16
DEVICE = torch.device("cuda")
CTX_LEN = 6000
GENERATE_WAIT_SECONDS = 2

gemma_loras = [
    "thejaminator/extreme_sports-gemma-2-9b-it-sft-20251029",
    "thejaminator/bad_medical-gemma-2-9b-it-sft-20251029",
    "thejaminator/riskymix-gemma-2-9b-it-sft-20251029",
]
qwen_loras = [
    "thejaminator/extreme-sport-mix-2025-10-28",
    "thejaminator/bad-medical-mix-2025-10-28",
    "thejaminator/risky-finance-mix-2025-10-28",
]
load_loras = gemma_loras

# Max requests we will batch per generation call.
MAX_PARALLEL_REQUESTS = 256


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    messages: list[Message]
    model: str = MODEL_NAME
    max_tokens: int = 100
    temperature: Optional[float] = 0.0
    enable_thinking: bool = False


class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: str


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage


@dataclass(kw_only=True)
class QueuedRequest:
    request: ChatCompletionRequest
    request_id: str
    timestamp: float
    future: asyncio.Future  # asyncio.Future for response


class HFServer:
    """Encapsulates HF model, tokenizer, and request queues."""

    def __init__(self):
        print("Initializing tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        print("Initializing HF model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=DTYPE,
            device_map="auto",
        )
        self.model.eval()

        # Load LoRA adapters (optional)
        self.adapter_name_map: dict[str, str] = {}
        if load_loras:
            print(f"Loading LoRA adapters: {load_loras}")
            for _, lora_id in enumerate(load_loras, 1):
                adapter_name = lora_id
                print(f"Loading LoRA adapter: {lora_id} as {adapter_name}")
                self.model.load_adapter(
                    lora_id,
                    adapter_name=adapter_name,
                    is_trainable=False,
                    low_cpu_mem_usage=True,
                )
                # Map requested model name → adapter name; here both are the same for simplicity
                self.adapter_name_map[lora_id] = adapter_name
                self.adapter_name_map[adapter_name] = adapter_name

        self.initialized = True
        print("Server ready!")

        # Queue management - separate queue per LoRA model
        self.queues: dict[str, deque[QueuedRequest]] = defaultdict(deque)
        self.processing_lock = asyncio.Lock()  # Ensure synchronous generation

    def get_model_key(self, model_name: str) -> str:
        """Get the key for queue management based on model name."""
        return model_name

    async def add_to_queue(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """Add request to appropriate queue and handle processing."""
        model_key = self.get_model_key(request.model)
        request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

        # Create future for async response
        future = asyncio.Future()

        queued_request = QueuedRequest(
            request=request, request_id=request_id, timestamp=time.time(), future=future
        )

        # Add to queue
        self.queues[model_key].append(queued_request)

        # Wait for response
        return await future

    async def process_queue(self, model_key: str):
        """Process up to MAX_PARALLEL_REQUESTS from the queue for a specific model."""
        async with self.processing_lock:  # Ensure only one batch processes at a time
            if not self.queues[model_key]:
                return

            # Get up to MAX_PARALLEL_REQUESTS current requests from queue
            batch_requests = []
            num_to_process = min(MAX_PARALLEL_REQUESTS, len(self.queues[model_key]))
            for _ in range(num_to_process):
                batch_requests.append(self.queues[model_key].popleft())

            # Process batch synchronously
            await self._process_batch(batch_requests, model_key)

    def _should_process_now(self, model_key: str) -> bool:
        """Return True if this queue should be processed now based on size or wait time."""
        queue = self.queues[model_key]
        if not queue:
            return False
        if len(queue) >= MAX_PARALLEL_REQUESTS:
            return True
        oldest_timestamp = queue[0].timestamp
        return (time.time() - oldest_timestamp) >= GENERATE_WAIT_SECONDS

    async def scheduler_loop(self) -> None:
        """Background loop that checks queues and processes them when ready."""
        while True:
            for model_key in list(self.queues.keys()):
                if self._should_process_now(model_key):
                    await self.process_queue(model_key)

            await asyncio.sleep(0.05)

    async def _process_batch(self, batch_requests: list[QueuedRequest], model_key: str):
        try:
            # Prepare batch data per request
            token_lists = []

            # Determine adapter for this queue (all requests share same model name)
            model_name = batch_requests[0].request.model
            if model_name == MODEL_NAME:
                # Base model
                self.model.set_adapter(None)  # type: ignore[attr-defined]
            else:
                if model_name in self.adapter_name_map:
                    active_adapter_name = self.adapter_name_map[model_name]
                    self.model.set_adapter(active_adapter_name)  # type: ignore[attr-defined]
                else:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Model {model_name} not found in loaded LoRAs",
                    )

            # Process each request in the batch
            for queued_request in batch_requests:
                request = queued_request.request

                # Format prompt
                formatted_prompt = self.tokenizer.apply_chat_template(
                    [msg.dict() for msg in request.messages],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=request.enable_thinking,
                )

                # Tokenize
                tokenized = self.tokenizer(
                    formatted_prompt, return_tensors="pt", add_special_tokens=False
                )
                token_list = tokenized["input_ids"].tolist()[0]
                token_lists.append(token_list)

            # Build common padded tensors (left pad)
            B = len(batch_requests)
            lengths = [len(tl) for tl in token_lists]
            max_len = max(lengths) if lengths else 0
            pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            input_ids = torch.full(
                (B, max_len), fill_value=pad_id, dtype=torch.long, device=DEVICE
            )
            attention_mask = torch.zeros((B, max_len), dtype=torch.bool, device=DEVICE)

            for b, tl in enumerate(token_lists):
                L = len(tl)
                if L:
                    input_ids[b, -L:] = torch.tensor(
                        tl, dtype=torch.long, device=DEVICE
                    )
                    attention_mask[b, -L:] = True

            temperature = float(batch_requests[0].request.temperature or 0.0)
            max_tokens = int(batch_requests[0].request.max_tokens)
            do_sample = temperature > 0.0

            gen_kwargs = {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "do_sample": do_sample,
                "pad_token_id": self.tokenizer.eos_token_id,
            }

            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **gen_kwargs,
                )

            # Decode
            gen_only = output_ids[:, input_ids.shape[1] :]

            # Build responses
            for i, queued_request in enumerate(batch_requests):
                try:
                    generated_text = self.tokenizer.decode(
                        gen_only[i], skip_special_tokens=True
                    )
                    response = ChatCompletionResponse(
                        id=queued_request.request_id,
                        created=int(time.time()),
                        model=queued_request.request.model or MODEL_NAME,
                        choices=[
                            Choice(
                                index=0,
                                message=Message(
                                    role="assistant", content=generated_text
                                ),
                                finish_reason="stop",
                            )
                        ],
                        usage=Usage(
                            prompt_tokens=lengths[i],
                            completion_tokens=len(
                                self.tokenizer.encode(
                                    generated_text, add_special_tokens=False
                                )
                            ),
                            total_tokens=lengths[i]
                            + len(
                                self.tokenizer.encode(
                                    generated_text, add_special_tokens=False
                                )
                            ),
                        ),
                    )

                    queued_request.future.set_result(response)

                except Exception as e:
                    # Handle individual request failure
                    print(
                        f"Error processing individual request {queued_request.request_id}: {e}"
                    )
                    queued_request.future.set_exception(e)

        except Exception as e:
            # If batch processing fails completely, set exception for all unprocessed requests
            print(f"Batch processing failed: {e}")
            for queued_request in batch_requests:
                if not queued_request.future.done():
                    queued_request.future.set_exception(e)


# Create and initialize server instance immediately
server = HFServer()

app = FastAPI(title="HF Server with LoRA", version="1.0.0")


def get_server() -> HFServer:
    """Dependency to get server instance."""
    return server


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(
    request: ChatCompletionRequest, server: HFServer = Depends(get_server)
):
    """Create a chat completion using queue system."""
    print(f"Received request: {request}")

    # Add request to queue and wait for response
    return await server.add_to_queue(request)


# @app.get("/v1/chat/completions", response_model=ChatCompletionResponse)
# async def get_chat_completion(
#     request: ChatCompletionRequest, server: HFServer = Depends(get_server)
# ):
#     """Create a chat completion using queue system."""
#     print(f"Received request: {request}")

#     # Add request to queue and wait for response
#     return await server.add_to_queue(request)


@app.get("/health")
async def health_check(server: HFServer = Depends(get_server)):
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": True}


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI API compatibility)."""
    return {
        "object": "list",
        "data": [
            {"id": MODEL_NAME, "object": "model", "owned_by": "local", "permission": []}
        ],
    }


@app.get("/queue/status")
async def queue_status(server: HFServer = Depends(get_server)):
    """Get current queue status for all models."""
    status = {}
    for model_key, queue in server.queues.items():
        status[model_key] = {
            "queue_length": len(queue),
            "oldest_timestamp": queue[0].timestamp if queue else None,
            "waiting_time": time.time() - queue[0].timestamp if queue else 0,
        }
    return {
        "queue_status": status,
        "max_parallel_requests": MAX_PARALLEL_REQUESTS,
        "generate_wait_seconds": GENERATE_WAIT_SECONDS,
    }


@app.on_event("startup")
async def start_scheduler():
    # Launch background scheduler loop
    asyncio.create_task(server.scheduler_loop())


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
