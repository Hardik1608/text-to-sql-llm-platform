"""
Reference OpenAI-compatible LLM server on Hugging Face transformers + PEFT.

vLLM is the production server (continuous batching, PagedAttention, LoRA hot-swap), but it
runs on Linux / WSL2 only. This server exposes the same API subset the platform relies on, so
the benchmark can run on a native-Windows GPU box straight after training:

    POST /v1/chat/completions   per-request LoRA selection via the `model` field
    GET  /v1/models             base model + loaded adapters
    POST /v1/load_lora_adapter  runtime adapter loading (vLLM-compatible payload)
    POST /v1/unload_lora_adapter

One request at a time, no batching — an evaluation tool, not a production server.

Usage (GPU box, training environment):
    python -m src.serving.hf_server --adapter spider-v1=adapters/spider-v1 --port 8001
    # then, in another shell:
    set T2S_LLM_BASE_URL=http://localhost:8001/v1
    python -m src.evaluation.compare_approaches --split dev --adapter spider-v1
"""

from __future__ import annotations

import argparse
import asyncio
import threading
import time
import uuid

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class ChatRequest(BaseModel):
    model: str
    messages: list[dict]
    max_tokens: int = 256
    temperature: float = 0.0
    stop: list[str] | None = None


class LoadAdapterRequest(BaseModel):
    lora_name: str
    lora_path: str


class UnloadAdapterRequest(BaseModel):
    lora_name: str


class HFBackend:
    def __init__(
        self, base_model: str, adapters: dict[str, str] | None = None, load_in_4bit: bool = False
    ):
        self.base_model = base_model
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        cuda = torch.cuda.is_available()
        dtype = (
            (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
            if cuda
            else torch.float32
        )
        kwargs: dict = {"torch_dtype": dtype}
        if load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            )
            kwargs["device_map"] = {"": 0}
        self.model = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)
        if not load_in_4bit and cuda:
            self.model = self.model.to("cuda")
        self.model.eval()
        self.adapters: set[str] = set()
        self._lock = threading.Lock()  # adapter selection is model-global state
        for name, path in (adapters or {}).items():
            self.load_adapter(name, path)

    def load_adapter(self, name: str, path: str) -> None:
        from peft import PeftModel

        with self._lock:
            if isinstance(self.model, PeftModel):
                self.model.load_adapter(path, adapter_name=name)
            else:
                self.model = PeftModel.from_pretrained(self.model, path, adapter_name=name)
            self.model.eval()
            self.adapters.add(name)

    def unload_adapter(self, name: str) -> None:
        with self._lock:
            if name not in self.adapters:
                raise KeyError(name)
            self.model.delete_adapter(name)
            self.adapters.discard(name)

    def chat(self, req: ChatRequest) -> dict:
        if req.model != self.base_model and req.model not in self.adapters:
            raise KeyError(req.model)
        inputs = self.tokenizer.apply_chat_template(
            req.messages, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to(self.model.device)
        gen_kwargs: dict = {
            "max_new_tokens": req.max_tokens,
            "do_sample": req.temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        }
        if req.temperature > 0:
            gen_kwargs["temperature"] = req.temperature
        with self._lock, torch.inference_mode():
            if req.model in self.adapters:
                self.model.set_adapter(req.model)
                output = self.model.generate(**inputs, **gen_kwargs)
            elif self.adapters:  # base model requested while adapters are attached
                with self.model.disable_adapter():
                    output = self.model.generate(**inputs, **gen_kwargs)
            else:
                output = self.model.generate(**inputs, **gen_kwargs)

        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = output[0, prompt_len:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        finish = "length" if len(new_tokens) >= req.max_tokens else "stop"
        for stop in req.stop or []:
            if stop in text:
                text, finish = text[: text.index(stop)], "stop"
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": finish,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_len,
                "completion_tokens": len(new_tokens),
                "total_tokens": prompt_len + len(new_tokens),
            },
        }


def create_app(backend: HFBackend) -> FastAPI:
    app = FastAPI(title="HF reference LLM server", version="0.1.0")

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models() -> dict:
        data = [{"id": backend.base_model, "object": "model"}]
        data += [
            {"id": a, "object": "model", "parent": backend.base_model}
            for a in sorted(backend.adapters)
        ]
        return {"object": "list", "data": data}

    @app.post("/v1/chat/completions")
    async def chat(req: ChatRequest) -> dict:
        try:
            return await asyncio.to_thread(backend.chat, req)
        except KeyError:
            raise HTTPException(404, f"The model `{req.model}` does not exist.") from None

    @app.post("/v1/load_lora_adapter")
    async def load(req: LoadAdapterRequest) -> str:
        await asyncio.to_thread(backend.load_adapter, req.lora_name, req.lora_path)
        return f"Success: LoRA adapter '{req.lora_name}' added successfully."

    @app.post("/v1/unload_lora_adapter")
    async def unload(req: UnloadAdapterRequest) -> str:
        try:
            await asyncio.to_thread(backend.unload_adapter, req.lora_name)
        except KeyError:
            raise HTTPException(404, f"Adapter '{req.lora_name}' is not loaded.") from None
        return f"Success: LoRA adapter '{req.lora_name}' removed successfully."

    return app


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OpenAI-compatible server on transformers + PEFT")
    p.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--adapter", action="append", default=[], help="name=path (repeatable)")
    p.add_argument("--load-in-4bit", action="store_true", help="NF4 base weights (saves VRAM)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8001)
    return p.parse_args(argv)


if __name__ == "__main__":
    import uvicorn

    args = parse_args()
    adapters = dict(item.split("=", 1) for item in args.adapter)
    uvicorn.run(
        create_app(HFBackend(args.base_model, adapters, args.load_in_4bit)),
        host=args.host,
        port=args.port,
    )
