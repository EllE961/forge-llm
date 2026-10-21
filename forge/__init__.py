"""Forge: a Llama-style language model built from first principles.

Modules
-------
tokenizer   byte-level BPE (trainer, encoder, tiktoken interop)
data        dataset download, parallel tokenization, memmap batch sampling
model       decoder-only transformer (RMSNorm, RoPE, GQA, SwiGLU, KV cache)
train       training loop (AdamW, warmup+cosine, bf16, grad accumulation, resume)
generate    sampling (temperature, top-k, top-p, repetition penalty), streaming
speculative speculative decoding with a draft model (exact rejection sampling)
lora        low-rank adapters (inject, freeze, merge/unmerge, adapter-only I/O)
server      FastAPI playground backend
"""

__version__ = "0.1.0"
