# Model providers

The demo supports OpenAI, OpenRouter, and local Ollama through
OpenAI-compatible chat APIs. The local model is `qwen2.5:0.5b`, and OpenRouter
serves the free-tier `nvidia/nemotron-3-ultra-550b-a55b-20260604:free` model.
Ollama serves the OpenAI-compatible chat API under `/v1`, while the native
Ollama endpoint is used by the `nomic-embed-text` embedding model.