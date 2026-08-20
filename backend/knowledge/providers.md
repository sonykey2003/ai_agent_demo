# Model providers

The demo supports OpenAI, NVIDIA NIM, and local Ollama through OpenAI-compatible
chat APIs. Local Qwen uses the `qwen2.5:0.5b` model and local Gemma uses
`gemma4:latest`. Ollama serves the OpenAI-compatible chat API under `/v1`, while
the native Ollama endpoint is used by the `nomic-embed-text` embedding model.