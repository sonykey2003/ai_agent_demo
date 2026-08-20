#!/usr/bin/env bash

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
ollama_url="${OLLAMA_HOST_URL:-http://127.0.0.1:11434}"
embedding_model="${OLLAMA_EMBEDDING_MODEL:-nomic-embed-text}"
ollama_log="${TMPDIR:-/tmp}/ai-agent-demo-ollama.log"

ollama_ready() {
    curl --fail --silent --max-time 2 "${ollama_url}/api/tags" >/dev/null 2>&1
}

if ! ollama_ready; then
    if ! command -v ollama >/dev/null 2>&1; then
        echo >&2 "Ollama is not installed or is not on PATH."
        echo >&2 "Install it from https://ollama.com/download"
        exit 1
    fi

    echo "Ollama is not running; starting 'ollama serve'..."
    nohup ollama serve >"${ollama_log}" 2>&1 </dev/null &

    attempt=1
    while ! ollama_ready; do
        if (( attempt >= 30 )); then
            echo >&2 "Ollama did not become ready within 30 seconds."
            echo >&2 "Log: ${ollama_log}"
            tail -n 20 "${ollama_log}" >&2
            exit 1
        fi
        sleep 1
        ((attempt += 1))
    done
    echo "Ollama is ready. Log: ${ollama_log}"
else
    echo "Ollama is already running."
fi

models="$(curl --fail --silent --show-error --max-time 5 "${ollama_url}/api/tags")" || {
    echo >&2 "Could not read models from ${ollama_url}."
    exit 1
}
if ! grep -q "${embedding_model}" <<<"${models}"; then
    echo >&2 "Required embedding model '${embedding_model}' is missing."
    echo >&2 "Install it with: ollama pull ${embedding_model}"
    exit 1
fi

cd "${repo_root}" || exit 1
docker compose up -d --build "$@"