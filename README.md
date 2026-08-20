# AI Agent Demo

Vendor-neutral LangGraph chat demo instrumented with OpenTelemetry GenAI
semantic conventions. The app exports OTLP to a local Collector, which routes
traces to Galileo or another configured backend.

The model dropdown includes OpenAI, NVIDIA NIM, local Qwen, and local Gemma 4.
Install the local models before selecting them:

```bash
ollama pull qwen2.5:0.5b
ollama pull gemma4:latest
ollama pull nomic-embed-text
```

The included Kubernetes Ollama manifest only pre-pulls the small Qwen model and
uses a 5 GiB model volume. To select Gemma against that deployment, provide an
Ollama endpoint with `gemma4:latest` already installed and sufficient storage
and memory.

## Vector RAG

The agent grounds knowledge questions with PostgreSQL/pgvector and Ollama's
`nomic-embed-text` embedding model. Retrieval runs in the LangGraph pre-model
hook, so relevant passages are added before the selected chat model runs. The
retriever span includes the query, document IDs, content, metadata, and scores
needed by Galileo RAG evaluations.

The chat endpoint and embedding endpoint are intentionally separate:

- `OLLAMA_BASE_URL=http://localhost:11434/v1` uses the OpenAI-compatible chat API.
- `OLLAMA_EMBEDDING_BASE_URL=http://localhost:11434` uses Ollama's native API.

RAG ingestion is a critical backend startup step. When `RAG_ENABLED=true`, the
backend rebuilds the collection from the bundled knowledge files before FastAPI
becomes ready. Startup fails if PostgreSQL, Ollama embeddings, or ingestion is
unavailable, so the app cannot silently run without grounding.

For a host-run backend, start pgvector and then start FastAPI normally:

```bash
cp .env.example .env  # first run only; then add provider/Galileo credentials
ollama pull nomic-embed-text
docker compose up -d postgres
TELEMETRY_ENABLED=false .venv/bin/python -m uvicorn app.main:app \
	--app-dir backend --host 127.0.0.1 --port 8001
```

The app's database is exposed on `localhost:5433`, avoiding the golden demo's
PostgreSQL port `5432`. Add or edit Markdown/text files under
`backend/knowledge`, then restart the backend to rebuild the collection. Chunk
IDs are deterministic and citations use readable IDs such as `[otel-0]`.

For a complete Compose deployment, first make sure Ollama is running on the
host and already contains the required models. Compose connects to the host at
`host.docker.internal` and does not create another Ollama model volume:

```bash
ollama list  # should include qwen2.5:0.5b and nomic-embed-text
./scripts/up.sh
```

The wrapper starts `ollama serve` in the background when the host API is not
already running, waits for it to become ready, and then starts Compose. Compose
starts PostgreSQL and the Collector, rebuilds the vector collection during
backend startup, and serves the application at `http://localhost:8800`. A
one-shot `ollama-check` service independently verifies host connectivity and the
`nomic-embed-text` model before allowing the backend to start.

You can still run `docker compose up -d --build` directly when Ollama is already
running.

## Demo prompts

Use these for a consistent live demo. Pick the matching **Domain** in the UI
dropdown first — each domain grounds retrieval on its own pgvector collection
(`ai_agent_demo_<domain>`). Domain questions fire the retrieval pre-hook (a
`retrieve knowledge_base` retriever span), not a tool call. Arithmetic questions
fire the only real LLM tool, `calculator`.

### Knowledge / RAG prompts (per domain)

**Platform** (`ai_agent_demo_knowledge`, grounded in `backend/knowledge`):

- What OpenTelemetry GenAI semantic conventions does this app emit?
- How does the retriever span get evaluated by Galileo RAG metrics?
- Which LLM providers are supported and how do I switch between them?

**Bank** (credit-card call center):

- How can I dispute fraudulent charges on my card?
- How do I check my balance and balance due?
- How do I increase my credit limit?
- Does my online password expire?

**Healthcare** (drug reference):

- Tell me about Lisinopril — dosage and common side effects.
- What are the contraindications for Metformin?
- What monitoring is required when taking Atorvastatin?

**Insurance** (auto claims):

- How will my claim affect my premium?
- My vehicle was determined a total loss — what does that mean?
- Who pays my deductible if I'm not at fault?
- Will you pay for a rental while my car is in the shop?

**Restaurant** (operations / shift manager):

- What is the closing checklist for the kitchen?
- What is the morning prep sequence for the line?
- What proteins need to be prepped for a typical Saturday?
- How do I give positive recognition to an employee?

### Calculator tool prompts (any domain)

`calculator` is the only tool the model can call. Arithmetic prompts trigger it:

- What is 17 * 23 + 4?
- Calculate 15% of 8000.
- What's (1250 * 3) + 499?
- Compute 2 ** 10 / 4.

Tool-calling reliability depends on the model: OpenAI and NVIDIA NIM call the
tool consistently; the tiny local `qwen2.5:0.5b` often answers arithmetic inline
without a tool call. Select OpenAI or NIM to demo a guaranteed `calculator` span.

The Kubernetes backend uses the same integrated startup ingestion. Build the
backend image from the repository root and deploy PostgreSQL, Ollama, and the
backend; the backend pod becomes ready only after indexing succeeds:

```bash
docker build -t agent-chat-demo:latest -f backend/Dockerfile .
kubectl apply -f deploy/k8s/postgres.yaml -f deploy/k8s/ollama.yaml
kubectl -n agent-demo rollout status deployment/postgres
kubectl -n agent-demo rollout status deployment/ollama
kubectl apply -f deploy/k8s/backend.yaml
```

Create `deploy/k8s/secrets.yaml` from the example before deployment. Restart the
backend deployment after changing bundled knowledge to rebuild the collection.

## Trace model

Every `/chat` turn creates one explicit `invoke_agent Agent Chat` root span. The
OpenLLMetry LangGraph, LLM, tool, and manually-created retriever spans are nested
under that root. Input-blocked requests still produce a trace containing the
turn root and an input-guardrail child.

The browser generates one conversation ID when the page loads and sends it with
every turn. The backend records it as `gen_ai.conversation.id` and `session.id`,
so separate turn traces can be grouped into one conversation. Non-browser API
clients may omit it and receive a generated ID per request.

FastAPI auto-instrumentation is intentionally disabled. The agent executes in an
SSE response generator after a conventional HTTP server span would close, which
would otherwise create a detached, empty HTTP trace.

## Privacy boundary

`OTEL_REDACT_PII=true` wraps the OTLP trace exporter and masks known email,
phone, and US SSN patterns in all string-valued span attributes and span-event
attributes. This includes JSON strings stored in `gen_ai.input.messages`,
`gen_ai.output.messages`, `input.value`, and `output.value`. Non-PII content is
retained so content-based Galileo metrics can still run.

Redaction occurs in the application before the first OTLP hop. It does not
redact stdout application logs. Production deployments should avoid logging
sensitive values or add a separate logging filter/pipeline.

## Token usage

`LLM_STREAM_USAGE=true` requests usage metadata in the final streaming response
chunk. OpenLLMetry maps compatible responses to `gen_ai.usage.input_tokens` and
`gen_ai.usage.output_tokens`. Set it to `false` if an OpenAI-compatible endpoint
rejects `stream_options.include_usage`.

## Evaluation traces

The generic evaluator exports one trace per dataset row through the same OTel
Collector pipeline. Each trace records the dataset, case ID, expected strings,
actual output, and pass/fail result.

```bash
python eval/run_eval.py --provider local --dataset eval/datasets/smoke.jsonl
```

Telemetry is force-flushed when either FastAPI or the evaluator shuts down.
Native Galileo sessions are deliberately not runtime dependencies; Galileo
remains a pluggable Collector destination. Galileo Experiments are available only
as an offline `eval/` admin helper (see
[Pre-prod experiments](#pre-prod-experiments-llm--guardrails), below) — the
vendor-neutral app never imports the Galileo SDK for tracing.

## Agent Control (optional runtime guardrail)

The app can optionally run [Galileo Agent Control](https://docs.galileo.ai/how-to-guides/agent-control/initialize-and-configure-agent-control)
as a centralized runtime guardrail. It is **off by default** and fully guarded,
so the vendor-neutral default is unchanged. When enabled, every `/chat` turn runs
two controlled steps — `user_input` (evaluated before the agent; a *deny* control
blocks the turn) and `assistant_output` (evaluated after; a match withholds the
answer). Control spans are bridged back into the Galileo log stream.

Enable it by installing the Agent Control SDK (ships with your Galileo
deployment, not on public PyPI) and setting these in `.env`:

```bash
pip install "agent-control-sdk[galileo]>=7.10.0" "agent-control-evaluator-galileo>=7.10.0"
# .env
AGENT_CONTROL_ENABLED=true
AGENT_CONTROL_URL=https://agent-control.multitenant.galileocloud.io
```

Then create a control in the console bound to the same log stream
(`GALILEO_LOG_STREAM`) and to the step names `user_input` / `assistant_output`,
and rebuild the backend (`docker compose up -d --build backend`). At startup the
log shows `Agent Control initialised (agent=..., log_stream_id=...)`. If the SDK
or `AGENT_CONTROL_URL` is missing, controls stay off and the app runs normally.

A standalone, non-embedded version also lives in `eval/agent_control_demo.py`.

## Domain datasets (Galileo eval + coherence baselining)

`domains/` packages evaluation datasets like the golden demo's `domains/` layout.
`platform/` is the demo's own subject matter (grounded in `backend/knowledge`),
and `bank/`, `healthcare/`, `insurance/`, and `restaurant/` are reused from the
golden demo. Each domain has `config.yaml`, `system_prompt.json`, `docs/qa.csv`,
and generated Galileo upload artifacts.

Each domain is also selectable in the chat UI (the **Domain** dropdown). Selecting
a domain switches the agent's system prompt and grounds retrieval on that domain's
own pgvector collection (`ai_agent_demo_<domain>`, or `ai_agent_demo_knowledge`
for `platform`). Every domain is indexed into its collection at backend startup.

Regenerate the datasets after editing any `docs/qa.csv` (each answer is checked
for grounding — `platform` against `backend/knowledge`, reused domains against
their own `docs/`):

```bash
python eval/build_dataset.py --all --check
```

For each domain this writes `dataset.csv` (`input,output` for the Galileo UI
upload) and `dataset.jsonl` (`input`/`output`/`metadata` for the SDK).

Upload a domain to Galileo, then optionally run a context-adherence / coherence
baseline experiment:

```bash
pip install -r eval/requirements-galileo.txt
python eval/upload_galileo_dataset.py --domain bank --preview   # inspect first
python eval/upload_galileo_dataset.py --domain bank             # upload dataset
python eval/upload_galileo_dataset.py --domain bank --baseline  # upload + score
```

The baseline scores `context_adherence`, `ground_truth_adherence`,
`chunk_attribution_utilization`, and `completeness` where available. Register the
NIM-judged variants first with `eval/configure_galileo_metrics.py` if you don't
have an OpenAI integration. The `--baseline` runner uses the app agent, which is
grounded on `backend/knowledge`; to baseline a reused domain against its own
knowledge, ingest that domain's docs into pgvector first
(`python -m app.rag.ingest --reset domains/<domain>/docs`).

## Pre-prod experiments (LLM & guardrails)

`eval/run_experiment.py` runs the **real** application agent over datasets
through Galileo's offline *experiment* feature, so you can validate the LLM and
the guardrails before shipping. It stays on the app's **OpenTelemetry path**:
the agent emits the same OTel spans it does in production — the `invoke_agent
Agent Chat` root, input/output `guardrail` spans, the `retrieve knowledge_base`
retriever span, the LLM span, and tool spans — and a `GalileoSpanProcessor`
(with `GALILEO_LOGGING_DISABLED=true`) routes them to the experiment. There is no
native `@log` decorator or hand-built span; the experiment traces are identical
to production traces.

The central control is a `--guardrails` **toggle** — `on` (production guardrails
wrap the agent, emitting guardrail spans), `off` (raw baseline), or `both` (runs
the dataset twice as a matched `…-guardrails-on` / `-off` pair in one experiment
group, so you can open Galileo → Experiments → **Compare Experiments** and see
the guardrail impact side by side).

Four suites (`--suite`) target different pre-prod questions. The two priority
use cases:

- **`rag` — RAG Retrieval Quality Control.** Runs the domain Q&A dataset through
  the full agent; the real `retrieve knowledge_base` retriever span (retrieved
  chunks with content, source, and score) is scored by Galileo's
  retrieval-quality metrics: `context_adherence`, `context_relevance`,
  `chunk_attribution_utilization`, `chunk_relevance`, `completeness`,
  `ground_truth_adherence`.
- **`multiturn` — Multi-Turn Conversational Guardrails.** Replays built-in
  multi-turn conversations, enforcing the guardrails on *every* turn — a
  jailbreak or PII request placed on a *later* turn must still be caught. Turns
  share one `session.id`, so Galileo groups each conversation into a **session**.
  A blocked turn still emits its input-`guardrail` span (the block is visible
  even though the agent never runs). Prints a per-run summary (`input-blocked` /
  `output-pii-redacted` turn counts).

The two supporting suites: `quality` (single-turn domain Q&A answer quality) and
`adversarial` (single-turn jailbreak + PII probes).

Every run also computes deterministic **local** metrics that need no LLM-judge
integration — `Input Blocked (guardrail)` and `Output PII Present` — so guardrail
health is measurable in any environment. The suite's Galileo out-of-the-box judge
metrics are added on top unless you pass `--local-metrics-only`.

The local stack (postgres + ollama) must be up for the agent to answer, e.g.
`./scripts/up.sh`.

```bash
pip install -r eval/requirements-galileo.txt
python eval/run_experiment.py --suite rag --domain bank --guardrails on              # retrieval QC
python eval/run_experiment.py --suite multiturn --domain bank --guardrails both      # multi-turn guardrails
python eval/run_experiment.py --suite adversarial --guardrails both --local-metrics-only
python eval/run_experiment.py --suite multiturn --domain bank --preview              # inspect conversations
```