# LangGraph + OpenTelemetry -> Braintrust

Traces a custom LangGraph agent (with explicit nodes, edges, and tool calls) into Braintrust using OpenTelemetry (OTLP).

## The problem

When sending LangGraph OTEL spans to Braintrust without a wrapping root span,
each span appears individually in the logs table instead of being grouped into
a single trace. Braintrust requires a **root span** (a span with no parent) to
anchor a trace — child-only spans won't appear.

## The fix

Three things must be configured correctly:

1. **Set up the OTEL `TracerProvider` before importing LangChain/LangGraph.**
   The provider is configured with an `OTLPSpanExporter` pointing at
   `https://api.braintrust.dev/otel/v1/traces`, with `Authorization` and
   `x-bt-parent` headers.

2. **Enable LangSmith's OTEL-only mode.** Three env vars are required:
   - `LANGSMITH_TRACING=true` — activates LangChain's tracing callbacks
   - `LANGSMITH_OTEL_ENABLED=true` — routes those callbacks through OTEL
   - `LANGSMITH_OTEL_ONLY=true` — prevents LangSmith from also POSTing
     to its own API (without this you'll see 401 errors)

3. **Wrap each agent invocation in a root span.** All LangGraph spans
   automatically become children of this root span, producing a single
   grouped trace in the Braintrust UI.

## Setup

```bash
uv sync
```

Create a `.env` file with your API keys (see `.env.example`):

```
OPENAI_API_KEY=sk-...
BRAINTRUST_API_KEY=sk-...
```

## Run

```bash
uv run app.py
```

Open the **LangGraph-OTEL-Example** project in Braintrust to see the trace.
