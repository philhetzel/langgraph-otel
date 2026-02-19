"""
LangGraph + OpenTelemetry -> Braintrust Tracing Example

Traces a custom LangGraph agent (with explicit nodes, edges, and tool calls)
into Braintrust using the OTLP exporter.

The key to correct trace grouping is:
  1. Set up the OTEL TracerProvider BEFORE any LangChain imports
  2. Enable LangSmith's OTEL mode so LangChain emits OTEL spans
  3. Wrap agent invocations in a root span so Braintrust groups all
     child spans into a single trace
"""

import os

from dotenv import load_dotenv

load_dotenv()

# ── Step 1: Configure OTEL TracerProvider (must happen before LangChain imports) ──

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Map langsmith.span.kind -> braintrust span type.
# Without this, Braintrust classifies every span with gen_ai.* attributes as "llm".
LANGSMITH_KIND_TO_BT_TYPE = {
    "llm": "llm",
    "chain": "function",
    "tool": "tool",
    "retriever": "function",
    "prompt": "function",
}


class BraintrustSpanTypeProcessor(SpanProcessor):
    """Sets braintrust.span_attributes.type based on langsmith.span.kind."""

    def on_end(self, span: ReadableSpan) -> None:
        ls_kind = None
        for attr_key in span.attributes or {}:
            if attr_key == "langsmith.span.kind":
                ls_kind = span.attributes[attr_key]
                break

        if ls_kind and ls_kind in LANGSMITH_KIND_TO_BT_TYPE:
            bt_type = LANGSMITH_KIND_TO_BT_TYPE[ls_kind]
            # ReadableSpan.attributes is a MappingProxy; mutate the underlying dict
            span._attributes["braintrust.span_attributes.type"] = bt_type


bt_api_key = os.environ["BRAINTRUST_API_KEY"]
bt_project = os.environ.get("BRAINTRUST_PROJECT_NAME", "LangGraph-OTEL-Example")

provider = TracerProvider(
    resource=Resource.create({"service.name": "langgraph-otel-example"})
)
provider.add_span_processor(BraintrustSpanTypeProcessor())
provider.add_span_processor(
    BatchSpanProcessor(
        OTLPSpanExporter(
            endpoint="https://api.braintrust.dev/otel/v1/traces",
            headers={
                "Authorization": f"Bearer {bt_api_key}",
                "x-bt-parent": f"project_name:{bt_project}",
            },
        )
    )
)
trace.set_tracer_provider(provider)

# ── Step 2: Enable LangSmith OTEL-only mode ──
# LANGSMITH_TRACING + OTEL_ENABLED makes LangChain emit OTEL spans.
# OTEL_ONLY prevents LangSmith from also POSTing to its own API.
os.environ.setdefault("LANGSMITH_TRACING", "true")
os.environ.setdefault("LANGSMITH_OTEL_ENABLED", "true")
os.environ.setdefault("LANGSMITH_OTEL_ONLY", "true")

# ── Step 3: Build a custom LangGraph agent with explicit nodes and edges ──

from typing import Literal

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import MessagesState


# -- Tools --


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    data = {
        "san francisco": "Foggy, 58°F",
        "new york": "Sunny, 72°F",
        "london": "Rainy, 55°F",
        "tokyo": "Clear, 68°F",
    }
    return data.get(city.lower(), f"No weather data for {city}")


@tool
def get_population(city: str) -> str:
    """Get the approximate population of a city."""
    data = {
        "san francisco": "873,965",
        "new york": "8,336,817",
        "london": "8,982,000",
        "tokyo": "13,960,000",
    }
    return data.get(city.lower(), f"No population data for {city}")


tools = [get_weather, get_population]
tools_by_name = {t.name: t for t in tools}

# -- LLM with tools bound --

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)


# -- Graph nodes --


async def call_model(state: MessagesState) -> dict:
    """Call the LLM. It may return tool_calls in its response."""
    response = await llm.ainvoke(state["messages"])
    return {"messages": [response]}


async def call_tools(state: MessagesState) -> dict:
    """Execute every tool call the LLM requested."""
    last_message: AIMessage = state["messages"][-1]
    results = []
    for call in last_message.tool_calls:
        tool_fn = tools_by_name[call["name"]]
        result = await tool_fn.ainvoke(call["args"])
        results.append(
            ToolMessage(content=str(result), tool_call_id=call["id"])
        )
    return {"messages": results}


# -- Conditional edge --


def should_continue(state: MessagesState) -> Literal["tools", "end"]:
    """Route to 'tools' if the LLM made tool calls, otherwise end."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"


# -- Assemble the graph --

graph = StateGraph(MessagesState)

graph.add_node("agent", call_model)
graph.add_node("tools", call_tools)

graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
graph.add_edge("tools", "agent")

agent = graph.compile()

# ── Step 4: Run the agent inside a root span ──

tracer = trace.get_tracer("langgraph-otel-example")


async def run_agent(query: str) -> str:
    """Invoke the agent wrapped in a root OTEL span.

    The root span is critical: Braintrust only shows traces that have a root
    span (a span with no parent). All LangGraph spans become children of this
    root, producing a single grouped trace in the Braintrust UI.
    """
    with tracer.start_as_current_span("LangGraph Agent") as span:
        span.set_attribute("braintrust.span_attributes.type", "task")
        span.set_attribute("braintrust.input", query)

        result = await agent.ainvoke({"messages": [("user", query)]})
        output = result["messages"][-1].content

        span.set_attribute("braintrust.output", output)
        return output


if __name__ == "__main__":
    import asyncio

    async def main():
        query = "What's the weather and population in San Francisco and Tokyo?"
        response = await run_agent(query)
        provider.force_flush()
        print(response)

    asyncio.run(main())
