"""Message and usage serialization for OpenTelemetry span attributes."""

from __future__ import annotations

import json
from typing import Any, Optional

try:
    from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
except ImportError:
    from enum import Enum
    class OpenInferenceSpanKindValues(str, Enum):
        AGENT = "AGENT"
        LLM = "LLM"
        TOOL = "TOOL"
        CHAIN = "CHAIN"
        RETRIEVER = "RETRIEVER"
    class SpanAttributes:
        OPENINFERENCE_SPAN_KIND = "openinference.span.kind"
        SESSION_ID = "session.id"
        USER_ID = "user.id"
        INPUT_VALUE = "input.value"
        OUTPUT_VALUE = "output.value"
        LLM_INPUT_MESSAGES = "llm.input_messages"
        LLM_OUTPUT_MESSAGES = "llm.output_messages"
        LLM_MODEL_NAME = "llm.model_name"
        LLM_PROVIDER_NAME = "llm.provider"
        LLM_PROVIDER = "llm.provider"
        LLM_COST_TOTAL = "llm.cost.total"
        LLM_TOKEN_COUNT_PROMPT = "llm.token_count.prompt"
        LLM_TOKEN_COUNT_COMPLETION = "llm.token_count.completion"
        LLM_TOKEN_COUNT_TOTAL = "llm.token_count.total"
        TOOL_NAME = "tool.name"
        TOOL_PARAMETERS = "tool.parameters"
        TOOL_OUTPUT = "tool.output"

from opentelemetry.trace import Span, Status, StatusCode

MAX_FIELD_CHARS = 12_000
TOOL_OUTPUT_ATTR = "tool.output"


def truncate(value: str, limit: int = MAX_FIELD_CHARS) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + "…"


def safe_json(value: Any) -> str:
    try:
        return truncate(json.dumps(value, default=str, ensure_ascii=False))
    except Exception:
        return truncate(str(value))


def serialize_messages(messages: Any) -> str:
    if not isinstance(messages, list):
        return safe_json(messages)
    serialized: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, dict):
            role = message.get("role")
            content = message.get("content")
        else:
            role = getattr(message, "role", None)
            content = getattr(message, "content", None)
        serialized.append({"role": role, "content": content})
    return safe_json(serialized)


def serialize_assistant_output(message: Any, *, assistant_response: Any = None) -> str:
    if message is not None:
        content = getattr(message, "content", None)
        tool_calls = getattr(message, "tool_calls", None)
        payload = {
            "role": "assistant",
            "content": content,
            "tool_calls": [
                {
                    "id": getattr(call, "id", None),
                    "name": getattr(getattr(call, "function", None), "name", None),
                }
                for call in (tool_calls or [])
            ],
        }
        return safe_json(payload)
    if assistant_response is not None:
        return safe_json({"role": "assistant", "content": assistant_response})
    return safe_json({"role": "assistant", "content": None})


def coerce_request_messages(
    *,
    request_messages: Any = None,
    messages: Any = None,
    conversation_history: Any = None,
    user_message: Any = None,
) -> Any:
    for candidate in (request_messages, messages, conversation_history):
        if isinstance(candidate, list):
            return candidate
    if user_message is not None:
        return [{"role": "user", "content": user_message}]
    return []


def usage_from_response(response: Any, *, provider: str, api_mode: str) -> tuple[dict[str, int], float | None]:
    raw_usage = getattr(response, "usage", None)
    if not raw_usage:
        return {}, None
    try:
        from agent.usage_pricing import normalize_usage

        canonical = normalize_usage(raw_usage, provider=provider, api_mode=api_mode)
        return usage_from_canonical(canonical, provider=provider, model="", base_url="")
    except Exception:
        return {}, None


def usage_from_canonical(
    canonical: Any,
    *,
    provider: str,
    model: str,
    base_url: str,
) -> tuple[dict[str, int], float | None]:
    usage_details = {
        "input": int(getattr(canonical, "input_tokens", 0) or 0),
        "output": int(getattr(canonical, "output_tokens", 0) or 0),
    }
    total = usage_details["input"] + usage_details["output"]
    usage_details["total"] = total
    cost_total: float | None = None
    try:
        from agent.usage_pricing import estimate_usage_cost

        cost = estimate_usage_cost(
            model,
            canonical,
            provider=provider,
            base_url=base_url,
            api_key="",
        )
        if cost.amount_usd is not None and float(cost.amount_usd) > 0:
            cost_total = float(cost.amount_usd)
    except Exception:
        pass
    return usage_details, cost_total


def usage_from_summary(
    usage: dict[str, Any],
    *,
    provider: str,
    model: str,
    base_url: str,
) -> tuple[dict[str, int], float | None]:
    input_tokens = int(usage.get("input_tokens", 0) or 0)
    output_tokens = int(usage.get("output_tokens", 0) or usage.get("completion_tokens", 0) or 0)
    usage_details = {
        "input": input_tokens,
        "output": output_tokens,
        "total": input_tokens + output_tokens,
    }
    try:
        from agent.usage_pricing import CanonicalUsage

        canonical = CanonicalUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=int(usage.get("cache_read_tokens", 0) or 0),
            cache_write_tokens=int(usage.get("cache_write_tokens", 0) or 0),
            reasoning_tokens=int(usage.get("reasoning_tokens", 0) or 0),
            request_count=int(usage.get("request_count", 1) or 1),
        )
        return usage_from_canonical(
            canonical,
            provider=provider,
            model=model,
            base_url=base_url,
        )
    except Exception:
        return usage_details, None


def resolve_usage(
    *,
    response: Any,
    usage: Any,
    provider: str,
    api_mode: str,
    model: str,
    base_url: str,
) -> tuple[dict[str, int], float | None]:
    if getattr(response, "usage", None) is not None:
        return usage_from_response(response, provider=provider, api_mode=api_mode)
    if isinstance(usage, dict) and usage:
        return usage_from_summary(usage, provider=provider, model=model, base_url=base_url)
    return {}, None


def set_llm_attributes(
    span: Span,
    *,
    model: str,
    provider: str,
    input_messages: Any = None,
    output_messages: str | None = None,
    usage_details: dict[str, int] | None = None,
    cost_total: float | None = None,
) -> None:
    span.set_attribute(
        SpanAttributes.OPENINFERENCE_SPAN_KIND,
        OpenInferenceSpanKindValues.LLM.value,
    )
    if model:
        span.set_attribute(SpanAttributes.LLM_MODEL_NAME, model)
    if provider:
        span.set_attribute(SpanAttributes.LLM_PROVIDER, provider)
    if input_messages is not None:
        span.set_attribute(SpanAttributes.LLM_INPUT_MESSAGES, serialize_messages(input_messages))
    if output_messages is not None:
        span.set_attribute(SpanAttributes.LLM_OUTPUT_MESSAGES, output_messages)
    usage = usage_details or {}
    prompt = usage.get("input") or usage.get("prompt") or 0
    completion = usage.get("output") or usage.get("completion") or 0
    total = usage.get("total") or (prompt + completion)
    if prompt:
        span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_PROMPT, int(prompt))
    if completion:
        span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, int(completion))
    if total:
        span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_TOTAL, int(total))
    if cost_total is not None and cost_total > 0:
        span.set_attribute(SpanAttributes.LLM_COST_TOTAL, float(cost_total))


def mark_span_error(
    span: Any,
    *,
    error_type: str = "",
    error_message: str = "",
    description: str = "",
    exc: Optional[BaseException] = None,
) -> None:
    status_description = description or error_message or error_type
    span.set_status(Status(StatusCode.ERROR, description=status_description))
    span.set_attribute("error", True)
    span.set_attribute("error.type", error_type or "Error")
    span.set_attribute("error.message", error_message or description)
    if exc is not None:
        span.record_exception(exc)
        span.set_attribute("exception.type", type(exc).__name__)
        span.set_attribute("exception.message", str(exc))


def set_tool_attributes(span: Span, *, tool_name: str, parameters: Any, output: Any = None) -> None:
    span.set_attribute(
        SpanAttributes.OPENINFERENCE_SPAN_KIND,
        OpenInferenceSpanKindValues.TOOL.value,
    )
    span.set_attribute(SpanAttributes.TOOL_NAME, tool_name)
    span.set_attribute(SpanAttributes.TOOL_PARAMETERS, safe_json(parameters))
    if output is not None:
        span.set_attribute(TOOL_OUTPUT_ATTR, safe_json(output))
