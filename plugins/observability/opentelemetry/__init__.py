"""opentelemetry — Hermes plugin for OpenTelemetry + OpenInference tracing.

Exports Hermes turns, LLM calls, and tool executions via OTLP to the local
collector (default ``127.0.0.1:4317``). Replaces the legacy Langfuse SDK
plugin with standard OpenTelemetry semantics.

Enable via ``hermes plugins enable observability/opentelemetry`` or set
``plugins.enabled`` in config.yaml.

Environment:
  OTEL_EXPORTER_OTLP_ENDPOINT - OTLP endpoint (default: 127.0.0.1:4317)
  HERMES_OTEL_ENABLED - set to ``false`` to disable tracing
  HERMES_PROFILE - profile name attached as ``hermes.profile``
"""
from __future__ import annotations

try:
    from .core import (
        on_api_request_error,
        on_post_llm_call,
        on_post_tool_call,
        on_pre_llm_call,
        on_pre_llm_request,
        on_pre_tool_call,
        on_session_end,
        on_session_finalize,
        register,
    )
except ImportError:
    from hermes.plugins.opentelemetry.core import (
        on_api_request_error,
        on_post_llm_call,
        on_post_tool_call,
        on_pre_llm_call,
        on_pre_llm_request,
        on_pre_tool_call,
        on_session_end,
        on_session_finalize,
        register,
    )

__all__ = [
    "on_api_request_error",
    "on_post_llm_call",
    "on_post_tool_call",
    "on_pre_llm_call",
    "on_pre_llm_request",
    "on_pre_tool_call",
    "on_session_end",
    "on_session_finalize",
    "register",
]
