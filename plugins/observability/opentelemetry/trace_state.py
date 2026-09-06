"""In-memory turn trace state for the OpenTelemetry plugin."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

try:
    from .serialization import OpenInferenceSpanKindValues, SpanAttributes, mark_span_error
except ImportError:
    from hermes.plugins.opentelemetry.serialization import (
        OpenInferenceSpanKindValues,
        SpanAttributes,
        mark_span_error,
    )

from opentelemetry import trace
from opentelemetry.trace import Span, Tracer

def _force_flush_traces() -> None:
    """Push batched spans to the OTLP exporter immediately."""
    try:
        provider = trace.get_tracer_provider()
        force_flush = getattr(provider, "force_flush", None)
        if callable(force_flush):
            force_flush(timeout_millis=5000)
    except Exception:
        pass


@dataclass
class TraceState:
    root_span: Span
    session_id: str = ""
    generations: dict[str, Span] = field(default_factory=dict)
    tools: dict[str, Span] = field(default_factory=dict)
    pending_tools_by_name: dict[str, list[Span]] = field(default_factory=dict)
    last_updated_at: float = field(default_factory=time.time)


_STATE_LOCK = threading.RLock()
_TRACE_STATE: dict[str, TraceState] = {}
_MAX_TRACE_STATE = 256


def clear_trace_state() -> None:
    with _STATE_LOCK:
        _TRACE_STATE.clear()


def _trace_key(
    task_id: str,
    session_id: str,
    *,
    turn_id: str = "",
    api_request_id: str = "",
) -> str:
    if turn_id:
        return f"turn:{turn_id}"
    if api_request_id:
        return f"request:{api_request_id}"
    if session_id:
        return f"session:{session_id}"
    if task_id:
        return f"task:{task_id}"
    return "anonymous"


def _request_key(api_call_count: Any) -> str:
    return str(api_call_count or 0)


def _session_attributes(
    *,
    session_id: str,
    task_id: str,
    platform: str,
    profile: str = "",
    user_id: str = "",
) -> dict[str, Any]:
    try:
        from .core import _env
    except ImportError:
        from hermes.plugins.opentelemetry.core import _env

    attrs: dict[str, Any] = {
        SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.AGENT.value,
        SpanAttributes.SESSION_ID: session_id or task_id,
        "hermes.task_id": task_id,
        "hermes.platform": platform,
    }
    if user_id:
        attrs["user.id"] = user_id
    resolved_profile = profile or _env("HERMES_PROFILE")
    if resolved_profile:
        attrs["hermes.profile"] = resolved_profile
    return attrs


def _evict_stale_locked() -> None:
    if len(_TRACE_STATE) <= _MAX_TRACE_STATE:
        return
    victims = sorted(_TRACE_STATE.items(), key=lambda item: item[1].last_updated_at)
    for key, state in victims[: len(_TRACE_STATE) - _MAX_TRACE_STATE]:
        try:
            state.root_span.end()
        except Exception:
            pass
        _TRACE_STATE.pop(key, None)


def _start_root_span(
    tracer: Tracer,
    *,
    session_id: str,
    task_id: str,
    platform: str,
    profile: str = "",
    user_id: str = "",
) -> Span:
    name = f"Hermes turn: {session_id or task_id or 'unknown'}"
    span = tracer.start_span(name)
    for key, value in _session_attributes(
        session_id=session_id,
        task_id=task_id,
        platform=platform,
        profile=profile,
        user_id=user_id,
    ).items():
        span.set_attribute(key, value)
    return span


def _child_context(parent: Span) -> trace.Context:
    return trace.set_span_in_context(parent)


def _finish_trace(
    task_key: str,
    *,
    status_ok: bool = True,
    error_type: str = "",
    error_message: str = "",
) -> None:
    with _STATE_LOCK:
        state = _TRACE_STATE.pop(task_key, None)
    if state is None:
        return
    for span in list(state.generations.values()):
        try:
            span.end()
        except Exception:
            pass
    for span in list(state.tools.values()):
        try:
            span.end()
        except Exception:
            pass
    for spans in state.pending_tools_by_name.values():
        for span in spans:
            try:
                span.end()
            except Exception:
                pass
    try:
        if not status_ok:
            mark_span_error(
                state.root_span,
                error_type=error_type,
                error_message=error_message,
            )
        state.root_span.end()
    except Exception:
        pass
    _force_flush_traces()


def _finish_session_traces(session_id: str, *, status_ok: bool = True) -> None:
    if not session_id:
        return
    with _STATE_LOCK:
        keys = [key for key, state in _TRACE_STATE.items() if state.session_id == session_id]
    for key in keys:
        _finish_trace(key, status_ok=status_ok)
    _force_flush_traces()
