"""Hermes Agent hook handlers for OpenTelemetry tracing."""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, TypeVar

try:
    from .serialization import (
        coerce_request_messages,
        mark_span_error,
        safe_json,
        serialize_assistant_output,
        set_llm_attributes,
        set_tool_attributes,
    )
    from .trace_state import (
        TraceState,
        _STATE_LOCK,
        _TRACE_STATE,
        _child_context,
        _evict_stale_locked,
        _finish_session_traces,
        _finish_trace,
        _request_key,
        _start_root_span,
        _trace_key,
    )
except ImportError:
    from hermes.plugins.opentelemetry.serialization import (
        coerce_request_messages,
        mark_span_error,
        safe_json,
        serialize_assistant_output,
        set_llm_attributes,
        set_tool_attributes,
    )
    from hermes.plugins.opentelemetry.trace_state import (
        TraceState,
        _STATE_LOCK,
        _TRACE_STATE,
        _child_context,
        _evict_stale_locked,
        _finish_session_traces,
        _finish_trace,
        _request_key,
        _start_root_span,
        _trace_key,
    )

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def _get_tracer_instance() -> Any:
    try:
        from .core import _get_tracer
        return _get_tracer()
    except ImportError:
        from hermes.plugins.opentelemetry.core import _get_tracer
        return _get_tracer()


def _resolve_usage_helper(**kwargs: Any) -> Any:
    try:
        from .core import _resolve_usage
        return _resolve_usage(**kwargs)
    except ImportError:
        from hermes.plugins.opentelemetry.core import _resolve_usage
        return _resolve_usage(**kwargs)


def _fail_open(func: F) -> F:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.debug("OpenTelemetry plugin: %s failed open: %s", func.__name__, exc)
            return None

    return wrapper  # type: ignore[return-value]


def _tool_call_failed(*, result: Any, **kwargs: Any) -> tuple[bool, str]:
    if kwargs.get("error"):
        return True, str(kwargs["error"])
    if isinstance(result, Exception):
        return True, str(result)
    if isinstance(result, dict):
        if "error" in result:
            return True, str(result["error"])
        if result.get("status") == "error":
            return True, str(result.get("message") or result.get("error") or "tool error")
        exit_code = result.get("exit_code", result.get("returncode"))
        if exit_code is not None and exit_code != 0:
            return True, f"exit code {exit_code}"
    return False, ""


def _error_metadata(error: Any, **kwargs: Any) -> tuple[str, str]:
    err = error if error is not None else kwargs.get("exception")
    if err is None:
        err = kwargs.get("error_message")
    err_type = kwargs.get("error_type")
    if not err_type and isinstance(err, dict):
        err_type = err.get("type") or err.get("error_type")
    if not err_type and err is not None and not isinstance(err, dict):
        err_type = type(err).__name__
    if not err_type:
        err_type = "APIError"
    if isinstance(err, dict):
        err_msg = err.get("message") or err.get("error_message") or str(err)
    elif err is not None:
        err_msg = str(err)
    else:
        err_msg = str(kwargs.get("error_message", "API request failed"))
    return err_type, err_msg

@_fail_open
def on_pre_llm_call(
    *,
    task_id: str = "",
    session_id: str = "",
    platform: str = "",
    model: str = "",
    provider: str = "",
    base_url: str = "",
    api_mode: str = "",
    api_call_count: int = 0,
    messages: Any = None,
    turn_id: str = "",
    api_request_id: str = "",
    profile: str = "",
    user_id: str = "",
    **_: Any,
) -> None:
    tracer = _get_tracer_instance()
    if tracer is None:
        return
    task_key = _trace_key(task_id, session_id, turn_id=turn_id, api_request_id=api_request_id)
    with _STATE_LOCK:
        if task_key not in _TRACE_STATE:
            _TRACE_STATE[task_key] = TraceState(
                root_span=_start_root_span(
                    tracer,
                    session_id=session_id,
                    task_id=task_id,
                    platform=platform,
                    profile=profile,
                    user_id=user_id,
                ),
                session_id=session_id,
            )
            _evict_stale_locked()
        _TRACE_STATE[task_key].last_updated_at = time.time()


@_fail_open
def on_pre_llm_request(
    *,
    task_id: str = "",
    session_id: str = "",
    platform: str = "",
    model: str = "",
    provider: str = "",
    base_url: str = "",
    api_mode: str = "",
    api_call_count: int = 0,
    request_messages: Any = None,
    messages: Any = None,
    conversation_history: Any = None,
    user_message: Any = None,
    turn_id: str = "",
    api_request_id: str = "",
    profile: str = "",
    user_id: str = "",
    request: Any = None,
    **_: Any,
) -> None:
    tracer = _get_tracer_instance()
    if tracer is None:
        return
    if isinstance(request, dict):
        body = request.get("body")
        if isinstance(body, dict):
            body_model = body.get("model")
            if isinstance(body_model, str) and body_model:
                model = body_model
    input_messages = coerce_request_messages(
        request_messages=request_messages,
        messages=messages,
        conversation_history=conversation_history,
        user_message=user_message,
    )
    task_key = _trace_key(task_id, session_id, turn_id=turn_id, api_request_id=api_request_id)
    req_key = _request_key(api_call_count)
    with _STATE_LOCK:
        state = _TRACE_STATE.get(task_key)
        if state is None:
            state = TraceState(
                root_span=_start_root_span(
                    tracer,
                    session_id=session_id,
                    task_id=task_id,
                    platform=platform,
                    profile=profile,
                    user_id=user_id,
                ),
                session_id=session_id,
            )
            _TRACE_STATE[task_key] = state
            _evict_stale_locked()
        previous = state.generations.pop(req_key, None)
        if previous is not None:
            previous.end()
        llm_span = tracer.start_span(
            f"LLM call {api_call_count}",
            context=_child_context(state.root_span),
        )
        set_llm_attributes(
            llm_span,
            model=model,
            provider=provider,
            input_messages=input_messages,
        )
        state.generations[req_key] = llm_span
        state.last_updated_at = time.time()


@_fail_open
def on_post_llm_call(
    *,
    task_id: str = "",
    session_id: str = "",
    provider: str = "",
    base_url: str = "",
    api_mode: str = "",
    model: str = "",
    api_call_count: int = 0,
    assistant_message: Any = None,
    response: Any = None,
    api_duration: float = 0.0,
    finish_reason: str = "",
    usage: Any = None,
    assistant_content_chars: int = 0,
    assistant_tool_call_count: int = 0,
    assistant_response: Any = None,
    turn_id: str = "",
    api_request_id: str = "",
    response_model: Any = None,
    **_: Any,
) -> None:
    tracer = _get_tracer_instance()
    if tracer is None:
        return
    if isinstance(response_model, str) and response_model:
        model = response_model
    task_key = _trace_key(task_id, session_id, turn_id=turn_id, api_request_id=api_request_id)
    req_key = _request_key(api_call_count)
    with _STATE_LOCK:
        state = _TRACE_STATE.get(task_key)
        llm_span = state.generations.pop(req_key, None) if state else None
    if state is None or llm_span is None:
        return
    output_messages = serialize_assistant_output(
        assistant_message,
        assistant_response=assistant_response,
    )
    if assistant_message is None and assistant_response is None and assistant_content_chars:
        output_messages = safe_json(
            {"role": "assistant", "content": f"[{assistant_content_chars} chars]"}
        )
    usage_details, cost_total = _resolve_usage_helper(
        response=response,
        usage=usage,
        provider=provider,
        api_mode=api_mode,
        model=model,
        base_url=base_url,
    )
    set_llm_attributes(
        llm_span,
        model=model,
        provider=provider,
        output_messages=output_messages,
        usage_details=usage_details,
        cost_total=cost_total,
    )
    if finish_reason:
        llm_span.set_attribute("llm.response.finish_reason", finish_reason)
    if api_duration and api_duration > 0:
        llm_span.set_attribute("hermes.api_duration_s", round(api_duration, 3))
    llm_span.end()
    has_tools = bool(getattr(assistant_message, "tool_calls", None)) or assistant_tool_call_count > 0
    has_content = bool(getattr(assistant_message, "content", None)) or bool(assistant_response)
    if not has_tools and has_content:
        _finish_trace(task_key)


@_fail_open
def on_pre_tool_call(
    *,
    tool_name: str = "",
    args: Any = None,
    task_id: str = "",
    session_id: str = "",
    tool_call_id: str = "",
    turn_id: str = "",
    api_request_id: str = "",
    **_: Any,
) -> None:
    tracer = _get_tracer_instance()
    if tracer is None:
        return
    task_key = _trace_key(task_id, session_id, turn_id=turn_id, api_request_id=api_request_id)
    with _STATE_LOCK:
        state = _TRACE_STATE.get(task_key)
        if state is None:
            return
        tool_span = tracer.start_span(
            f"Tool: {tool_name}",
            context=_child_context(state.root_span),
        )
        set_tool_attributes(tool_span, tool_name=tool_name, parameters=args)
        if tool_call_id:
            state.tools[tool_call_id] = tool_span
        else:
            state.pending_tools_by_name.setdefault(tool_name, []).append(tool_span)
        state.last_updated_at = time.time()


@_fail_open
def on_post_tool_call(
    *,
    tool_name: str = "",
    args: Any = None,
    result: Any = None,
    task_id: str = "",
    session_id: str = "",
    tool_call_id: str = "",
    turn_id: str = "",
    api_request_id: str = "",
    **kwargs: Any,
) -> None:
    task_key = _trace_key(task_id, session_id, turn_id=turn_id, api_request_id=api_request_id)
    with _STATE_LOCK:
        state = _TRACE_STATE.get(task_key)
        if state is None:
            return
        tool_span = state.tools.pop(tool_call_id, None) if tool_call_id else None
        if tool_span is None:
            pending = state.pending_tools_by_name.get(tool_name) or []
            tool_span = pending.pop(0) if pending else None
        if tool_span is None:
            return
        state.last_updated_at = time.time()
    set_tool_attributes(tool_span, tool_name=tool_name, parameters=args, output=result)
    failed, description = _tool_call_failed(result=result, **kwargs)
    if failed:
        err_type = "Error"
        if isinstance(result, Exception):
            err_type = type(result).__name__
        elif kwargs.get("error") is not None:
            err = kwargs["error"]
            err_type = type(err).__name__ if isinstance(err, Exception) else "Error"
        mark_span_error(
            tool_span,
            error_type=err_type,
            error_message=description,
            exc=result if isinstance(result, Exception) else None,
        )
    tool_span.end()


@_fail_open
def on_api_request_error(
    *,
    task_id: str = "",
    session_id: str = "",
    api_call_count: int = 0,
    turn_id: str = "",
    api_request_id: str = "",
    error: Any = None,
    retryable: bool = False,
    **kwargs: Any,
) -> None:
    task_key = _trace_key(task_id, session_id, turn_id=turn_id, api_request_id=api_request_id)
    req_key = _request_key(api_call_count)
    with _STATE_LOCK:
        state = _TRACE_STATE.get(task_key)
        llm_span = state.generations.pop(req_key, None) if state else None
    if llm_span is None:
        return
    err_type, err_msg = _error_metadata(error, **kwargs)
    mark_span_error(
        llm_span,
        error_type=err_type,
        error_message=err_msg,
        exc=error if isinstance(error, BaseException) else None,
    )
    llm_span.end()
    if not retryable:
        _finish_trace(
            task_key,
            status_ok=False,
            error_type=err_type,
            error_message=err_msg,
        )


@_fail_open
def on_session_finalize(*, session_id: str = "", **kwargs: Any) -> None:
    _finish_session_traces(session_id)
    try:
        from .trace_state import _force_flush_traces
    except ImportError:
        from hermes.plugins.opentelemetry.trace_state import _force_flush_traces
    _force_flush_traces()


on_session_end = on_session_finalize


def register(ctx: Any) -> None:
    ctx.register_hook("pre_api_request", on_pre_llm_request)
    ctx.register_hook("post_api_request", on_post_llm_call)
    ctx.register_hook("api_request_error", on_api_request_error)
    ctx.register_hook("pre_llm_call", on_pre_llm_call)
    ctx.register_hook("post_llm_call", on_post_llm_call)
    ctx.register_hook("pre_tool_call", on_pre_tool_call)
    ctx.register_hook("post_tool_call", on_post_tool_call)
    ctx.register_hook("on_session_finalize", on_session_finalize)
    ctx.register_hook("on_session_end", on_session_finalize)
