"""OpenTelemetry tracing with OpenInference GenAI semantics for Hermes hooks."""

from __future__ import annotations

import logging
import os
import threading

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import SpanExporter
from opentelemetry.trace import Tracer

try:
    from .serialization import resolve_usage as _resolve_usage
except ImportError:
    from hermes.plugins.opentelemetry.serialization import resolve_usage as _resolve_usage

try:
    from hermes.telemetry.otlp import (
        build_otlp_span_exporter,
        install_tracer_provider,
        normalize_otlp_endpoint,
    )
except ImportError:
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    def normalize_otlp_endpoint(raw: str) -> str:
        value = (raw or "").strip()
        return value or "127.0.0.1:4318"

    def build_otlp_span_exporter(endpoint: str, *, insecure: bool = True) -> SpanExporter:
        # Prefer HTTP exporter
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter as OTLPHTTPSpanExporter
            http_endpoint = endpoint if endpoint.startswith("http") else f"http://{endpoint}"
            if not http_endpoint.endswith("/v1/traces"):
                http_endpoint = f"{http_endpoint.rstrip('/')}/v1/traces"
            return OTLPHTTPSpanExporter(endpoint=http_endpoint)
        except ImportError:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter as OTLPGRPCSpanExporter
            return OTLPGRPCSpanExporter(endpoint=endpoint, insecure=insecure)

    def install_tracer_provider(*, resource: Resource, exporter: SpanExporter, tracer_name: str = "hermes.telemetry") -> trace.Tracer:
        provider = TracerProvider(resource=resource)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        return trace.get_tracer(tracer_name)

logger = logging.getLogger(__name__)

_INIT_FAILED = object()
_TRACER: Tracer | None | object = None
_TRACER_LOCK = threading.RLock()


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _build_resource() -> Resource:
    attrs: dict[str, str] = {"service.name": "hermes-agent"}
    profile = _env("HERMES_PROFILE")
    if profile:
        attrs["hermes.profile"] = profile
    return Resource.create(attrs)


def _shutdown_provider() -> None:
    provider = trace.get_tracer_provider()
    shutdown = getattr(provider, "shutdown", None)
    if callable(shutdown):
        try:
            shutdown()
        except Exception:
            pass
    # ponytail: private OTel reset for tests only; production never calls reset_tracer().
    trace._TRACER_PROVIDER = None  # type: ignore[attr-defined]
    trace._TRACER_PROVIDER_SET_ONCE._done = False  # type: ignore[attr-defined]


def configure_tracer(
    *,
    exporter: SpanExporter | None = None,
    endpoint: str | None = None,
) -> Tracer | None:
    """Configure the process tracer; returns None when tracing stays disabled."""
    global _TRACER
    with _TRACER_LOCK:
        if exporter is None and _env("HERMES_OTEL_ENABLED").lower() in {"0", "false", "no", "off"}:
            _TRACER = _INIT_FAILED
            return None
        if exporter is None:
            endpoint = normalize_otlp_endpoint(endpoint or _env("OTEL_EXPORTER_OTLP_ENDPOINT"))
            try:
                exporter = build_otlp_span_exporter(endpoint)
            except Exception as exc:
                logger.warning(
                    "OpenTelemetry plugin: exporter init failed (%s); tracing disabled",
                    exc,
                )
                _TRACER = _INIT_FAILED
                return None
        _shutdown_provider()
        _TRACER = install_tracer_provider(
            resource=_build_resource(),
            exporter=exporter,
            tracer_name="hermes.plugins.opentelemetry",
        )
        return _TRACER  # type: ignore[return-value]


def reset_tracer() -> None:
    """Clear cached tracer and in-memory turn state (tests)."""
    try:
        from .trace_state import clear_trace_state
    except ImportError:
        from hermes.plugins.opentelemetry.trace_state import clear_trace_state

    global _TRACER
    with _TRACER_LOCK:
        _shutdown_provider()
        _TRACER = None
    clear_trace_state()


def _get_tracer() -> Tracer | None:
    global _TRACER
    if _TRACER is _INIT_FAILED:
        return None
    if _TRACER is not None:
        return _TRACER  # type: ignore[return-value]
    with _TRACER_LOCK:
        if _TRACER is _INIT_FAILED:
            return None
        if _TRACER is not None:
            return _TRACER  # type: ignore[return-value]
        if _env("HERMES_OTEL_ENABLED").lower() in {"0", "false", "no", "off"}:
            _TRACER = _INIT_FAILED
            return None
        return configure_tracer()


try:
    from .hooks import (  # noqa: E402
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
    from hermes.plugins.opentelemetry.hooks import (  # noqa: E402
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
    "configure_tracer",
    "on_api_request_error",
    "on_post_llm_call",
    "on_post_tool_call",
    "on_pre_llm_call",
    "on_pre_llm_request",
    "on_pre_tool_call",
    "on_session_end",
    "on_session_finalize",
    "register",
    "reset_tracer",
]
