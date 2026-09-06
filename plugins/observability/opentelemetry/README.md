# observability/opentelemetry

Standard OpenTelemetry tracing for Hermes Agent with [OpenInference](https://github.com/Arize-ai/openinference) GenAI semantics.

## Enable

```bash
hermes plugins enable observability/opentelemetry
```

Disable the legacy Langfuse plugin when enabling this one:

```bash
hermes plugins disable observability/langfuse
```

## Export

Spans export via OTLP gRPC to the local collector (default `127.0.0.1:4317`).

| Variable | Default | Purpose |
|----------|---------|---------|
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `127.0.0.1:4317` | Collector endpoint |
| `HERMES_OTEL_ENABLED` | enabled | Set `false` to disable tracing |
| `HERMES_PROFILE` | — | Attached as `hermes.profile` on turn spans |

Tracing fails open: collector downtime or export errors never interrupt the agent loop.
