#!/usr/bin/env python3
"""Verify Hermes OpenTelemetry trace capture for configured LLM providers.

Runs a minimal one-turn inference per provider and asserts spans land in the
local OTLP collector evidence files:
  /var/lib/hermes/evidence/traces/traces.jsonl
  /var/lib/hermes/evidence/traces/traces_error_only.jsonl

Usage:
  uv run python scripts/verify_providers_otel.py
  uv run python scripts/verify_providers_otel.py --provider opencode-go
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

PROMPT = "Reply with exactly: OTEL_SMOKE_OK"
HERMES_HOME = Path(os.environ.get("HERMES_HOME", "/var/lib/hermes"))
TRACES_JSONL = Path("/var/lib/hermes/evidence/traces/traces.jsonl")
ERROR_TRACES_JSONL = Path("/var/lib/hermes/evidence/traces/traces_error_only.jsonl")
SERVICE_NAME = "hermes-agent"
FLUSH_WAIT_S = 2.0


@dataclass(frozen=True)
class ProviderCase:
    key: str
    provider: str
    model: str
    expect_success: bool = True
    expect_error_substrings: tuple[str, ...] = ()


PROVIDER_CASES: tuple[ProviderCase, ...] = (
    ProviderCase("opencode-go", "opencode-go", "deepseek-v4-flash"),
    ProviderCase("openrouter", "openrouter", "deepseek/deepseek-v4-flash"),
    ProviderCase("openai-codex", "openai-codex", "gpt-5.5"),
    ProviderCase(
        "deepseek",
        "deepseek",
        "deepseek-v4-flash",
        expect_success=False,
        expect_error_substrings=("402", "Insufficient Balance", "insufficient balance"),
    ),
)


def _attr_value(raw: dict[str, Any]) -> Any:
    if "stringValue" in raw:
        return raw["stringValue"]
    if "boolValue" in raw:
        return raw["boolValue"]
    if "intValue" in raw:
        return int(raw["intValue"])
    if "doubleValue" in raw:
        return raw["doubleValue"]
    return raw


def _span_attrs(span: dict[str, Any]) -> dict[str, Any]:
    return {a["key"]: _attr_value(a["value"]) for a in span.get("attributes", [])}


def _resource_service(record: dict[str, Any]) -> str | None:
    for rs in record.get("resourceSpans", []):
        for attr in rs.get("resource", {}).get("attributes", []):
            if attr.get("key") == "service.name":
                return _attr_value(attr["value"])
    return None


def _iter_spans(record: dict[str, Any]) -> Iterable[dict[str, Any]]:
    service = _resource_service(record)
    for rs in record.get("resourceSpans", []):
        for scope in rs.get("scopeSpans", []):
            for span in scope.get("spans", []):
                yield {
                    "service": service,
                    "name": span.get("name"),
                    "status": span.get("status") or {},
                    "attrs": _span_attrs(span),
                }


def _read_new_lines(path: Path, offset: int) -> list[str]:
    if not path.exists():
        return []
    with path.open("rb") as handle:
        handle.seek(offset)
        data = handle.read()
    return [line for line in data.decode("utf-8", errors="replace").splitlines() if line.strip()]


def _file_offset(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def _bootstrap_env() -> None:
    os.environ.setdefault("HERMES_HOME", str(HERMES_HOME))
    os.environ.setdefault("HERMES_YOLO_MODE", "1")
    os.environ.setdefault("HERMES_ACCEPT_HOOKS", "1")
    os.environ.pop("HERMES_OTEL_ENABLED", None)
    repo = Path(__file__).resolve().parents[1]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from hermes_cli.env_loader import load_hermes_dotenv
    from hermes_constants import get_hermes_home

    load_hermes_dotenv(hermes_home=get_hermes_home())
    from hermes_cli.plugins import discover_plugins

    discover_plugins()
    from plugins.observability.opentelemetry.core import configure_tracer

    configure_tracer()


def _run_provider_turn(case: ProviderCase) -> tuple[dict[str, Any], str | None]:
    from hermes_cli.config import load_config
    from hermes_cli.runtime_provider import resolve_runtime_provider
    from run_agent import AIAgent

    cfg = load_config()
    runtime = resolve_runtime_provider(
        requested=case.provider,
        target_model=case.model,
    )
    session_id = f"otel-verify-{case.key}-{uuid.uuid4().hex[:8]}"
    agent = None
    result: dict[str, Any] = {}
    error: str | None = None
    try:
        agent = AIAgent(
            api_key=runtime.get("api_key"),
            base_url=runtime.get("base_url"),
            provider=runtime.get("provider"),
            requested_provider=runtime.get("requested_provider"),
            api_mode=runtime.get("api_mode"),
            model=case.model,
            enabled_toolsets=[],
            disabled_toolsets=["*"],
            quiet_mode=True,
            platform="cli",
            session_id=session_id,
            credential_pool=runtime.get("credential_pool"),
            skip_memory=True,
            skip_background_review=True,
            max_iterations=1,
        )
        agent.suppress_status_output = True
        result = agent.run_conversation(PROMPT)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if agent is not None:
            try:
                agent.close()
            except Exception:
                pass
    return result, error


def _span_is_error(span: dict[str, Any]) -> bool:
    attrs = span.get("attrs") or {}
    status = span.get("status") or {}
    if attrs.get("error") is True:
        return True
    return int(status.get("code") or 0) == 2


def _matches_provider(span: dict[str, Any], case: ProviderCase) -> bool:
    attrs = span.get("attrs") or {}
    provider = str(attrs.get("llm.provider") or attrs.get("hermes.platform") or "").lower()
    return case.provider.lower() in provider or provider in case.provider.lower()


def _matches_model(span: dict[str, Any], case: ProviderCase) -> bool:
    attrs = span.get("attrs") or {}
    model = str(attrs.get("llm.model_name") or "").lower()
    target = case.model.lower()
    if not model:
        return True
    return target in model or model in target or target.split("/")[-1] in model


def _verify_case(case: ProviderCase, *, verbose: bool) -> dict[str, Any]:
    before_main = _file_offset(TRACES_JSONL)
    before_err = _file_offset(ERROR_TRACES_JSONL)
    result, run_error = _run_provider_turn(case)
    time.sleep(FLUSH_WAIT_S)
    main_lines = _read_new_lines(TRACES_JSONL, before_main)
    err_lines = _read_new_lines(ERROR_TRACES_JSONL, before_err)
    main_spans = [span for line in main_lines for span in _iter_spans(json.loads(line))]
    err_spans = [span for line in err_lines for span in _iter_spans(json.loads(line))]
    hermes_spans = [s for s in main_spans if s.get("service") == SERVICE_NAME]
    llm_spans = [
        s for s in hermes_spans
        if s.get("name", "").startswith("LLM call") or s.get("attrs", {}).get("openinference.span.kind") == "LLM"
    ]
    matched_llm = [s for s in llm_spans if _matches_provider(s, case)]
    agent_spans = [s for s in hermes_spans if "Hermes turn" in str(s.get("name"))]
    errors = []
    if not hermes_spans:
        errors.append(f"no spans with service.name={SERVICE_NAME!r} in new traces.jsonl records")
    if not matched_llm:
        errors.append(f"no LLM span for provider={case.provider!r} model={case.model!r}")
    for span in matched_llm:
        if not _matches_model(span, case):
            errors.append(f"LLM span model mismatch: got {span.get('attrs', {}).get('llm.model_name')!r}")
    if case.expect_success:
        if run_error:
            errors.append(f"run raised: {run_error}")
        if not result.get("completed") and not result.get("final_response"):
            errors.append(f"turn did not complete: {result!r}")
        for span in matched_llm:
            if _span_is_error(span):
                errors.append(f"unexpected error LLM span: {span.get('attrs', {}).get('error.message')}")
    else:
        err_messages = " ".join(
            str((s.get("attrs") or {}).get("error.message") or (s.get("status") or {}).get("message") or "")
            for s in hermes_spans + err_spans
        ).lower()
        if not any(token.lower() in err_messages for token in case.expect_error_substrings):
            errors.append(
                "expected error trace containing one of "
                f"{case.expect_error_substrings!r}; got run_error={run_error!r} messages={err_messages!r}"
            )
        if not any(_span_is_error(s) for s in hermes_spans):
            errors.append("expected at least one error-marked span in traces.jsonl")
        if not err_spans:
            errors.append("expected new records in traces_error_only.jsonl")
    report = {
        "provider": case.provider,
        "model": case.model,
        "expect_success": case.expect_success,
        "run_error": run_error,
        "result_summary": {
            "completed": result.get("completed"),
            "failed": result.get("failed"),
            "api_calls": result.get("api_calls"),
            "final_response": (result.get("final_response") or "")[:120],
        },
        "new_main_records": len(main_lines),
        "new_error_records": len(err_lines),
        "hermes_spans": hermes_spans,
        "matched_llm_spans": matched_llm,
        "agent_spans": agent_spans,
        "errors": errors,
        "ok": not errors,
    }
    if verbose:
        print(json.dumps(report, indent=2, default=str))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider",
        choices=[c.key for c in PROVIDER_CASES],
        help="Run a single provider case (default: all)",
    )
    parser.add_argument("--verbose", action="store_true", help="Print full JSON reports")
    args = parser.parse_args(argv)
    _bootstrap_env()
    cases = [c for c in PROVIDER_CASES if args.provider is None or c.key == args.provider]
    print("Verifying {} provider(s); traces -> {}".format(len(cases), TRACES_JSONL))
    all_ok = True
    for case in cases:
        print("")
        print("=== {} ({}/{}) ===".format(case.key, case.provider, case.model))
        report = _verify_case(case, verbose=args.verbose)
        if report["ok"]:
            llm = report["matched_llm_spans"][0] if report["matched_llm_spans"] else {}
            attrs = llm.get("attrs") or {}
            print(
                "PASS"
                " main_records={}"
                " error_records={}"
                " llm.provider={!r}"
                " llm.model_name={!r}"
                " status={}".format(
                    report["new_main_records"],
                    report["new_error_records"],
                    attrs.get("llm.provider"),
                    attrs.get("llm.model_name"),
                    llm.get("status"),
                )
            )
        else:
            all_ok = False
            print("FAIL")
            for err in report["errors"]:
                print("  - {}".format(err))
            if report["matched_llm_spans"]:
                print("  matched_llm_spans:", json.dumps(report["matched_llm_spans"], indent=2, default=str))
            elif report["hermes_spans"]:
                print("  hermes_spans:", json.dumps(report["hermes_spans"], indent=2, default=str))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
