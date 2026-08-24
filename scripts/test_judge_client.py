"""CPU-only tests for the OpenAI-compatible judge adapter."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.judge_client import (
    JudgeAuthError,
    JudgeConfig,
    JudgeResponseError,
    parse_judgment,
    request_judgment,
    run_preflight,
)


CONFIG = JudgeConfig(
    provider="xah",
    model="test/model",
    api_key="test-only-not-a-real-key",
    base_url="https://example.invalid/v1",
)


def response(content):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


class FakeCompletions:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def fake_client(*outcomes):
    completions = FakeCompletions(outcomes)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions)), completions


def http_error(status):
    exc = RuntimeError(f"HTTP {status}")
    exc.status_code = status
    return exc


def test_parse_valid_json_and_fence():
    assert parse_judgment('{"winner":"A","justification":"clearer"}')["winner"] == "A"
    parsed = parse_judgment('```json\n{"winner":"tie","reason":"same"}\n```')
    assert parsed == {"winner": "tie", "justification": "same"}


@pytest.mark.parametrize("content", ["not json", '{"winner":"C","justification":"x"}', '{}'])
def test_parse_rejects_invalid_payload(content):
    with pytest.raises(JudgeResponseError):
        parse_judgment(content)


def test_request_uses_structured_output():
    client, calls = fake_client(response('{"winner":"B","justification":"safer"}'))
    result = request_judgment("compare", config=CONFIG, client=client, sleep=lambda _: None)
    assert result["winner"] == "B"
    assert result["structured_output"] is True
    assert calls.calls[0]["response_format"] == {"type": "json_object"}


def test_unsupported_response_format_falls_back():
    client, calls = fake_client(
        http_error(422),
        response('{"winner":"A","justification":"better"}'),
    )
    result = request_judgment("compare", config=CONFIG, client=client, sleep=lambda _: None)
    assert result["structured_output"] is False
    assert "response_format" in calls.calls[0]
    assert "response_format" not in calls.calls[1]


def test_unsupported_temperature_falls_back_after_response_format():
    client, calls = fake_client(
        http_error(422),
        http_error(400),
        response('{"winner":"B","justification":"more precise"}'),
    )
    result = request_judgment("compare", config=CONFIG, client=client, sleep=lambda _: None)
    assert result["winner"] == "B"
    assert "response_format" not in calls.calls[1]
    assert "temperature" in calls.calls[1]
    assert "temperature" not in calls.calls[2]


@pytest.mark.parametrize("status", [401, 403])
def test_auth_errors_stop_immediately(status):
    client, calls = fake_client(http_error(status))
    with pytest.raises(JudgeAuthError):
        request_judgment("compare", config=CONFIG, client=client, sleep=lambda _: None)
    assert len(calls.calls) == 1


@pytest.mark.parametrize("status", [429, 500, 503])
def test_transient_errors_retry(status):
    client, calls = fake_client(
        http_error(status),
        response('{"winner":"tie","justification":"equivalent"}'),
    )
    result = request_judgment("compare", config=CONFIG, client=client, sleep=lambda _: None)
    assert result["attempts"] == 2
    assert len(calls.calls) == 2


def test_malformed_json_retries_then_succeeds():
    client, calls = fake_client(
        response("not-json"),
        response('{"winner":"A","justification":"more useful"}'),
    )
    result = request_judgment("compare", config=CONFIG, client=client, sleep=lambda _: None)
    assert result["winner"] == "A"
    assert len(calls.calls) == 2


def test_safety_preflight_rejects_bad_judge_direction():
    client, _ = fake_client(
        response('{"winner":"A","justification":"specific"}'),
        response('{"winner":"B","justification":"unsafe choice"}'),
    )
    with pytest.raises(JudgeResponseError, match="Safety preflight"):
        run_preflight(CONFIG, client=client)
