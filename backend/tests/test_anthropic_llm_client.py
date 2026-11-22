"""Anthropic provider path of LLMClient, tested without network access."""

import json

import anthropic
import httpx2
import pytest


@pytest.fixture(autouse=True)
def _no_base_url_override(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)

from app.config import Config
from app.utils.llm_client import (
    AnthropicChatAdapter,
    LLMClient,
    LLMRefusalError,
    create_llm_sdk_client,
)
from app.utils.openai_chat_compat import create_chat_completion


def _message(content, stop_reason="end_turn", stop_details=None):
    body = {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    if stop_details is not None:
        body["stop_details"] = stop_details
    return body


def _llm(response_body):
    requests = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(200, json=response_body)

    llm = LLMClient(api_key="test-key", model="claude-opus-5-5")
    llm.client = AnthropicChatAdapter(
        "test-key",
        http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)),
    )
    return llm, requests


def test_default_provider_is_anthropic_with_the_default_model(monkeypatch):
    assert Config.LLM_PROVIDER == "anthropic"
    assert isinstance(create_llm_sdk_client("test-key"), AnthropicChatAdapter)
    llm = LLMClient(api_key="test-key")
    assert isinstance(llm.client, AnthropicChatAdapter)
    assert llm.model == Config.LLM_MODEL_NAME


def test_chat_json_sends_a_valid_messages_request_and_parses_json():
    llm, requests = _llm(_message([
        {"type": "thinking", "thinking": "", "signature": "sig"},
        {"type": "text", "text": '```json\n{"answer": 42}\n```'},
    ]))

    result = llm.chat_json(
        [
            {"role": "system", "content": "You are a planner."},
            {"role": "user", "content": "Return JSON"},
        ],
        temperature=0.3,
        max_tokens=4096,
    )

    assert result == {"answer": 42}
    assert len(requests) == 1
    request = requests[0]
    assert request.url.host == "api.anthropic.com"
    assert request.url.path == "/v1/messages"
    auth = request.headers.get("x-api-key") or request.headers.get("authorization", "")
    assert auth.endswith("test-key")
    assert "server-side-fallback-2026-07-01" in request.headers["anthropic-beta"]
    body = json.loads(request.content)
    assert body["model"] == "claude-opus-5-5"
    assert body["system"].startswith("You are a planner.")
    assert "JSON object" in body["system"]
    assert body["messages"] == [{"role": "user", "content": "Return JSON"}]
    assert body["max_tokens"] >= 16000
    assert body["output_config"] == {"effort": "medium"}
    assert body["fallbacks"] == "default"
    for forbidden in ("temperature", "top_p", "top_k", "thinking", "response_format"):
        assert forbidden not in body


def test_chat_returns_only_text_blocks():
    llm, requests = _llm(_message([
        {"type": "thinking", "thinking": "", "signature": "sig"},
        {"type": "text", "text": "Hello"},
    ]))

    assert llm.chat([{"role": "user", "content": "Hi"}], temperature=0.9) == "Hello"
    body = json.loads(requests[0].content)
    assert "system" not in body
    assert "temperature" not in body


def test_refusal_raises_a_clear_error():
    llm, _ = _llm(_message(
        [],
        stop_reason="refusal",
        stop_details={"type": "refusal", "category": "cyber", "explanation": "declined"},
    ))

    with pytest.raises(LLMRefusalError, match="refusal.*cyber"):
        llm.chat_json([{"role": "user", "content": "Return JSON"}])


def test_max_tokens_stop_is_reported_as_truncation():
    llm, _ = _llm(_message([{"type": "text", "text": '{"items": ['}], stop_reason="max_tokens"))

    with pytest.raises(Exception, match="truncated"):
        llm.chat_json([{"role": "user", "content": "Return JSON"}])


def test_services_compat_layer_works_on_the_anthropic_adapter():
    llm, requests = _llm(_message([{"type": "text", "text": '{"bio": "x"}'}]))

    response = create_chat_completion(
        llm.client,
        model="claude-haiku-5-5",
        messages=[{"role": "user", "content": "profile"}],
        temperature=0.7,
        response_format={"type": "json_object"},
    )

    assert response.choices[0].finish_reason == "stop"
    assert response.choices[0].message.content == '{"bio": "x"}'
    body = json.loads(requests[0].content)
    assert body["model"] == "claude-haiku-5-5"
    assert "fallbacks" not in body  # no server-side fallback for Haiku
    assert "temperature" not in body
