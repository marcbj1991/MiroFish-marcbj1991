"""
LLM客户端封装

Two providers share one OpenAI-shaped call surface
(``client.chat.completions.create``):

* ``anthropic`` (default): the official Anthropic SDK (Messages API).
* ``openai``: any OpenAI-compatible server (unchanged legacy path).
"""

import json
import logging
import re
from types import SimpleNamespace
from typing import Optional, Dict, Any, List

import anthropic
from openai import OpenAI

from ..config import Config, resolve_stage_model
from .openai_chat_compat import create_chat_completion, extract_chat_completion_text


logger = logging.getLogger(__name__)


class LLMResponseError(ValueError):
    """A safe, structured error for unusable model responses."""

    def __init__(self, message: str, *, finish_reason: Optional[str] = None):
        super().__init__(message)
        self.finish_reason = finish_reason


def _is_response_format_unsupported(error: Exception) -> bool:
    """Detect an explicit provider rejection of JSON response_format."""

    if getattr(error, "status_code", None) not in {400, 422}:
        return False

    body = getattr(error, "body", None)
    if not isinstance(body, dict):
        return False

    details = body.get("error", body)
    if not isinstance(details, dict):
        return False

    param = str(details.get("param") or "").strip().lower()
    if param == "response_format" or param.startswith("response_format."):
        return True

    message = str(details.get("message") or "").lower()
    if "response_format" not in message:
        return False

    code = str(details.get("code") or "").lower()
    unsupported_codes = {
        "unsupported_parameter",
        "unsupported_value",
        "unknown_parameter",
        "invalid_parameter",
    }
    unsupported_phrases = (
        "not support",
        "unsupported",
        "unknown parameter",
        "unrecognized parameter",
    )
    return code in unsupported_codes or any(
        phrase in message for phrase in unsupported_phrases
    )


def _clean_chat_text(content: str) -> str:
    """Remove common reasoning wrappers and an outer Markdown JSON fence."""

    cleaned = re.sub(r'<think>[\s\S]*?</think>', '', content).strip()
    cleaned = cleaned.lstrip("\ufeff")
    cleaned = re.sub(r'^```(?:json)?\s*\n?', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\n?```\s*$', '', cleaned)
    return cleaned.strip()


def _contains_additional_json_container(content: str) -> bool:
    """Return True when trailing text embeds another JSON object or array."""

    decoder = json.JSONDecoder()
    for match in re.finditer(r"[\[{]", content):
        try:
            value, _ = decoder.raw_decode(content[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, (dict, list)):
            return True
    return False


class LLMRefusalError(LLMResponseError):
    """Claude declined the request (``stop_reason == "refusal"``)."""


# Anthropic request policy. Thinking is always on for current Claude models
# and its tokens count against max_tokens, so small caller caps are raised to
# this floor (non-streaming requests should stay around 16k).
ANTHROPIC_MIN_MAX_TOKENS = 16000
ANTHROPIC_EFFORT = "medium"
# Server-side refusal fallback (Anthropic's recommended substitute model).
ANTHROPIC_FALLBACK_BETA = "server-side-fallback-2026-07-01"
ANTHROPIC_JSON_INSTRUCTION = (
    "Respond with exactly one valid JSON object and nothing else: no Markdown "
    "code fences, no text before or after the object."
)
_ANTHROPIC_FINISH_REASONS = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
}


class _AnthropicChatCompletions:
    """Translate an OpenAI-style chat request into one Messages API call."""

    def __init__(self, client: "anthropic.Anthropic"):
        self._client = client

    def create(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_completion_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> Any:
        # temperature is accepted for signature compatibility and ignored:
        # current Claude models reject sampling parameters.
        system_parts = [
            str(message.get("content") or "")
            for message in messages
            if message.get("role") == "system"
        ]
        if response_format and response_format.get("type") == "json_object":
            system_parts.append(ANTHROPIC_JSON_INSTRUCTION)
        chat_messages = [
            {"role": message["role"], "content": message.get("content") or ""}
            for message in messages
            if message.get("role") != "system"
        ]

        requested = max_tokens or max_completion_tokens or 0
        kwargs: Dict[str, Any] = {
            "model": model,
            "max_tokens": max(requested, ANTHROPIC_MIN_MAX_TOKENS),
            "messages": chat_messages,
            "output_config": {"effort": ANTHROPIC_EFFORT},
        }
        system = "\n\n".join(part for part in system_parts if part.strip())
        if system:
            kwargs["system"] = system
        # Haiku has no server-side fallback; every other current model does.
        if not model.startswith("claude-haiku"):
            kwargs["betas"] = [ANTHROPIC_FALLBACK_BETA]
            kwargs["fallbacks"] = "default"

        response = self._client.beta.messages.create(**kwargs)

        stop_reason = getattr(response, "stop_reason", None)
        if stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            raise LLMRefusalError(
                f"Claude declined the request (stop_reason=refusal, category={category})",
                finish_reason="refusal",
            )
        text = "".join(
            block.text
            for block in (getattr(response, "content", None) or [])
            if getattr(block, "type", None) == "text"
        )
        return SimpleNamespace(
            model=getattr(response, "model", model),
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=text),
                    finish_reason=_ANTHROPIC_FINISH_REASONS.get(
                        stop_reason, stop_reason
                    ),
                )
            ],
        )


class AnthropicChatAdapter:
    """Anthropic SDK client exposing ``.chat.completions.create``."""

    def __init__(self, api_key: str, http_client: Any = None):
        client_kwargs: Dict[str, Any] = {"api_key": api_key}
        if http_client is not None:
            client_kwargs["http_client"] = http_client
        self.anthropic = anthropic.Anthropic(**client_kwargs)
        self.chat = SimpleNamespace(
            completions=_AnthropicChatCompletions(self.anthropic)
        )


def create_llm_sdk_client(
    api_key: str,
    base_url: Optional[str] = None,
    provider: Optional[str] = None,
) -> Any:
    """Return the SDK client for the configured provider.

    Both return values support ``client.chat.completions.create(...)`` as
    used by ``openai_chat_compat.create_chat_completion``.
    """

    provider = (provider or Config.LLM_PROVIDER).strip().lower()
    if provider == "anthropic":
        return AnthropicChatAdapter(api_key)
    if provider == "openai":
        return OpenAI(api_key=api_key, base_url=base_url or Config.LLM_BASE_URL)
    raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")


class LLMClient:
    """LLM客户端"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        stage: Optional[str] = None,
    ):
        self.api_key = api_key or Config.LLM_API_KEY
        self.base_url = base_url or Config.LLM_BASE_URL
        # An explicit model wins; otherwise the pipeline stage picks it
        # (see config.resolve_stage_model).
        if model:
            self.model = model
        elif stage:
            self.model = resolve_stage_model(stage)
        else:
            self.model = Config.LLM_MODEL_NAME
        
        if not self.api_key:
            raise ValueError("LLM_API_KEY 未配置")
        
        self.client = create_llm_sdk_client(self.api_key, self.base_url)

    def _create_completion(
        self,
        *,
        messages: List[Dict[str, str]],
        temperature: Optional[float],
        max_tokens: Optional[int],
        response_format: Optional[Dict[str, Any]],
    ) -> Any:
        """Send one raw Chat Completions request through the compatibility layer."""

        return create_chat_completion(
            self.client,
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = 4096,
        response_format: Optional[Dict] = None
    ) -> str:
        """
        发送聊天请求
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大token数
            response_format: 响应格式（如JSON模式）
            
        Returns:
            模型响应文本
        """
        response = self._create_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )
        content = extract_chat_completion_text(response)
        return _clean_chat_text(content)
    
    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: Optional[int] = 4096,
        max_attempts: int = 1,
    ) -> Dict[str, Any]:
        """
        发送聊天请求并返回JSON
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大token数
            max_attempts: 内容生成尝试次数（不含一次明确的JSON模式能力降级）
            
        Returns:
            解析后的JSON对象
        """
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")

        response_format: Optional[Dict[str, str]] = {"type": "json_object"}
        request_max_tokens = max_tokens
        last_error: Optional[LLMResponseError] = None

        for attempt in range(1, max_attempts + 1):
            # JSON-mode capability negotiation is separate from content
            # regeneration. An explicit response_format rejection may add one
            # request, but it must not consume a content attempt.
            while True:
                try:
                    response = self._create_completion(
                        messages=messages,
                        temperature=temperature,
                        max_tokens=request_max_tokens,
                        response_format=response_format,
                    )
                except Exception as error:
                    if (
                        response_format is not None
                        and _is_response_format_unsupported(error)
                    ):
                        logger.warning(
                            "LLM provider explicitly rejected response_format; "
                            "retrying once with prompt-only JSON guidance"
                        )
                        response_format = None
                        continue
                    raise
                break

            try:
                return self._parse_json_response(response)
            except LLMResponseError as error:
                last_error = error
                if attempt >= max_attempts:
                    raise

                # A caller-supplied cap is the common cause of a partial JSON
                # object. Omit it for the one bounded retry so the provider can
                # use its model-specific output limit.
                had_token_cap = request_max_tokens is not None
                request_max_tokens = None
                logger.warning(
                    "LLM returned unusable JSON (finish_reason=%s); "
                    "retrying content generation%s",
                    error.finish_reason or "unknown",
                    " without an output token cap" if had_token_cap else "",
                )

        if last_error is not None:  # pragma: no cover - defensive loop guard
            raise last_error
        raise LLMResponseError("LLM did not produce a JSON response")

    @staticmethod
    def _parse_json_response(response: Any) -> Dict[str, Any]:
        choices = getattr(response, "choices", None) or []
        if not choices:
            raise LLMResponseError("LLM returned no choices")

        choice = choices[0]
        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason == "length":
            raise LLMResponseError(
                "LLM JSON output was truncated at the token limit",
                finish_reason=finish_reason,
            )
        if finish_reason not in {None, "stop"}:
            raise LLMResponseError(
                f"LLM JSON generation stopped unexpectedly ({finish_reason})",
                finish_reason=finish_reason,
            )

        content = _clean_chat_text(extract_chat_completion_text(response))
        if not content:
            raise LLMResponseError(
                "LLM returned empty JSON content",
                finish_reason=finish_reason,
            )

        try:
            value = json.loads(content)
        except json.JSONDecodeError as strict_error:
            # Some compatible providers append a short explanation after an
            # otherwise complete JSON object. Accept only an object decoded
            # from the beginning; never repair or invent truncated JSON.
            try:
                value, end = json.JSONDecoder().raw_decode(content)
            except json.JSONDecodeError:
                raise LLMResponseError(
                    "LLM returned invalid JSON "
                    f"(line {strict_error.lineno}, column {strict_error.colno})",
                    finish_reason=finish_reason,
                ) from strict_error

            trailing = content[end:].strip()
            if trailing:
                if _contains_additional_json_container(trailing):
                    raise LLMResponseError(
                        "LLM returned multiple JSON values",
                        finish_reason=finish_reason,
                    )
                logger.warning("Ignoring text after a complete LLM JSON object")

        if not isinstance(value, dict):
            raise LLMResponseError(
                "LLM JSON response must be a top-level JSON object",
                finish_reason=finish_reason,
            )

        return value
