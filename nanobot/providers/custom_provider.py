"""Direct OpenAI-compatible provider — bypasses LiteLLM."""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

import json_repair
from openai import AsyncOpenAI

from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest


class CustomProvider(LLMProvider):

    def __init__(self, api_key: str = "no-key", api_base: str = "http://localhost:8000/v1", default_model: str = "default"):
        super().__init__(api_key, api_base)
        self.default_model = self._resolve_model(default_model)

        # Keep affinity stable for this provider instance to improve backend cache locality.
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=api_base,
            default_headers={"x-session-affinity": uuid.uuid4().hex},
        )

    def _resolve_model(self, model: str) -> str:
        """Strip provider prefix if present (e.g. custom/model -> model)."""
        if "/" in model:
            parts = model.split("/", 1)
            if parts[0] in ("custom", "auto"):
                return parts[1]
        return model

    async def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
                   model: str | None = None, max_tokens: int = 4096, temperature: float = 0.7,
                   reasoning_effort: str | None = None) -> LLMResponse:
        
        target_model = self._resolve_model(model or self.default_model)
        
        kwargs: dict[str, Any] = {
            "model": target_model,
            "messages": self._sanitize_empty_content(messages),
            "max_tokens": max(1, max_tokens),
            "temperature": temperature,
        }
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
        if tools:
            kwargs.update(tools=tools, tool_choice="auto")
        try:
            return self._parse(await self._client.chat.completions.create(**kwargs))
        except asyncio.CancelledError:
            raise
        except Exception as e:
            # Check for vision-unsupported errors
            err_str = str(e).lower()
            if any(kw in err_str for kw in ("image input", "multimodal", "image_url", "vision")):
                stripped_messages = self._strip_vision_content(kwargs["messages"])
                if stripped_messages != kwargs["messages"]:
                    from loguru import logger
                    logger.warning("Custom provider model {} appears to not support images. Retrying without images...", target_model)
                    kwargs["messages"] = stripped_messages
                    try:
                        return self._parse(await self._client.chat.completions.create(**kwargs))
                    except Exception as retry_e:
                        return LLMResponse(content=f"Error after image-strip retry: {retry_e}", finish_reason="error")

            return LLMResponse(content=f"Error: {e}", finish_reason="error")

    def _parse(self, response: Any) -> LLMResponse:
        choice = response.choices[0]
        msg = choice.message
        tool_calls = [
            ToolCallRequest(id=tc.id, name=tc.function.name,
                            arguments=json_repair.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments)
            for tc in (msg.tool_calls or [])
        ]
        u = response.usage
        return LLMResponse(
            content=msg.content, tool_calls=tool_calls, finish_reason=choice.finish_reason or "stop",
            usage={"prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens, "total_tokens": u.total_tokens} if u else {},
            reasoning_content=getattr(msg, "reasoning_content", None) or None,
        )

    def get_default_model(self) -> str:
        return self.default_model

