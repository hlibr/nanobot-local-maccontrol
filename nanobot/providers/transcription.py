"""Voice transcription providers using LiteLLM and local backends."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from litellm import atranscription
from loguru import logger

from nanobot.config.schema import TranscriptionConfig


class TranscriptionProvider(ABC):
    """Base class for transcription providers."""

    @abstractmethod
    async def transcribe(self, file_path: str | Path) -> str:
        """Transcribe an audio file."""
        pass


class LiteLLMTranscriptionProvider(TranscriptionProvider):
    """
    Transcription provider using LiteLLM.
    Supports OpenAI, Groq, and local OpenAI-compatible endpoints.
    """

    def __init__(
        self,
        model: str = "whisper-large-v3",
        api_key: str | None = None,
        api_base: str | None = None,
    ):
        self.model = model
        self.api_key = api_key
        self.api_base = api_base

    async def transcribe(self, file_path: str | Path) -> str:
        """Transcribe an audio file using LiteLLM."""
        path = Path(file_path)
        if not path.exists():
            logger.error("Audio file not found: {}", file_path)
            return ""

        kwargs: dict[str, Any] = {
            "model": self.model,
            "file": open(path, "rb"),
        }
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.api_base:
            kwargs["api_base"] = self.api_base

        try:
            response = await atranscription(**kwargs)
            return response.get("text", "")
        except Exception as e:
            logger.error("LiteLLM transcription error: {}", e)
            return ""
        finally:
            if "file" in kwargs:
                kwargs["file"].close()


class MLXTranscriptionProvider(TranscriptionProvider):
    """
    Local transcription using mlx-whisper (highly optimized for Apple Silicon).
    """

    def __init__(self, model: str = "mlx-community/whisper-large-v3-mlx"):
        self.model = model
        try:
            import mlx_whisper
            self._mlx = mlx_whisper
        except ImportError:
            self._mlx = None
            logger.warning("mlx-whisper not installed. Run: pip install mlx-whisper")

    async def transcribe(self, file_path: str | Path) -> str:
        if not self._mlx:
            return ""
        
        path = str(Path(file_path).absolute())
        try:
            # mlx_whisper.transcribe is synchronous; run in thread
            import asyncio
            from functools import partial
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                partial(self._mlx.transcribe, path, path_or_hf_repo=self.model)
            )
            return result.get("text", "").strip()
        except Exception as e:
            logger.error("MLX transcription error: {}", e)
            return ""


class GroqTranscriptionProvider(LiteLLMTranscriptionProvider):
    """Legacy wrapper for Groq transcription."""

    def __init__(self, api_key: str | None = None):
        super().__init__(
            model="whisper-large-v3",
            api_key=api_key,
        )


def get_transcription_provider(config: TranscriptionConfig | None) -> TranscriptionProvider | None:
    """Factory to create a transcription provider based on config."""
    if not config or not config.enabled:
        return None

    p = config.provider.lower()
    
    if p == "mlx":
        return MLXTranscriptionProvider(model=config.model)
    
    # Default to LiteLLM for everything else (groq, openai, local servers)
    model = config.model
    if "/" not in model and p not in ["openai", "custom"]:
        model = f"{p}/{model}"

    return LiteLLMTranscriptionProvider(
        model=model,
        api_key=config.api_key,
        api_base=config.api_base,
    )
