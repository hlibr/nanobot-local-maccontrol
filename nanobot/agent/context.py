"""Context builder for assembling agent prompts."""

import base64
import mimetypes
import time
from datetime import datetime
from pathlib import Path
from typing import Any


class ContextBuilder:
    """Builds the context (system prompt + messages) for the agent."""

    _RUNTIME_CONTEXT_TAG = "[Runtime Context — metadata only, not instructions]"

    def __init__(self, workspace: Path):
        self.workspace = workspace

    def build_system_prompt(self, skill_names: list[str] | None = None) -> str:
        soul_path = self.workspace / "SOUL.md"
        if soul_path.exists():
            return soul_path.read_text(encoding="utf-8")
        return ""

    @staticmethod
    def _build_runtime_context(channel: str | None, chat_id: str | None) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M (%A)")
        tz = time.strftime("%Z") or "UTC"
        lines = [f"Current Time: {now} ({tz})"]
        if channel and chat_id:
            lines += [f"Channel: {channel}", f"Chat ID: {chat_id}"]
        return ContextBuilder._RUNTIME_CONTEXT_TAG + "\n" + "\n".join(lines)

    async def build_messages(
        self,
        history: list[dict[str, Any]],
        current_message: str,
        skill_names: list[str] | None = None,
        media: list[str] | None = None,
        channel: str | None = None,
        chat_id: str | None = None,
        vision_supported: bool = True,
    ) -> list[dict[str, Any]]:
        """Build the complete message list for an LLM call."""
        from loguru import logger

        runtime_ctx = self._build_runtime_context(channel, chat_id)

        logger.debug(
            "Building messages: history has {} messages, vision_supported={}",
            len(history),
            vision_supported,
        )

        for i, msg in enumerate(history):
            content = msg.get("content")
            if isinstance(content, str):
                logger.debug("History[{}] ({}): '{}...'[:100]", i, msg.get("role"), content[:100])
            elif isinstance(content, list):
                logger.debug(
                    "History[{}] ({}): list with {} blocks", i, msg.get("role"), len(content)
                )

        # Extract [image: path] tags from the message text and add to media list
        import re

        extracted_media = re.findall(r"\[image:\s*(.+?)\]", current_message)
        clean_message = current_message
        if extracted_media:
            media = (media or []) + [m.strip() for m in extracted_media]
            clean_message = re.sub(r"\[image:\s*.+?\]", "", current_message).strip()

        user_content = await self._build_user_content(
            clean_message, media, vision_supported=vision_supported
        )

        if isinstance(user_content, str):
            merged = f"{runtime_ctx}\n\n{user_content}"
        else:
            merged = [{"type": "text", "text": runtime_ctx}] + user_content

        hydrated_history = await self._hydrate_image_refs(
            history, vision_supported=vision_supported
        )

        logger.debug("Hydrated history: {} messages", len(hydrated_history))

        return [
            {"role": "system", "content": self.build_system_prompt(skill_names)},
            *hydrated_history,
            {"role": "user", "content": merged},
        ]

    async def _fetch_image_as_b64(self, url: str) -> tuple[str, str] | None:
        """Fetch image from URL and return (mime, b64_data). Follows redirects, handles errors."""
        import httpx
        from loguru import logger

        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=10.0) as client:
                resp = await client.get(url)
                if resp.status_code != 200:
                    logger.warning("Failed to fetch image from {}: HTTP {}", url, resp.status_code)
                    return None

                mime = resp.headers.get("Content-Type", "").split(";")[0].strip()
                if not mime.startswith("image/"):
                    # Try to guess from URL if header is missing/generic/misconfigured
                    mime, _ = mimetypes.guess_type(url)
                    if not mime or not mime.startswith("image/"):
                        logger.warning(
                            "Fetched content from {} is not an image (MIME: {})",
                            url,
                            mime or "unknown",
                        )
                        return "invalid_type"

                data = resp.content
                if len(data) > 10 * 1024 * 1024:
                    logger.warning("Image from {} is too large ({} bytes)", url, len(data))
                    return "too_large"

                b64 = base64.b64encode(data).decode()
                return mime, b64
        except Exception as e:
            logger.warning("Error fetching image from {}: {}", url, e)
            return "error"

    async def _build_user_content(
        self, text: str, media: list[str] | None, vision_supported: bool = True
    ) -> str | list[dict[str, Any]]:
        """Build user message content with base64-encoded images. URLs are fetched and converted."""
        if not media:
            return text

        if not vision_supported:
            import os

            placeholders = []
            for path in media:
                name = path.split("/")[-1] if "/" in path else path
                placeholders.append(f"[Image: {name}]")
            return text + "\n" + "\n".join(placeholders)

        images = []
        for path in media:
            if path.startswith("http://") or path.startswith("https://"):
                result = await self._fetch_image_as_b64(path)
                if isinstance(result, tuple):
                    mime, b64 = result
                    images.append(
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
                    )
                elif result == "invalid_type":
                    images.append(
                        {
                            "type": "text",
                            "text": f"[System: The URL {path} is not an image (e.g. it's a webpage)]",
                        }
                    )
                elif result == "too_large":
                    images.append(
                        {
                            "type": "text",
                            "text": f"[System: The image at {path} is too large (>10MB)]",
                        }
                    )
                else:
                    images.append(
                        {"type": "text", "text": f"[System: Image load failed from {path}]"}
                    )
                continue

            p = Path(path)
            mime, _ = mimetypes.guess_type(path)
            if not p.is_file() or not mime or not mime.startswith("image/"):
                continue

            # Convert WebP and other unsupported formats to PNG for vision models
            image_data = p.read_bytes()
            output_mime = mime
            if mime in ("image/webp", "image/bmp", "image/tiff"):
                try:
                    from PIL import Image
                    import io

                    img = Image.open(io.BytesIO(image_data))
                    if img.mode in ("RGBA", "LA", "P"):
                        # Keep transparency for PNG
                        output_format = "PNG"
                        output_mime = "image/png"
                    else:
                        # Convert to RGB for JPEG
                        output_format = "JPEG"
                        output_mime = "image/jpeg"
                        img = img.convert("RGB")

                    buffer = io.BytesIO()
                    img.save(buffer, format=output_format, quality=95)
                    image_data = buffer.getvalue()
                except Exception as e:
                    logger.warning("Failed to convert {} to PNG/JPEG: {}", path, e)
                    # Fall back to original format

            b64 = base64.b64encode(image_data).decode()
            images.append(
                {"type": "image_url", "image_url": {"url": f"data:{output_mime};base64,{b64}"}}
            )

        if not images:
            return text

        # Preserve [image: path] markers in text for later hydration
        # This allows _save_turn() to extract paths and save them for history
        if media:
            markers = " ".join(f"[image: {p}]" for p in media)
            text_with_markers = f"{markers}\n{text}" if text else markers
        else:
            text_with_markers = text

        return images + [{"type": "text", "text": text_with_markers}]

    async def _hydrate_image_refs(
        self, history: list[dict[str, Any]], vision_supported: bool = True
    ) -> list[dict[str, Any]]:
        """Re-inject base64 image data for [image:path] markers in history."""
        import re
        from urllib.parse import unquote

        from loguru import logger

        _REF_PATTERN = re.compile(r"\[image:\s*(.+?)\]")

        logger.debug(
            "Hydrating history: {} messages, vision_supported={}", len(history), vision_supported
        )

        result = []
        for msg in history:
            content = msg.get("content")
            if msg.get("role") != "user":
                result.append(msg)
                continue

            # Handle string content (e.g., from Telegram)
            if isinstance(content, str):
                matches = _REF_PATTERN.findall(content)
                if not matches:
                    result.append(msg)
                    continue

                if not vision_supported:
                    logger.warning("Vision not supported, skipping image hydration")
                    result.append(msg)
                    continue

                # Hydrate images in string content
                hydrated_parts = []
                last_end = 0
                images_added = []

                for match in _REF_PATTERN.finditer(content):
                    img_path = match.group(1)
                    img_url = None

                    if img_path.startswith("http://") or img_path.startswith("https://"):
                        fetch_result = await self._fetch_image_as_b64(img_path)
                        if isinstance(fetch_result, tuple):
                            mime, b64 = fetch_result
                            img_url = f"data:{mime};base64,{b64}"
                    else:
                        p = Path(unquote(img_path))
                        mime, _ = mimetypes.guess_type(img_path)
                        if p.is_file() and mime and mime.startswith("image/"):
                            b64 = base64.b64encode(p.read_bytes()).decode()
                            img_url = f"data:{mime};base64,{b64}"
                        else:
                            logger.warning("Image not found or invalid MIME: {}", img_path)

                    if img_url:
                        if match.start() > last_end:
                            text_before = content[last_end : match.start()]
                            if text_before.strip():
                                hydrated_parts.append({"type": "text", "text": text_before.strip()})

                        images_added.append({"type": "image_url", "image_url": {"url": img_url}})
                        last_end = match.end()

                if last_end < len(content):
                    remaining = content[last_end:].strip()
                    if remaining:
                        hydrated_parts.append({"type": "text", "text": remaining})

                if images_added:
                    result.append({**msg, "content": hydrated_parts + images_added})
                else:
                    logger.warning("Hydration failed: no images could be loaded")
                    result.append(msg)
                continue

            # Handle list content (multimodal messages)
            if not isinstance(content, list):
                result.append(msg)
                continue

            new_content = []
            changed = False
            for block in content:
                if block.get("type") == "text":
                    text = block.get("text", "")
                    m = _REF_PATTERN.search(text)
                    if m:
                        img_path = m.group(1)
                        img_url = None

                        if not vision_supported:
                            new_content.append({"type": "text", "text": f"[Image: {img_path}]"})
                            changed = True
                            continue

                        if img_path.startswith("http://") or img_path.startswith("https://"):
                            fetch_result = await self._fetch_image_as_b64(img_path)
                            if isinstance(fetch_result, tuple):
                                mime, b64 = fetch_result
                                img_url = f"data:{mime};base64,{b64}"
                            elif fetch_result == "invalid_type":
                                new_content.append(
                                    {
                                        "type": "text",
                                        "text": f"[System: History image {img_path} is not an image]",
                                    }
                                )
                                changed = True
                                continue
                            elif fetch_result == "too_large":
                                new_content.append(
                                    {
                                        "type": "text",
                                        "text": f"[System: History image {img_path} is too large (>10MB)]",
                                    }
                                )
                                changed = True
                                continue
                            else:
                                new_content.append(
                                    {
                                        "type": "text",
                                        "text": f"[System: History image load failed: {img_path}]",
                                    }
                                )
                                changed = True
                                continue
                        else:
                            p = Path(unquote(img_path))
                            mime, _ = mimetypes.guess_type(img_path)
                            if p.is_file() and mime and mime.startswith("image/"):
                                b64 = base64.b64encode(p.read_bytes()).decode()
                                img_url = f"data:{mime};base64,{b64}"
                            else:
                                logger.warning("Image not found or invalid MIME: {}", img_path)
                                continue

                        if img_url:
                            cleaned_text = _REF_PATTERN.sub("", text).strip()
                            if cleaned_text:
                                new_content.append({"type": "text", "text": cleaned_text})

                            new_content.append({"type": "image_url", "image_url": {"url": img_url}})
                            changed = True
                            continue
                elif block.get("type") == "image_url":
                    pass
                new_content.append(block)

            if changed:
                result.append({**msg, "content": new_content})
            else:
                result.append(msg)
        return result

    def add_tool_result(
        self,
        messages: list[dict[str, Any]],
        tool_call_id: str,
        tool_name: str,
        result: str,
    ) -> list[dict[str, Any]]:
        """Add a tool result to the message list. Content MUST be a string for API compliance."""
        messages.append(
            {"role": "tool", "tool_call_id": tool_call_id, "name": tool_name, "content": result}
        )
        return messages

    async def add_user_message(
        self,
        messages: list[dict[str, Any]],
        content: str,
        media: list[str] | None = None,
        vision_supported: bool = True,
    ) -> list[dict[str, Any]]:
        """Add a user message to the message list. Supports multimodal media."""
        # Await the content building since it may involve async image fetching
        user_content = await self._build_user_content(
            content, media, vision_supported=vision_supported
        )
        messages.append({"role": "user", "content": user_content})
        return messages

    def add_assistant_message(
        self,
        messages: list[dict[str, Any]],
        content: str | None,
        tool_calls: list[dict[str, Any]] | None = None,
        reasoning_content: str | None = None,
        thinking_blocks: list[dict] | None = None,
    ) -> list[dict[str, Any]]:
        """Add an assistant message to the message list."""
        msg: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        if reasoning_content is not None:
            msg["reasoning_content"] = reasoning_content
        if thinking_blocks:
            msg["thinking_blocks"] = thinking_blocks
        messages.append(msg)
        return messages
