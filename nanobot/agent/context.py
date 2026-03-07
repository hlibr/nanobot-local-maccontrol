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
    ) -> list[dict[str, Any]]:
        """Build the complete message list for an LLM call."""
        runtime_ctx = self._build_runtime_context(channel, chat_id)
        
        # Extract [image: path] tags from the message text and add to media list
        import re
        extracted_media = re.findall(r'\[image:\s*(.+?)\]', current_message)
        clean_message = current_message
        if extracted_media:
            media = (media or []) + [m.strip() for m in extracted_media]
            # Remove the tags from the text so the LLM doesn't get confused by them
            clean_message = re.sub(r'\[image:\s*.+?\]', '', current_message).strip()

        user_content = await self._build_user_content(clean_message, media)

        # Merge runtime context and user content into a single user message
        # to avoid consecutive same-role messages that some providers reject.
        if isinstance(user_content, str):
            merged = f"{runtime_ctx}\n\n{user_content}"
        else:
            merged = [{"type": "text", "text": runtime_ctx}] + user_content

        # Re-hydrate image references in history so the model can "see" them again
        hydrated_history = await self._hydrate_image_refs(history)

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
                
                mime = resp.headers.get("Content-Type", "")
                if not mime.startswith("image/"):
                    # Try to guess from URL if header is missing/generic
                    mime, _ = mimetypes.guess_type(url)
                    if not mime or not mime.startswith("image/"):
                        mime = "image/jpeg" # Fallback
                
                data = resp.content
                if len(data) > 10 * 1024 * 1024:
                    logger.warning("Image from {} is too large ({} bytes)", url, len(data))
                    return None
                
                b64 = base64.b64encode(data).decode()
                return mime, b64
        except Exception as e:
            logger.warning("Error fetching image from {}: {}", url, e)
            return None

    async def _build_user_content(self, text: str, media: list[str] | None) -> str | list[dict[str, Any]]:
        """Build user message content with base64-encoded images. URLs are fetched and converted."""
        if not media:
            return text

        images = []
        for path in media:
            if path.startswith("http://") or path.startswith("https://"):
                result = await self._fetch_image_as_b64(path)
                if result:
                    mime, b64 = result
                    images.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
                else:
                    images.append({"type": "text", "text": f"[System: Image load failed from {path}]"})
                continue

            p = Path(path)
            mime, _ = mimetypes.guess_type(path)
            if not p.is_file() or not mime or not mime.startswith("image/"):
                continue
            b64 = base64.b64encode(p.read_bytes()).decode()
            images.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})

        if not images:
            return text
        return images + [{"type": "text", "text": text}]

    async def _hydrate_image_refs(self, history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Re-inject base64 image data for [image:path] markers in history."""
        import re
        _REF_PATTERN = re.compile(r'\[image:\s*(.+?)\]')

        result = []
        for msg in history:
            content = msg.get("content")
            if msg.get("role") != "user" or not isinstance(content, list):
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
                        img_data = None
                        
                        if img_path.startswith("http://") or img_path.startswith("https://"):
                            img_data = await self._fetch_image_as_b64(img_path)
                            if img_data:
                                mime, b64 = img_data
                                img_url = f"data:{mime};base64,{b64}"
                            else:
                                # Keep the text error node
                                new_content.append({"type": "text", "text": f"[System: History image load failed: {img_path}]"})
                                changed = True
                                continue
                        else:
                            p = Path(img_path)
                            mime, _ = mimetypes.guess_type(img_path)
                            if p.is_file() and mime and mime.startswith("image/"):
                                b64 = base64.b64encode(p.read_bytes()).decode()
                                img_url = f"data:{mime};base64,{b64}"
                            else:
                                continue

                        if img_url:
                            # Filter out the tag from the text block
                            cleaned_text = _REF_PATTERN.sub('', text).strip()
                            if cleaned_text:
                                new_content.append({"type": "text", "text": cleaned_text})
                            
                            new_content.append({
                                "type": "image_url",
                                "image_url": {"url": img_url}
                            })
                            changed = True
                            continue
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
    ) -> list[dict[str, Any]]:
        """Add a user message to the message list. Supports multimodal media."""
        # Await the content building since it may involve async image fetching
        user_content = await self._build_user_content(content, media)
        messages.append(
            {"role": "user", "content": user_content}
        )
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
