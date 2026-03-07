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

    def build_messages(
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
        user_content = self._build_user_content(current_message, media)

        # Merge runtime context and user content into a single user message
        # to avoid consecutive same-role messages that some providers reject.
        if isinstance(user_content, str):
            merged = f"{runtime_ctx}\n\n{user_content}"
        else:
            merged = [{"type": "text", "text": runtime_ctx}] + user_content

        # Re-hydrate image references in history so the model can "see" them again
        hydrated_history = self._hydrate_image_refs(history)

        return [
            {"role": "system", "content": self.build_system_prompt(skill_names)},
            *hydrated_history,
            {"role": "user", "content": merged},
        ]

    def _build_user_content(self, text: str, media: list[str] | None) -> str | list[dict[str, Any]]:
        """Build user message content with optional base64-encoded images."""
        if not media:
            return text

        images = []
        for path in media:
            p = Path(path)
            mime, _ = mimetypes.guess_type(path)
            if not p.is_file() or not mime or not mime.startswith("image/"):
                continue
            b64 = base64.b64encode(p.read_bytes()).decode()
            images.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})

        if not images:
            return text
        return images + [{"type": "text", "text": text}]

    @staticmethod
    def _hydrate_image_refs(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Re-inject base64 image data for [image_ref:path] markers in history."""
        import re
        _REF_PATTERN = re.compile(r'^\[image_ref:(.+)\]$')

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
                    m = _REF_PATTERN.match(block.get("text", ""))
                    if m:
                        img_path = m.group(1)
                        p = Path(img_path)
                        mime, _ = mimetypes.guess_type(img_path)
                        if p.is_file() and mime and mime.startswith("image/"):
                            b64 = base64.b64encode(p.read_bytes()).decode()
                            new_content.append({
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime};base64,{b64}"}
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
        result: str | list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Add a tool result to the message list. Supports multimodal [image: path] tags."""
        import base64
        import mimetypes
        import re

        content: Any = result

        if isinstance(result, str):
            # Detect [image: path] markers
            pattern = re.compile(r"\[image:\s*(.+?)\]")
            matches = list(pattern.finditer(result))
            
            if matches:
                content_blocks = []
                last_pos = 0
                for match in matches:
                    # Add preceding text
                    txt = result[last_pos:match.start()].strip()
                    if txt:
                        content_blocks.append({"type": "text", "text": txt})
                    
                    # Add image block
                    img_path = match.group(1).strip()
                    p = Path(img_path)
                    if p.is_file():
                        mime, _ = mimetypes.guess_type(img_path)
                        mime = mime or "image/png"
                        if mime.startswith("image/"):
                            try:
                                b64 = base64.b64encode(p.read_bytes()).decode()
                                content_blocks.append({
                                    "type": "image_url",
                                    "image_url": {"url": f"data:{mime};base64,{b64}"}
                                })
                            except Exception as e:
                                content_blocks.append({"type": "text", "text": f"(Error loading image: {e})"})
                        else:
                            content_blocks.append({"type": "text", "text": f"(Error: File is not an image: {img_path})"})
                    else:
                        content_blocks.append({"type": "text", "text": f"(Error: Image not found: {img_path})"})
                    
                    last_pos = match.end()
                
                # Add remaining text
                rem = result[last_pos:].strip()
                if rem:
                    content_blocks.append({"type": "text", "text": rem})
                
                content = content_blocks

        messages.append(
            {"role": "tool", "tool_call_id": tool_call_id, "name": tool_name, "content": content}
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
