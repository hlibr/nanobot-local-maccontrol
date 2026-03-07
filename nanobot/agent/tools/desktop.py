"""macOS-native desktop control tools."""

import asyncio
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from loguru import logger
from nanobot.agent.tools.base import Tool


class AppleScriptTool(Tool):
    """Tool to execute arbitrary AppleScript."""

    @property
    def name(self) -> str:
        return "applescript"

    @property
    def description(self) -> str:
        return (
            "Execute an AppleScript and return its output. "
            "Use this to control macOS system settings and applications "
            "(e.g., Music, Calendar, Finder, Safari, System Settings)."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "script": {
                    "type": "string",
                    "description": "The AppleScript code to execute"
                }
            },
            "required": ["script"]
        }

    async def execute(self, script: str, **kwargs: Any) -> str:
        try:
            # osascript -e "script code"
            process = await asyncio.create_subprocess_exec(
                "osascript", "-e", script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                err = stderr.decode().strip()
                return f"Error: AppleScript failed with exit code {process.returncode}: {err}"
            
            return stdout.decode().strip() or "Success (no output)"
        except Exception as e:
            return f"Error executing AppleScript: {str(e)}"


class CaptureScreenTool(Tool):
    """Tool to capture the current screen."""

    def __init__(self, media_dir: Path | None = None):
        self.media_dir = media_dir or Path.home() / ".nanobot" / "media"
        self.media_dir.mkdir(parents=True, exist_ok=True)

    @property
    def name(self) -> str:
        return "capture_screen"

    @property
    def description(self) -> str:
        return "Take a screenshot of the main display and return the file path."

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs: Any) -> str:
        try:
            # Use native 'screencapture' tool on macOS
            import time
            timestamp = int(time.time() * 1000)
            file_path = self.media_dir / f"screenshot_{timestamp}.png"
            
            # -x flag: silent (no shutter sound)
            # screencapture -x path.png
            process = await asyncio.create_subprocess_exec(
                "screencapture", "-x", str(file_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()
            
            if not file_path.exists():
                return "Error: Failed to capture screen (check Screen Recording permissions)."
            
            # The AgentLoop will see this path and re-hydrate it into vision context
            return f"[image: {file_path}]"
        except Exception as e:
            return f"Error capturing screen: {str(e)}"


class DesktopUIMetadataTool(Tool):
    """Tool to extract the UI Accessibility Tree (metadata) of the screen."""

    @property
    def name(self) -> str:
        return "get_ui_metadata"

    @property
    def description(self) -> str:
        return (
            "Extract semantic information (Window name, element roles, names, and pixel coordinates) "
            "from the frontmost window. Use this to determine where to click or type."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs: Any) -> str:
        # A more focused script that only retrieves 'interactable' elements to save tokens
        script = """
        tell application "System Events"
            set frontProcess to first application process whose frontmost is true
            set processName to name of frontProcess
            set elementSummary to ""
            
            try
                set win to window 1 of frontProcess
                set winPos to position of win
                set winSize to size of win
                set elementSummary to "Process: " & processName & " | Window: " & (name of win) & " | Window Rect: [" & (item 1 of winPos) & "," & (item 2 of winPos) & "," & (item 1 of winSize) & "," & (item 2 of winSize) & "]" & linefeed
                
                # Get common interactable elements: buttons, text fields, menu buttons, etc.
                set roleList to {"AXButton", "AXTextField", "AXTextArea", "AXCheckBox", "AXRadioButton", "AXPopUpButton", "AXMenuButton", "AXStaticText", "AXImage", "AXLink", "AXList", "AXRow", "AXCell", "AXScrollArea", "AXWebArea"}
                
                set uiElements to entire contents of win
                repeat with anElement in uiElements
                    try
                        set theRole to role of anElement
                        set theName to name of anElement
                        if (theRole is in roleList) then
                            set thePos to position of anElement
                            set theSize to size of anElement
                            set elementSummary to elementSummary & "- " & theRole & ": '" & theName & "' at [" & (item 1 of thePos) & "," & (item 2 of thePos) & "], size [" & (item 1 of theSize) & "," & (item 2 of theSize) & "]" & linefeed
                        end if
                    end try
                end repeat
            on error err
                return "Error: " & err
            end try
            
            return elementSummary
        end tell
        """
        try:
            process = await asyncio.create_subprocess_exec(
                "osascript", "-e", script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            if process.returncode != 0:
                return f"Error: {stderr.decode().strip()}"
            return stdout.decode().strip() or "No interactable UI elements found."
        except Exception as e:
            return f"Error: {str(e)}"


class DesktopActionTool(Tool):
    """Tool to perform physical mouse/keyboard actions."""

    def __init__(self, send_callback=None):
        self.send_callback = send_callback

    @property
    def name(self) -> str:
        return "desktop_action"

    @property
    def description(self) -> str:
        return "Click, double-click, type, or enter a key combination on the macOS desktop."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "Action to perform",
                    "enum": ["click", "double_click", "right_click", "type", "key_combo"]
                },
                "params": {
                    "type": "object",
                    "description": "Parameters for the action (e.g., {x: 100, y: 200} for click, {text: 'hi'} for type, {keys: 'command+space'} for key_combo)",
                    "properties": {
                        "x": {"type": "integer"},
                        "y": {"type": "integer"},
                        "text": {"type": "string"},
                        "keys": {"type": "string"}
                    }
                }
            },
            "required": ["action", "params"]
        }

    async def execute(self, action: str, params: dict[str, Any], **kwargs: Any) -> str:
        try:
            script = ""
            if action == "click":
                x, y = params.get("x"), params.get("y")
                script = f'tell application "System Events" to click at {{{x}, {y}}}'
            elif action == "double_click":
                # Double click is trickier in raw AppleScript, often requires System Events keystroke or using physical mouse via shell
                # For simplicity, we trigger two clicks in a sequence or a double click via shell
                x, y = params.get("x"), params.get("y")
                # Using python for double click is more reliable here, but we promised lightweight AppleScript first
                script = f'tell application "System Events" to click at {{{x}, {y}}}\ndelay 0.1\ntell application "System Events" to click at {{{x}, {y}}}'
            elif action == "type":
                text = params.get("text", "").replace('"', '\\"')
                script = f'tell application "System Events" to keystroke "{text}"'
            elif action == "key_combo":
                # e.g. "command+space" -> "keystroke space using {command down}"
                keys = params.get("keys", "").lower().split("+")
                if len(keys) > 1:
                    main_key = keys[-1]
                    modifiers = "{" + ", ".join(f"{k} down" for k in keys[:-1]) + "}"
                    script = f'tell application "System Events" to keystroke "{main_key}" using {modifiers}'
                else:
                    script = f'tell application "System Events" to keystroke "{keys[0]}"'

            if not script:
                return "Error: Invalid action or missing parameters."

            process = await asyncio.create_subprocess_exec(
                "osascript", "-e", script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            if process.returncode != 0:
                err = stderr.decode().strip()
                return f"Error: {err}"
            
            return f"Successfully performed {action}."
        except Exception as e:
            return f"Error: {str(e)}"


class ViewImageTool(Tool):
    """Tool to view an image file and understand its content."""

    @property
    def name(self) -> str:
        return "view_image"

    @property
    def description(self) -> str:
        return "View an image file from the local filesystem. Returns the image for visual analysis by the agent."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to the image file"
                }
            },
            "required": ["path"]
        }

    async def execute(self, path: str, **kwargs: Any) -> str:
        try:
            if path.startswith("http://") or path.startswith("https://"):
                # Basic check to see if it looks like a webpage instead of an image
                ext = path.split("/")[-1].split("?")[0].split("#")[0].lower()
                if "." in ext:
                    suffix = "." + ext.split(".")[-1]
                    if suffix in (".php", ".html", ".htm", ".js", ".css", ".jsp", ".asp", ".aspx"):
                        return f"Error: URL appears to be a webpage, not an image (detected extension: {suffix})"
                
                # Always allow URLs to be viewed if they don't look like code/webpages
                return f"[image: {path}]"

            p = Path(path)
            if not p.is_file():
                return f"Error: File not found at {path}"
            
            import mimetypes
            mime, _ = mimetypes.guess_type(path)
            if not mime or not mime.startswith("image/"):
                # Check if it has a common image extension even if mime guess failed
                ext = p.suffix.lower()
                if ext not in (".png", ".jpg", ".jpeg", ".webp", ".gif"):
                    return f"Error: File is not an image (Mime: {mime}, Extension: {ext})"
            
            # The AgentLoop will see this path and re-hydrate it into vision context
            return f"[image: {path}]"
        except Exception as e:
            return f"Error viewing image: {str(e)}"


class SendImageTool(Tool):
    """Tool to send an image to the user."""

    def __init__(self, send_callback=None):
        self._send_callback = send_callback
        self._default_channel = ""
        self._default_chat_id = ""

    def set_context(self, channel: str, chat_id: str, **kwargs) -> None:
        self._default_channel = channel
        self._default_chat_id = chat_id

    @property
    def name(self) -> str:
        return "send_image"

    @property
    def description(self) -> str:
        return "Send an image to the user on Telegram. The 'path' can be an absolute local filesystem path OR a direct URL (http/https)."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to the image file"
                },
                "caption": {
                    "type": "string",
                    "description": "Optional caption for the image"
                }
            },
            "required": ["path"]
        }

    async def execute(self, path: str, caption: str = "", **kwargs: Any) -> str:
        if not self._send_callback:
            return "Error: Send callback not configured"
        
        from nanobot.bus.events import OutboundMessage
        msg = OutboundMessage(
            channel=self._default_channel,
            chat_id=self._default_chat_id,
            content=caption or "Image attached",
            media=[path]
        )
        await self._send_callback(msg)
        return f"Image sent to user: {path}"
