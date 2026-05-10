"""Compatibility helpers for providers that emit tool calls as text.

Some OpenAI-compatible local servers accept a tool schema but weaker models
still answer with plain text such as::

    <tool_call>
    <function=get_news>
    <parameter=ticker>NVDA</parameter>
    </function>
    </tool_call>

LangGraph only routes through ``ToolNode`` when the assistant message carries
real ``tool_calls`` metadata. These helpers translate that common text format
into a normal ``AIMessage`` so the existing graph can execute the tools and
ask the analyst to write the final report.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable

_FUNCTION_RE = re.compile(
    r"<function=(?P<name>[A-Za-z_][\w]*)>\s*(?P<body>.*?)\s*</function>",
    re.DOTALL,
)
_PARAMETER_RE = re.compile(
    r"<parameter=(?P<name>[A-Za-z_][\w]*)>\s*(?P<value>.*?)\s*</parameter>",
    re.DOTALL,
)
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(?P<body>.*?)\s*</tool_call>", re.DOTALL)


def _content_to_text(content: Any) -> str:
    """Return text from common chat-message content shapes."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts)
    return ""


def parse_plain_text_tool_calls(
    content: Any, allowed_tool_names: Iterable[str] | None = None
) -> list[dict[str, Any]]:
    """Parse text-form tool calls into LangChain-style tool call dictionaries."""
    text = _content_to_text(content)
    if "<function=" not in text and "<tool_call>" not in text:
        return []

    allowed = set(allowed_tool_names or [])
    tool_calls = []
    for match in _FUNCTION_RE.finditer(text):
        name = match.group("name").strip()
        if allowed and name not in allowed:
            continue

        args = {
            param.group("name").strip(): param.group("value").strip()
            for param in _PARAMETER_RE.finditer(match.group("body"))
        }
        tool_calls.append(
            {
                "name": name,
                "args": args,
                "id": f"plain_text_tool_call_{len(tool_calls)}",
                "type": "tool_call",
            }
        )

    for match in _TOOL_CALL_RE.finditer(text):
        body = match.group("body").strip()
        if "<function=" in body:
            continue

        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            continue

        name = (
            payload.get("name")
            or payload.get("tool")
            or payload.get("function")
            or payload.get("function_name")
        )
        if not name or (allowed and name not in allowed):
            continue

        args = (
            payload.get("args")
            or payload.get("arguments")
            or payload.get("parameters")
            or {}
        )
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        if not isinstance(args, dict):
            args = {}

        tool_calls.append(
            {
                "name": name,
                "args": args,
                "id": f"plain_text_tool_call_{len(tool_calls)}",
                "type": "tool_call",
            }
        )

    return tool_calls


def coerce_plain_text_tool_calls(message: Any, tools: Iterable[Any]) -> Any:
    """Convert text-form tool calls on ``message`` into real tool calls.

    Messages that already have provider-native tool calls, or that do not
    contain parseable tool-call markup, are returned unchanged.
    """
    if getattr(message, "tool_calls", None):
        return message

    allowed_tool_names = [tool.name for tool in tools]
    tool_calls = parse_plain_text_tool_calls(
        getattr(message, "content", ""), allowed_tool_names
    )
    if not tool_calls:
        return message

    from langchain_core.messages import AIMessage

    return AIMessage(content="", tool_calls=tool_calls)
