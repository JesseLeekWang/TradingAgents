import sys
import types
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

_HELPER_PATH = (
    Path(__file__).resolve().parents[1]
    / "tradingagents"
    / "agents"
    / "utils"
    / "tool_call_compat.py"
)
_SPEC = importlib.util.spec_from_file_location("tool_call_compat", _HELPER_PATH)
tool_call_compat = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(tool_call_compat)

coerce_plain_text_tool_calls = tool_call_compat.coerce_plain_text_tool_calls
parse_plain_text_tool_calls = tool_call_compat.parse_plain_text_tool_calls


@pytest.mark.unit
def test_parse_plain_text_tool_calls_from_local_model_markup():
    content = """
    I'll fetch the data.
    <tool_call>
    <function=get_stock_data>
    <parameter=symbol>
    SOX
    </parameter>
    <parameter=start_date>
    2026-02-01
    </parameter>
    <parameter=end_date>
    2026-05-09
    </parameter>
    </function>
    </tool_call>
    """

    assert parse_plain_text_tool_calls(content, {"get_stock_data"}) == [
        {
            "name": "get_stock_data",
            "args": {
                "symbol": "SOX",
                "start_date": "2026-02-01",
                "end_date": "2026-05-09",
            },
            "id": "plain_text_tool_call_0",
            "type": "tool_call",
        }
    ]


@pytest.mark.unit
def test_parse_plain_text_tool_calls_filters_unknown_tools():
    content = """
    <tool_call>
    <function=unknown_tool>
    <parameter=ticker>SOX</parameter>
    </function>
    </tool_call>
    <tool_call>
    <function=get_news>
    <parameter=ticker>SOX</parameter>
    </function>
    </tool_call>
    """

    assert parse_plain_text_tool_calls(content, {"get_news"}) == [
        {
            "name": "get_news",
            "args": {"ticker": "SOX"},
            "id": "plain_text_tool_call_0",
            "type": "tool_call",
        }
    ]


@pytest.mark.unit
def test_parse_plain_text_tool_calls_from_qwen_json_markup():
    content = """
    <tool_call>
    {"name": "get_news", "arguments": {"ticker": "SOX", "start_date": "2026-05-02", "end_date": "2026-05-09"}}
    </tool_call>
    """

    assert parse_plain_text_tool_calls(content, {"get_news"}) == [
        {
            "name": "get_news",
            "args": {
                "ticker": "SOX",
                "start_date": "2026-05-02",
                "end_date": "2026-05-09",
            },
            "id": "plain_text_tool_call_0",
            "type": "tool_call",
        }
    ]


@pytest.mark.unit
def test_parse_plain_text_tool_calls_from_qwen_json_string_arguments():
    payload = {
        "function": "get_stock_data",
        "arguments": json.dumps(
            {
                "symbol": "SOX",
                "start_date": "2026-02-01",
                "end_date": "2026-05-09",
            }
        ),
    }
    content = f"<tool_call>{json.dumps(payload)}</tool_call>"

    assert parse_plain_text_tool_calls(content, {"get_stock_data"}) == [
        {
            "name": "get_stock_data",
            "args": {
                "symbol": "SOX",
                "start_date": "2026-02-01",
                "end_date": "2026-05-09",
            },
            "id": "plain_text_tool_call_0",
            "type": "tool_call",
        }
    ]


@pytest.mark.unit
def test_coerce_plain_text_tool_calls_returns_ai_message(monkeypatch):
    langchain_core = types.ModuleType("langchain_core")
    messages = types.ModuleType("langchain_core.messages")

    class AIMessage:
        def __init__(self, content, tool_calls):
            self.content = content
            self.tool_calls = tool_calls

    messages.AIMessage = AIMessage
    monkeypatch.setitem(sys.modules, "langchain_core", langchain_core)
    monkeypatch.setitem(sys.modules, "langchain_core.messages", messages)

    message = SimpleNamespace(
        content=(
            "<tool_call><function=get_news>"
            "<parameter=ticker>SOX</parameter>"
            "<parameter=start_date>2026-05-02</parameter>"
            "<parameter=end_date>2026-05-09</parameter>"
            "</function></tool_call>"
        ),
        tool_calls=[],
    )
    tools = [SimpleNamespace(name="get_news")]

    coerced = coerce_plain_text_tool_calls(message, tools)

    assert isinstance(coerced, AIMessage)
    assert coerced.content == ""
    assert coerced.tool_calls[0]["name"] == "get_news"
    assert coerced.tool_calls[0]["args"]["ticker"] == "SOX"
