import importlib.util
from importlib.machinery import ModuleSpec
import sys
import types
from unittest.mock import patch

import pytest


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


if not _has_module("langchain_core.messages"):
    langchain_core = types.ModuleType("langchain_core")
    langchain_core.__path__ = []
    langchain_core.__spec__ = ModuleSpec(
        "langchain_core", loader=None, is_package=True
    )
    messages = types.ModuleType("langchain_core.messages")
    messages.__spec__ = ModuleSpec("langchain_core.messages", loader=None)

    class AIMessage:
        pass

    messages.AIMessage = AIMessage
    langchain_core.messages = messages
    sys.modules.setdefault("langchain_core", langchain_core)
    sys.modules["langchain_core.messages"] = messages

if not _has_module("langchain_openai"):
    langchain_openai = types.ModuleType("langchain_openai")
    langchain_openai.__spec__ = ModuleSpec("langchain_openai", loader=None)

    class ChatOpenAI:
        pass

    langchain_openai.ChatOpenAI = ChatOpenAI
    sys.modules["langchain_openai"] = langchain_openai

from tradingagents.llm_clients.factory import create_llm_client
from tradingagents.llm_clients.openai_client import OpenAIClient


@pytest.mark.unit
@pytest.mark.parametrize(
    ("provider", "default_base_url", "env_var", "fallback_key"),
    [
        ("vllm", "http://localhost:8000/v1", "VLLM_API_KEY", "vllm"),
        ("litellm", "http://localhost:4000/v1", "LITELLM_API_KEY", "litellm"),
    ],
)
def test_local_openai_compatible_providers_use_defaults(
    monkeypatch, provider, default_base_url, env_var, fallback_key
):
    monkeypatch.delenv(env_var, raising=False)

    with patch(
        "tradingagents.llm_clients.openai_client.NormalizedChatOpenAI"
    ) as chat_cls:
        client = create_llm_client(provider, "served-model")
        assert isinstance(client, OpenAIClient)

        client.get_llm()

    call_kwargs = chat_cls.call_args[1]
    assert call_kwargs["model"] == "served-model"
    assert call_kwargs["base_url"] == default_base_url
    assert call_kwargs["api_key"] == fallback_key


@pytest.mark.unit
@pytest.mark.parametrize(
    ("provider", "env_var"),
    [
        ("vllm", "VLLM_API_KEY"),
        ("litellm", "LITELLM_API_KEY"),
    ],
)
def test_local_openai_compatible_providers_use_env_api_keys(
    monkeypatch, provider, env_var
):
    monkeypatch.setenv(env_var, "secret-key")

    with patch(
        "tradingagents.llm_clients.openai_client.NormalizedChatOpenAI"
    ) as chat_cls:
        create_llm_client(provider, "served-model").get_llm()

    assert chat_cls.call_args[1]["api_key"] == "secret-key"
