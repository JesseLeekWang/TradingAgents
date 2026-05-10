import sys
import types

import pytest


if "questionary" not in sys.modules:
    questionary = types.ModuleType("questionary")

    class Choice:
        def __init__(self, title, value=None):
            self.title = title
            self.name = title
            self.value = title if value is None else value

    questionary.Choice = Choice
    questionary.Style = lambda *args, **kwargs: None
    questionary.select = lambda *args, **kwargs: None
    questionary.text = lambda *args, **kwargs: None
    sys.modules["questionary"] = questionary

if "rich.console" not in sys.modules:
    rich = types.ModuleType("rich")
    console_mod = types.ModuleType("rich.console")

    class Console:
        def print(self, *args, **kwargs):
            pass

    console_mod.Console = Console
    sys.modules.setdefault("rich", rich)
    sys.modules["rich.console"] = console_mod


from cli import utils


@pytest.mark.unit
def test_parse_openai_compatible_models_handles_common_shapes():
    payload = {
        "data": [
            {"id": "served-a"},
            {"id": "served-b", "name": "Served B"},
            "served-c",
            {"name": "served-d"},
            {"id": "served-a"},
            {},
        ]
    }

    assert utils._parse_openai_compatible_models(payload) == [
        ("served-a", "served-a"),
        ("Served B", "served-b"),
        ("served-c", "served-c"),
        ("served-d", "served-d"),
    ]


@pytest.mark.unit
def test_fetch_openai_compatible_models_uses_provider_endpoint_and_key(
    monkeypatch,
):
    captured = {}
    requests = types.ModuleType("requests")

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"data": [{"id": "served-model"}]}

    def get(url, headers=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["timeout"] = timeout
        return Response()

    requests.get = get
    monkeypatch.setitem(sys.modules, "requests", requests)
    monkeypatch.setenv("VLLM_API_KEY", "secret")

    assert utils._fetch_openai_compatible_models("vllm") == [
        ("served-model", "served-model")
    ]
    assert captured == {
        "url": "http://localhost:8000/v1/models",
        "headers": {"Authorization": "Bearer secret"},
        "timeout": 10,
    }


@pytest.mark.unit
def test_fetch_openai_compatible_models_uses_custom_base_url(monkeypatch):
    captured = {}
    requests = types.ModuleType("requests")

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"data": [{"id": "proxy-model"}]}

    def get(url, headers=None, timeout=None):
        captured["url"] = url
        return Response()

    requests.get = get
    monkeypatch.setitem(sys.modules, "requests", requests)

    assert utils._fetch_openai_compatible_models(
        "litellm", "http://proxy.local:4100/v1/"
    ) == [("proxy-model", "proxy-model")]
    assert captured["url"] == "http://proxy.local:4100/v1/models"


@pytest.mark.unit
def test_select_openai_compatible_base_url_accepts_custom_url(monkeypatch):
    class Prompt:
        def __init__(self, value):
            self.value = value

        def ask(self):
            return self.value

    def select(message, choices, **kwargs):
        return Prompt("custom")

    def text(message, **kwargs):
        assert kwargs["default"] == "http://localhost:4000/v1"
        assert kwargs["validate"]("https://llm.example.com/v1") is True
        return Prompt("https://llm.example.com/v1/")

    monkeypatch.setattr(utils.questionary, "select", select, raising=False)
    monkeypatch.setattr(utils.questionary, "text", text, raising=False)

    assert (
        utils._select_openai_compatible_base_url(
            "litellm", "http://localhost:4000/v1"
        )
        == "https://llm.example.com/v1"
    )


@pytest.mark.unit
def test_select_openai_compatible_model_uses_discovered_models(monkeypatch):
    captured = {}

    class Prompt:
        def ask(self):
            return "served-b"

    def select(message, choices, **kwargs):
        captured["message"] = message
        captured["choices"] = [(choice.title, choice.value) for choice in choices]
        return Prompt()

    monkeypatch.setattr(
        utils,
        "_fetch_openai_compatible_models",
        lambda provider, base_url=None: [
            ("Served A", "served-a"),
            ("Served B", "served-b"),
        ],
    )
    monkeypatch.setattr(utils.questionary, "select", select, raising=False)

    assert (
        utils.select_openai_compatible_model(
            "vllm", "quick", "http://custom.local:8001/v1/"
        )
        == "served-b"
    )
    assert captured["message"] == (
        "Select vLLM Model (available from http://custom.local:8001/v1):"
    )
    assert captured["choices"] == [
        ("Served A", "served-a"),
        ("Served B", "served-b"),
        ("Custom model ID", "custom"),
    ]
