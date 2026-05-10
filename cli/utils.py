import os
from typing import Any, Dict, List, Tuple

import questionary

from rich.console import Console

from cli.models import AnalystType
from tradingagents.llm_clients.model_catalog import get_model_options

console = Console()

TICKER_INPUT_EXAMPLES = "Examples: SPY, CNC.TO, 7203.T, 0700.HK"

# (display_name, provider_key, base_url)
PROVIDERS = [
    ("OpenAI", "openai", "https://api.openai.com/v1"),
    ("Google", "google", None),
    ("Anthropic", "anthropic", "https://api.anthropic.com/"),
    ("xAI", "xai", "https://api.x.ai/v1"),
    ("DeepSeek", "deepseek", "https://api.deepseek.com"),
    ("Qwen", "qwen", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    ("GLM", "glm", "https://open.bigmodel.cn/api/paas/v4/"),
    ("OpenRouter", "openrouter", "https://openrouter.ai/api/v1"),
    ("Azure OpenAI", "azure", None),
    ("Ollama", "ollama", "http://localhost:11434/v1"),
    ("vLLM", "vllm", "http://localhost:8000/v1"),
    ("LiteLLM", "litellm", "http://localhost:4000/v1"),
]

_OPENAI_COMPATIBLE_MODEL_DISCOVERY = {
    "vllm": ("vLLM", "http://localhost:8000/v1", "VLLM_API_KEY"),
    "litellm": ("LiteLLM", "http://localhost:4000/v1", "LITELLM_API_KEY"),
}

ANALYST_ORDER = [
    ("Market Analyst", AnalystType.MARKET),
    ("Social Media Analyst", AnalystType.SOCIAL),
    ("News Analyst", AnalystType.NEWS),
    ("Fundamentals Analyst", AnalystType.FUNDAMENTALS),
]


def get_ticker() -> str:
    """Prompt the user to enter a ticker symbol."""
    ticker = questionary.text(
        f"Enter the exact ticker symbol to analyze ({TICKER_INPUT_EXAMPLES}):",
        validate=lambda x: len(x.strip()) > 0 or "Please enter a valid ticker symbol.",
        style=questionary.Style(
            [
                ("text", "fg:green"),
                ("highlighted", "noinherit"),
            ]
        ),
    ).ask()

    if not ticker:
        console.print("\n[red]No ticker symbol provided. Exiting...[/red]")
        exit(1)

    return normalize_ticker_symbol(ticker)


def normalize_ticker_symbol(ticker: str) -> str:
    """Normalize ticker input while preserving exchange suffixes."""
    return ticker.strip().upper()


def get_analysis_date() -> str:
    """Prompt the user to enter a date in YYYY-MM-DD format."""
    import re
    from datetime import datetime

    def validate_date(date_str: str) -> bool:
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
            return False
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
            return True
        except ValueError:
            return False

    date = questionary.text(
        "Enter the analysis date (YYYY-MM-DD):",
        validate=lambda x: validate_date(x.strip())
        or "Please enter a valid date in YYYY-MM-DD format.",
        style=questionary.Style(
            [
                ("text", "fg:green"),
                ("highlighted", "noinherit"),
            ]
        ),
    ).ask()

    if not date:
        console.print("\n[red]No date provided. Exiting...[/red]")
        exit(1)

    return date.strip()


def select_analysts() -> List[AnalystType]:
    """Select analysts using an interactive checkbox."""
    choices = questionary.checkbox(
        "Select Your [Analysts Team]:",
        choices=[
            questionary.Choice(display, value=value) for display, value in ANALYST_ORDER
        ],
        instruction="\n- Press Space to select/unselect analysts\n- Press 'a' to select/unselect all\n- Press Enter when done",
        validate=lambda x: len(x) > 0 or "You must select at least one analyst.",
        style=questionary.Style(
            [
                ("checkbox-selected", "fg:green"),
                ("selected", "fg:green noinherit"),
                ("highlighted", "noinherit"),
                ("pointer", "noinherit"),
            ]
        ),
    ).ask()

    if not choices:
        console.print("\n[red]No analysts selected. Exiting...[/red]")
        exit(1)

    return choices


def select_research_depth() -> int:
    """Select research depth using an interactive selection."""

    # Define research depth options with their corresponding values
    DEPTH_OPTIONS = [
        ("Shallow - Quick research, few debate and strategy discussion rounds", 1),
        ("Medium - Middle ground, moderate debate rounds and strategy discussion", 3),
        ("Deep - Comprehensive research, in depth debate and strategy discussion", 5),
    ]

    choice = questionary.select(
        "Select Your [Research Depth]:",
        choices=[
            questionary.Choice(display, value=value) for display, value in DEPTH_OPTIONS
        ],
        instruction="\n- Use arrow keys to navigate\n- Press Enter to select",
        style=questionary.Style(
            [
                ("selected", "fg:yellow noinherit"),
                ("highlighted", "fg:yellow noinherit"),
                ("pointer", "fg:yellow noinherit"),
            ]
        ),
    ).ask()

    if choice is None:
        console.print("\n[red]No research depth selected. Exiting...[/red]")
        exit(1)

    return choice


def _fetch_openrouter_models() -> List[Tuple[str, str]]:
    """Fetch available models from the OpenRouter API."""
    import requests
    try:
        resp = requests.get("https://openrouter.ai/api/v1/models", timeout=10)
        resp.raise_for_status()
        models = resp.json().get("data", [])
        return [(m.get("name") or m["id"], m["id"]) for m in models]
    except Exception as e:
        console.print(f"\n[yellow]Could not fetch OpenRouter models: {e}[/yellow]")
        return []


def select_openrouter_model() -> str:
    """Select an OpenRouter model from the newest available, or enter a custom ID."""
    models = _fetch_openrouter_models()

    choices = [questionary.Choice(name, value=mid) for name, mid in models[:5]]
    choices.append(questionary.Choice("Custom model ID", value="custom"))

    choice = questionary.select(
        "Select OpenRouter Model (latest available):",
        choices=choices,
        instruction="\n- Use arrow keys to navigate\n- Press Enter to select",
        style=questionary.Style([
            ("selected", "fg:magenta noinherit"),
            ("highlighted", "fg:magenta noinherit"),
            ("pointer", "fg:magenta noinherit"),
        ]),
    ).ask()

    if choice is None or choice == "custom":
        return questionary.text(
            "Enter OpenRouter model ID (e.g. google/gemma-4-26b-a4b-it):",
            validate=lambda x: len(x.strip()) > 0 or "Please enter a model ID.",
        ).ask().strip()

    return choice


def _parse_openai_compatible_models(payload: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Parse OpenAI-compatible /models responses into display/value pairs."""
    parsed = []
    seen = set()
    for model in payload.get("data", []):
        if isinstance(model, str):
            model_id = model
            display = model
        elif isinstance(model, dict):
            model_id = model.get("id") or model.get("name")
            display = model.get("name") or model_id
        else:
            continue

        if not model_id or model_id in seen:
            continue

        seen.add(model_id)
        parsed.append((display, model_id))

    return parsed


def _normalize_base_url(base_url: str) -> str:
    """Normalize user-entered OpenAI-compatible base URLs."""
    return base_url.strip().rstrip("/")


def _validate_base_url(value: str):
    """Return True for valid HTTP(S) base URLs, or a questionary error string."""
    url = value.strip()
    if url.startswith(("http://", "https://")) and len(url.split("://", 1)[1]) > 0:
        return True
    return "Please enter a base URL starting with http:// or https://."


def _select_openai_compatible_base_url(provider: str, default_base_url: str) -> str:
    """Let users keep the default local endpoint or type a custom base URL."""
    provider_lower = provider.lower()
    display_name, _, _ = _OPENAI_COMPATIBLE_MODEL_DISCOVERY[provider_lower]

    choice = questionary.select(
        f"Select {display_name} API endpoint:",
        choices=[
            questionary.Choice(
                f"Use default ({default_base_url})",
                value=default_base_url,
            ),
            questionary.Choice("Enter custom base URL", value="custom"),
        ],
        instruction="\n- Use arrow keys to navigate\n- Press Enter to select",
        style=questionary.Style([
            ("selected", "fg:magenta noinherit"),
            ("highlighted", "fg:magenta noinherit"),
            ("pointer", "fg:magenta noinherit"),
        ]),
    ).ask()

    if choice is None:
        console.print(f"\n[red]No {display_name} endpoint selected. Exiting...[/red]")
        exit(1)

    if choice == "custom":
        custom_url = questionary.text(
            f"Enter {display_name} base URL:",
            default=default_base_url,
            validate=_validate_base_url,
        ).ask()

        if not custom_url:
            console.print(f"\n[red]No {display_name} base URL provided. Exiting...[/red]")
            exit(1)

        return _normalize_base_url(custom_url)

    return _normalize_base_url(choice)


def _fetch_openai_compatible_models(
    provider: str, base_url: str | None = None
) -> List[Tuple[str, str]]:
    """Fetch models from a local OpenAI-compatible provider's /models endpoint."""
    import requests

    provider_lower = provider.lower()
    if provider_lower not in _OPENAI_COMPATIBLE_MODEL_DISCOVERY:
        return []

    display_name, default_base_url, api_key_env = _OPENAI_COMPATIBLE_MODEL_DISCOVERY[
        provider_lower
    ]
    discovery_base_url = _normalize_base_url(base_url or default_base_url)
    url = f"{discovery_base_url}/models"
    api_key = os.environ.get(api_key_env, "")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None

    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        return _parse_openai_compatible_models(resp.json())
    except Exception as e:
        console.print(f"\n[yellow]Could not fetch {display_name} models: {e}[/yellow]")
        return []


def select_openai_compatible_model(
    provider: str, mode: str, base_url: str | None = None
) -> str:
    """Select a model discovered from vLLM/LiteLLM, with catalog fallback."""
    provider_lower = provider.lower()
    display_name, default_base_url, _ = _OPENAI_COMPATIBLE_MODEL_DISCOVERY[
        provider_lower
    ]
    discovery_base_url = _normalize_base_url(base_url or default_base_url)
    models = _fetch_openai_compatible_models(provider_lower, discovery_base_url)

    if models:
        prompt = f"Select {display_name} Model (available from {discovery_base_url}):"
        choices = [
            questionary.Choice(name, value=model_id) for name, model_id in models
        ]
        choices.append(questionary.Choice("Custom model ID", value="custom"))
    else:
        prompt = f"Select Your [{mode.title()}-Thinking LLM Engine]:"
        choices = [
            questionary.Choice(display, value=value)
            for display, value in get_model_options(provider_lower, mode)
        ]

    choice = questionary.select(
        prompt,
        choices=choices,
        instruction="\n- Use arrow keys to navigate\n- Press Enter to select",
        style=questionary.Style([
            ("selected", "fg:magenta noinherit"),
            ("highlighted", "fg:magenta noinherit"),
            ("pointer", "fg:magenta noinherit"),
        ]),
    ).ask()

    if choice is None:
        console.print(
            f"\n[red]No {mode} thinking llm engine selected. Exiting...[/red]"
        )
        exit(1)

    if choice == "custom":
        return _prompt_custom_model_id()

    return choice


def _prompt_custom_model_id() -> str:
    """Prompt user to type a custom model ID."""
    return questionary.text(
        "Enter model ID:",
        validate=lambda x: len(x.strip()) > 0 or "Please enter a model ID.",
    ).ask().strip()


def _select_model(provider: str, mode: str, base_url: str | None = None) -> str:
    """Select a model for the given provider and mode (quick/deep)."""
    provider_lower = provider.lower()

    if provider_lower == "openrouter":
        return select_openrouter_model()

    if provider_lower in _OPENAI_COMPATIBLE_MODEL_DISCOVERY:
        return select_openai_compatible_model(provider_lower, mode, base_url)

    if provider_lower == "azure":
        return questionary.text(
            f"Enter Azure deployment name ({mode}-thinking):",
            validate=lambda x: len(x.strip()) > 0 or "Please enter a deployment name.",
        ).ask().strip()

    choice = questionary.select(
        f"Select Your [{mode.title()}-Thinking LLM Engine]:",
        choices=[
            questionary.Choice(display, value=value)
            for display, value in get_model_options(provider, mode)
        ],
        instruction="\n- Use arrow keys to navigate\n- Press Enter to select",
        style=questionary.Style(
            [
                ("selected", "fg:magenta noinherit"),
                ("highlighted", "fg:magenta noinherit"),
                ("pointer", "fg:magenta noinherit"),
            ]
        ),
    ).ask()

    if choice is None:
        console.print(f"\n[red]No {mode} thinking llm engine selected. Exiting...[/red]")
        exit(1)

    if choice == "custom":
        return _prompt_custom_model_id()

    return choice


def select_shallow_thinking_agent(provider, base_url: str | None = None) -> str:
    """Select shallow thinking llm engine using an interactive selection."""
    return _select_model(provider, "quick", base_url)


def select_deep_thinking_agent(provider, base_url: str | None = None) -> str:
    """Select deep thinking llm engine using an interactive selection."""
    return _select_model(provider, "deep", base_url)

def select_llm_provider() -> tuple[str, str | None]:
    """Select the LLM provider and its API endpoint."""
    choice = questionary.select(
        "Select your LLM Provider:",
        choices=[
            questionary.Choice(display, value=(provider_key, url))
            for display, provider_key, url in PROVIDERS
        ],
        instruction="\n- Use arrow keys to navigate\n- Press Enter to select",
        style=questionary.Style(
            [
                ("selected", "fg:magenta noinherit"),
                ("highlighted", "fg:magenta noinherit"),
                ("pointer", "fg:magenta noinherit"),
            ]
        ),
    ).ask()
    
    if choice is None:
        console.print("\n[red]No LLM provider selected. Exiting...[/red]")
        exit(1)

    provider, url = choice
    if provider in _OPENAI_COMPATIBLE_MODEL_DISCOVERY and url:
        url = _select_openai_compatible_base_url(provider, url)

    return provider, url


def ask_openai_reasoning_effort() -> str:
    """Ask for OpenAI reasoning effort level."""
    choices = [
        questionary.Choice("Medium (Default)", "medium"),
        questionary.Choice("High (More thorough)", "high"),
        questionary.Choice("Low (Faster)", "low"),
    ]
    return questionary.select(
        "Select Reasoning Effort:",
        choices=choices,
        style=questionary.Style([
            ("selected", "fg:cyan noinherit"),
            ("highlighted", "fg:cyan noinherit"),
            ("pointer", "fg:cyan noinherit"),
        ]),
    ).ask()


def ask_anthropic_effort() -> str | None:
    """Ask for Anthropic effort level.

    Controls token usage and response thoroughness on Claude 4.5+ and 4.6 models.
    """
    return questionary.select(
        "Select Effort Level:",
        choices=[
            questionary.Choice("High (recommended)", "high"),
            questionary.Choice("Medium (balanced)", "medium"),
            questionary.Choice("Low (faster, cheaper)", "low"),
        ],
        style=questionary.Style([
            ("selected", "fg:cyan noinherit"),
            ("highlighted", "fg:cyan noinherit"),
            ("pointer", "fg:cyan noinherit"),
        ]),
    ).ask()


def ask_gemini_thinking_config() -> str | None:
    """Ask for Gemini thinking configuration.

    Returns thinking_level: "high" or "minimal".
    Client maps to appropriate API param based on model series.
    """
    return questionary.select(
        "Select Thinking Mode:",
        choices=[
            questionary.Choice("Enable Thinking (recommended)", "high"),
            questionary.Choice("Minimal/Disable Thinking", "minimal"),
        ],
        style=questionary.Style([
            ("selected", "fg:green noinherit"),
            ("highlighted", "fg:green noinherit"),
            ("pointer", "fg:green noinherit"),
        ]),
    ).ask()


def ask_output_language() -> str:
    """Ask for report output language."""
    choice = questionary.select(
        "Select Output Language:",
        choices=[
            questionary.Choice("English (default)", "English"),
            questionary.Choice("Chinese (中文)", "Chinese"),
            questionary.Choice("Japanese (日本語)", "Japanese"),
            questionary.Choice("Korean (한국어)", "Korean"),
            questionary.Choice("Hindi (हिन्दी)", "Hindi"),
            questionary.Choice("Spanish (Español)", "Spanish"),
            questionary.Choice("Portuguese (Português)", "Portuguese"),
            questionary.Choice("French (Français)", "French"),
            questionary.Choice("German (Deutsch)", "German"),
            questionary.Choice("Arabic (العربية)", "Arabic"),
            questionary.Choice("Russian (Русский)", "Russian"),
            questionary.Choice("Custom language", "custom"),
        ],
        style=questionary.Style([
            ("selected", "fg:yellow noinherit"),
            ("highlighted", "fg:yellow noinherit"),
            ("pointer", "fg:yellow noinherit"),
        ]),
    ).ask()

    if choice == "custom":
        return questionary.text(
            "Enter language name (e.g. Turkish, Vietnamese, Thai, Indonesian):",
            validate=lambda x: len(x.strip()) > 0 or "Please enter a language name.",
        ).ask().strip()

    return choice
