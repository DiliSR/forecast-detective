"""The only module that talks to an LLM provider (currently OpenAI).

Why isolate it? The explanation writer and, later, the judge both call `complete(...)`. Changing provider or
model then touches this file only, and tests can pass a fake client instead of calling the network.

Configuration (in the project's .env file, which is git-ignored - see .env.example):
    OPENAI_API_KEY   your key (never put it in code or notebooks)
    LLM_MODEL        default model name; any call can override it with `model=`
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

from forecast_explainer import PROJECT_ROOT

load_dotenv(PROJECT_ROOT / ".env")

FALLBACK_MODEL = "gpt-4o-mini"  # used only if LLM_MODEL is not set; set LLM_MODEL to a model your account can use


@dataclass
class LlmResult:
    """What one LLM call returned.

    text:              the model's answer.
    model:             the model name the API reports having used (can be more specific than the one requested).
    prompt_tokens / completion_tokens: for cost tracking (None when the client does not report them).
    """
    text: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


def default_model() -> str:
    """The model to use when the caller does not choose: LLM_MODEL from .env, else FALLBACK_MODEL."""
    return os.getenv("LLM_MODEL") or FALLBACK_MODEL


def get_client():
    """An OpenAI client built from OPENAI_API_KEY, or a clear error telling you what to do.

    Imported inside the function so that importing this module (and everything that depends on it) works
    in environments without the openai package or a key - e.g. when only reading cached narratives.
    """
    from openai import OpenAI

    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set. Copy .env.example to .env and put your key in it.")
    return OpenAI(api_key=key)


def available_models(client=None) -> list[str]:
    """Names of the models your key can use, sorted. Handy to choose LLM_MODEL (one cheap network call)."""
    client = client or get_client()
    return sorted(m.id for m in client.models.list())


def _create(client, temperature: float, **kwargs):
    """client.chat.completions.create with one safety net shared by every call in this module.

    Some models (reasoning models) reject the `temperature` parameter. We detect that from the error message
    (so tests need not import openai) and repeat the call without it.
    """
    try:
        return client.chat.completions.create(temperature=temperature, **kwargs)
    except Exception as exc:  # openai.BadRequestError
        if "temperature" not in str(exc).lower():
            raise
        return client.chat.completions.create(**kwargs)


def complete(system: str, user: str, model: str | None = None, temperature: float = 0.0,
             max_tokens: int = 1200, client=None) -> LlmResult:
    """One chat completion: a system message (rules) and a user message (the task and evidence).

    system:      standing instructions - who the model is and what it must / must not do.
    user:        this request: the evidence, the sources, the question.
    model:       override the default model.
    temperature: 0 = as deterministic as the API allows. We want the same evidence to give (nearly) the same
                 explanation, so runs can be compared.
    max_tokens:  hard cap on the answer length (output tokens), a safety net against runaway answers/costs.
    client:      a ready client; pass a fake one in tests.
    """
    client = client or get_client()
    model = model or default_model()
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    response = _create(client, temperature, model=model, messages=messages, max_completion_tokens=max_tokens)
    usage = getattr(response, "usage", None)
    return LlmResult(
        text=response.choices[0].message.content or "",
        model=response.model,
        prompt_tokens=getattr(usage, "prompt_tokens", None),
        completion_tokens=getattr(usage, "completion_tokens", None),
    )


@dataclass
class ToolCall:
    """One tool the model asked to run: its id (to pair the reply), the tool name, and the JSON argument string."""
    id: str
    name: str
    arguments: str


@dataclass
class Turn:
    """One assistant turn in a tool-using conversation.

    content:    text the model said (can be empty when it only calls tools).
    tool_calls: tools it wants run; empty means the model is done and `content` is its final answer.
    message:    the turn in the dict form the API expects back in `messages` (append it before the tool results).
    """
    content: str
    tool_calls: list[ToolCall]
    message: dict
    prompt_tokens: int = 0
    completion_tokens: int = 0


def chat(messages: list[dict], tools: list[dict] | None = None, model: str | None = None,
         temperature: float = 0.0, max_tokens: int = 1200, client=None) -> Turn:
    """One assistant turn of a conversation that may use tools (function calling).

    messages: the whole conversation so far (system, user, assistant, tool messages).
    tools:    tool definitions in the OpenAI function format; the model may answer with tool_calls instead of text.
    The caller runs the tools, appends `turn.message` and one {"role": "tool", ...} message per call, and calls again.
    """
    client = client or get_client()
    kwargs = dict(model=model or default_model(), messages=messages, max_completion_tokens=max_tokens)
    if tools:
        kwargs.update(tools=tools, tool_choice="auto")
    response = _create(client, temperature, **kwargs)
    msg = response.choices[0].message
    calls = [ToolCall(c.id, c.function.name, c.function.arguments) for c in (getattr(msg, "tool_calls", None) or [])]
    message = {"role": "assistant", "content": msg.content or ""}
    if calls:
        message["tool_calls"] = [{"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}} for c in calls]
    usage = getattr(response, "usage", None)
    return Turn(msg.content or "", calls, message, getattr(usage, "prompt_tokens", 0) or 0, getattr(usage, "completion_tokens", 0) or 0)
