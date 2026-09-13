"""LLM clients. W&B Inference speaks the OpenAI chat-completions API."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Protocol

from looppp.config import ModelSettings

WANDB_INFERENCE_BASE_URL = "https://api.inference.wandb.ai/v1"


@dataclass
class LLMReply:
    text: str
    finish_reason: str | None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    attempts: int = 1
    max_tokens_used: int = 0


class EmptyCompletionError(RuntimeError):
    """The model produced no usable text even after raising max_tokens."""


class LLM(Protocol):
    model: str

    def complete(self, messages: list[dict]) -> LLMReply: ...


def _usable(text: str, finish_reason: str | None) -> bool:
    """Reasoning models can spend the whole budget thinking and return nothing, or get cut
    off mid code block. Both need a retry with more tokens, not a submission."""
    if not text.strip():
        return False
    if finish_reason == "length" and text.count("```") % 2 == 1:
        return False
    return True


class WandbInferenceLLM:
    def __init__(self, model: str, settings: ModelSettings, api_key: str, project: str | None = None,
                 base_url: str = WANDB_INFERENCE_BASE_URL, sleep: Callable[[float], None] = time.sleep):
        from openai import OpenAI

        self.model, self.settings, self.sleep = model, settings, sleep
        # `project` ("team/project") attributes usage to a W&B project.
        self.client = OpenAI(base_url=base_url, api_key=api_key, project=project,
                             timeout=settings.timeout_seconds, max_retries=0)

    def complete(self, messages: list[dict]) -> LLMReply:
        import openai

        max_tokens = self.settings.max_tokens
        transient = (openai.APIConnectionError, openai.APITimeoutError, openai.RateLimitError,
                     openai.InternalServerError)
        last_reason = None
        for attempt in range(1, 7):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model, messages=messages, max_tokens=max_tokens,
                    temperature=self.settings.temperature, extra_body=self.settings.extra_body or None,
                )
            except transient as e:
                wait = min(300.0, 10.0 * 2 ** (attempt - 1))
                last_reason = f"{type(e).__name__}: {e}"
                self.sleep(wait)
                continue
            choice = resp.choices[0]
            text = choice.message.content or ""
            usage = resp.usage
            reply = LLMReply(text, choice.finish_reason,
                             getattr(usage, "prompt_tokens", 0) or 0, getattr(usage, "completion_tokens", 0) or 0,
                             attempt, max_tokens)
            if _usable(text, choice.finish_reason):
                return reply
            last_reason = f"unusable reply (finish_reason={choice.finish_reason}, {len(text)} chars)"
            if max_tokens >= self.settings.max_tokens_cap:
                break
            max_tokens = min(self.settings.max_tokens_cap, max_tokens * 2)
        raise EmptyCompletionError(f"{self.model}: {last_reason}")


class StubLLM:
    """Offline stand-in. Each call returns the next scripted reply (cycling), or a reply
    built by ``factory(messages, call_index)``."""

    def __init__(self, replies: list[str] | None = None,
                 factory: Callable[[list[dict], int], str] | None = None, model: str = "stub"):
        if not replies and not factory:
            raise ValueError("StubLLM needs replies or a factory")
        self.replies, self.factory, self.model = replies or [], factory, model
        self.calls: list[list[dict]] = []

    def complete(self, messages: list[dict]) -> LLMReply:
        i = len(self.calls)
        self.calls.append(messages)
        text = self.factory(messages, i) if self.factory else self.replies[i % len(self.replies)]
        return LLMReply(text, "stop", 0, len(text) // 4)
