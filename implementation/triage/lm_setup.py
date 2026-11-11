"""LM construction and call/token/cost accounting."""

from __future__ import annotations

import datetime
import os
import threading
import time
from contextlib import contextmanager

import dspy
from dspy.utils.callback import BaseCallback


def build_lm(cfg: dict) -> dspy.LM:
    """cfg keys: model, temperature, max_tokens, cache, api_base (optional).

    `model` is any LiteLLM-style id, e.g.
        openai/gpt-4o-mini            (OPENAI_API_KEY)
        groq/llama-3.1-8b-instant     (GROQ_API_KEY, free tier)
        gemini/gemini-2.0-flash       (GEMINI_API_KEY, free tier)
        ollama_chat/llama3.1:8b       (local Ollama, api_base http://localhost:11434)
        mock                          (offline fake model for smoke tests)
    """
    if cfg["model"] == "mock":
        from .mock_lm import MockTriageLM
        return MockTriageLM()
    kwargs = dict(temperature=cfg.get("temperature", 0.0), max_tokens=cfg.get("max_tokens", 600),
                  cache=cfg.get("cache", False), num_retries=cfg.get("num_retries", 5))
    if cfg.get("reasoning_effort"):  # reasoning models (e.g. gpt-oss): low|medium|high
        kwargs["reasoning_effort"] = cfg["reasoning_effort"]
    if cfg.get("api_base"):
        kwargs["api_base"] = cfg["api_base"]
    if cfg.get("api_key_env"):
        kwargs["api_key"] = os.environ[cfg["api_key_env"]]
    return dspy.LM(cfg["model"], **kwargs)


def _tok(usage: dict, *names) -> int:
    for n in names:
        if n in usage and isinstance(usage[n], (int, float)):
            return int(usage[n])
    return 0


class _LMWatcher(BaseCallback):
    """Global callback that remembers every LM instance used (including optimiser copies
    and calls made from worker threads, which dspy.track_usage does not see)."""

    def __init__(self):
        self.instances = {}
        self.lock = threading.Lock()

    def on_lm_start(self, call_id, instance, inputs):
        with self.lock:
            self.instances[id(instance)] = instance


WATCHER = _LMWatcher()


def install_watcher():
    cbs = list(dspy.settings.get("callbacks", None) or [])
    if WATCHER not in cbs:
        dspy.configure(callbacks=cbs + [WATCHER])


@contextmanager
def track(stage: str, prices: dict | None = None, sink: dict | None = None):
    """Measure wall time, #LM calls, tokens and estimated USD cost of a block.

    Counts are read from the history of every LM instance seen during the block,
    filtered by timestamp. prices = {"input_per_1m": float, "output_per_1m": float}.
    Results are written into sink[stage].
    """
    prices = prices or {}
    install_watcher()
    start_iso = datetime.datetime.now().isoformat()
    t0 = time.time()
    yield
    calls = tin = tout = 0
    with WATCHER.lock:
        instances = list(WATCHER.instances.values())
    for lm in instances:
        for e in list(getattr(lm, "history", [])):
            if e.get("timestamp", "") < start_iso:
                continue
            calls += 1
            u = e.get("usage") or {}
            tin += _tok(u, "prompt_tokens", "input_tokens")
            tout += _tok(u, "completion_tokens", "output_tokens")
    rec = {
        "wall_time_s": round(time.time() - t0, 2),
        "lm_calls": calls,
        "input_tokens": tin,
        "output_tokens": tout,
        "est_cost_usd": round(tin / 1e6 * prices.get("input_per_1m", 0) + tout / 1e6 * prices.get("output_per_1m", 0), 4),
    }
    if sink is not None:
        sink[stage] = rec
