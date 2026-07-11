"""
Minimal OpenAI-compatible chat-completions client for the Use Case 2
generator (see USE_CASES_IMPLEMENTATION.md).

Deliberately standalone (not app.agent_core._call_general_llm) so the
backend is a pure env-var swap:

    OpenRouter (today):   LLM_BASE_URL=https://openrouter.ai/api/v1
                          LLM_API_KEY=<or reuse OPENROUTER_API_KEY>
                          LLM_MODEL=openai/gpt-5
    vLLM on AMD (target): LLM_BASE_URL=http://<mi300x-host>:8000/v1
                          LLM_API_KEY=anything-nonempty
                          LLM_MODEL=Qwen/Qwen2.5-Coder-32B-Instruct

Falls back to the OPENROUTER_* variables already used elsewhere in this
repo, so no .env changes are needed to run against OpenRouter.
"""

import os
import re

import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BASE_URL = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
API_KEY = os.getenv("LLM_API_KEY") or os.getenv("OPENROUTER_API_KEY", "")
MODEL = os.getenv("LLM_MODEL") or os.getenv("OPENROUTER_MODEL", "openai/gpt-5")


class LLMError(Exception):
    """Raised when the chat-completions call fails or returns no content."""


def call_llm(user_prompt: str, system_prompt: str = "",
             temperature: float = 0.2, max_tokens: int = 8000,
             timeout_s: int = 180) -> str:
    """Single-turn chat completion; returns the assistant message text."""
    if not API_KEY:
        raise LLMError("No API key: set LLM_API_KEY or OPENROUTER_API_KEY.")

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:
        resp = requests.post(
            f"{BASE_URL}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {API_KEY}",
                     "Content-Type": "application/json"},
            timeout=timeout_s,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        status = getattr(e.response, "status_code", None)
        status_text = f" (HTTP {status})" if status else ""
        raise LLMError(f"LLM request failed{status_text}: {e}") from e
    try:
        data = resp.json()
    except ValueError as e:
        raise LLMError("LLM returned a non-JSON response.") from e
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise LLMError(f"Malformed completion response: {data}") from e
    if not content:
        raise LLMError("Empty completion content.")
    return content


_FENCE_RE = re.compile(r"```[^\n]*\r?\n(.*?)```", re.DOTALL)
_NETLIST_START_RE = re.compile(
    r"^\s*(?:\*|[RVCLIDXMQEFG]\S*\s+\S+|\.(?:ac|backanno|dc|end|ends|include|lib|model|op|options|param|subckt|temp|tran))",
    re.IGNORECASE,
)


def extract_netlist(text: str) -> str:
    """Pull the netlist out of an LLM reply.

    Handles: fenced ```spice/```netlist blocks, bare fences, or raw netlist
    text with surrounding chatter. Returns text through the final `.end`
    line when one exists, so trailing commentary is dropped.
    """
    m = _FENCE_RE.search(text)
    candidate = m.group(1) if m else text

    lines = candidate.strip().splitlines()
    if not m:
        # A title comment is the most reliable netlist boundary and avoids
        # mistaking prose such as "I changed R2..." for a current-source
        # line. Fall back to element/directive recognition if a model omitted
        # a title comment entirely.
        comment_start = next(
            (i for i, line in enumerate(lines) if line.lstrip().startswith("*")),
            None,
        )
        if comment_start is not None:
            lines = lines[comment_start:]
        else:
            # For an unfenced chatty response, discard prose before the first
            # recognizable SPICE line. Fenced responses have already had
            # their wrapper removed and must be left exactly as supplied.
            for i, line in enumerate(lines):
                if _NETLIST_START_RE.match(line):
                    lines = lines[i:]
                    break
    end_idx = None
    for i, line in enumerate(lines):
        if line.strip().lower() in (".end", ".ends"):
            end_idx = i
    if end_idx is not None:
        lines = lines[: end_idx + 1]
    return "\n".join(lines).strip() + "\n"
