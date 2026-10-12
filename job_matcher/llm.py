"""A tiny LLM client written by hand with httpx (no SDK, no LangChain).

Everything an LLM framework does starts with this: send a list of messages
to an HTTP endpoint, get a message back. Reading this file teaches you what
those frameworks hide.
"""
import json
import logging
import re
import time

import httpx

from .config import Settings

log = logging.getLogger(__name__)


class LLMError(Exception):
    """Raised when the LLM call fails in a way retrying won't fix."""


def chat(settings: Settings, messages: list[dict], temperature: float = 0.2) -> str:
    """Send messages to the model and return the reply text.

    Retries on rate limits (429), server errors (5xx) and timeouts,
    with exponential backoff: wait 3s, 6s, 12s...
    """
    url = f"{settings.base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.api_key}",
        "Content-Type": "application/json",
        "X-Title": "job-matcher",  # OpenRouter shows this name in your usage dashboard
    }
    body = {"model": settings.model, "messages": messages, "temperature": temperature}

    for attempt in range(4):
        try:
            resp = httpx.post(url, headers=headers, json=body, timeout=settings.request_timeout)
        except httpx.TimeoutException:
            wait = 3 * 2**attempt
            log.warning("LLM timed out (attempt %d), retrying in %ds", attempt + 1, wait)
            time.sleep(wait)
            continue

        if resp.status_code == 429 or resp.status_code >= 500:
            wait = 3 * 2**attempt
            log.warning("LLM returned %d (attempt %d), retrying in %ds",
                        resp.status_code, attempt + 1, wait)
            time.sleep(wait)
            continue

        if resp.status_code != 200:
            # 401 = bad key, 400 = bad request, 404 = model name wrong. Retrying won't help.
            raise LLMError(f"LLM request failed ({resp.status_code}): {resp.text[:400]}")

        data = resp.json()
        if "error" in data:
            raise LLMError(f"LLM returned an error: {data['error']}")
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise LLMError(f"Unexpected response shape: {str(data)[:400]}")

        usage = data.get("usage") or {}
        log.info("LLM tokens: prompt=%s completion=%s",
                 usage.get("prompt_tokens"), usage.get("completion_tokens"))
        return content or ""

    raise LLMError(
        "LLM failed after 4 attempts. Free models have per-minute and daily limits; "
        "wait a minute and try again, or pick another free model."
    )


def extract_json(text: str):
    """Pull a JSON object/array out of an LLM reply.

    Models often wrap JSON in ```json fences or add a sentence before it.
    This is the most common real-world LLM bug, so we handle it in one place.
    """
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
    if not starts:
        raise ValueError("No JSON found in model reply")
    start = min(starts)
    closer = "}" if text[start] == "{" else "]"
    end = text.rfind(closer)
    if end <= start:
        raise ValueError("JSON in model reply is incomplete")
    return json.loads(text[start:end + 1])


def chat_json(settings: Settings, system: str, user: str):
    """Ask for JSON. If the reply isn't valid JSON, show the model its mistake once."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    reply = chat(settings, messages)
    try:
        return extract_json(reply)
    except (ValueError, json.JSONDecodeError) as exc:
        log.warning("Model returned invalid JSON (%s); asking it to fix it", exc)
        messages += [
            {"role": "assistant", "content": reply},
            {"role": "user", "content": "That was not valid JSON. Reply again with ONLY the JSON, no other text."},
        ]
        return extract_json(chat(settings, messages))
