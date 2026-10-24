import json
import re
import time

from google import genai
from google.genai import types

import config
from utils.errors import (ProviderError, ConfigError, InvalidKeyError,
                          RateLimitError, QuotaExhaustedError, redact)

_client = None
MAX_RETRIES = 3


def _get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _client


def _classify(e: Exception):
    """Returns (error_to_raise, retryable, suggested_delay_seconds)."""
    code = getattr(e, "code", None)
    msg = redact(getattr(e, "message", None) or e)
    low = msg.lower()

    if code in (401, 403) or "api key not valid" in low or "api_key_invalid" in low \
            or (code == 400 and "api key" in low):
        return InvalidKeyError(
            "Gemini rejected GEMINI_API_KEY (invalid, revoked, or lacks permission). "
            "Create a new key at https://aistudio.google.com/apikey and update .env."
        ), False, None

    if code == 404:
        return ConfigError(
            f"Gemini model '{config.GEMINI_MODEL}' was not found or isn't available to you. "
            "Set GEMINI_MODEL in .env to a model shown in Google AI Studio."
        ), False, None

    if code == 429:
        if "perday" in low or "per day" in low or "daily" in low:
            return QuotaExhaustedError(
                "Gemini free-tier DAILY quota is used up for this model. It resets daily; "
                "wait, switch GEMINI_MODEL in .env, or enable billing."
            ), False, None
        m = re.search(r"retry in ([\d.]+)s", low) or re.search(r"retrydelay\W+(\d+)", low)
        delay = float(m.group(1)) if m else None
        return RateLimitError(
            "Gemini rate limit hit (too many requests per minute on the free tier). "
            "Wait a minute and try again."
        ), True, delay

    if code in (500, 503, 504):
        return ProviderError(
            f"Gemini is temporarily unavailable (HTTP {code}). Try again shortly."
        ), True, 10

    return ProviderError(f"Gemini API error (HTTP {code}): {msg[:300]}"), False, None


def ask(prompt: str, system: str = "", max_tokens: int = 4096,
        json_mode: bool = False) -> str:
    config_kwargs = dict(max_output_tokens=max_tokens)
    if system:
        config_kwargs["system_instruction"] = system
    if json_mode:
        config_kwargs["response_mime_type"] = "application/json"

    for attempt in range(MAX_RETRIES):
        try:
            resp = _get_client().models.generate_content(
                model=config.GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(**config_kwargs),
            )
            text = resp.text
            if not text:
                raise ProviderError(
                    "Gemini returned an empty response (possibly blocked by safety "
                    "filters or cut off). Try rephrasing or raising max_tokens."
                )
            return text
        except ProviderError:
            raise
        except Exception as e:
            err, retryable, delay = _classify(e) if hasattr(e, "code") else (
                ProviderError(f"Could not reach Gemini: {redact(e)[:300]}"), True, 5)
            if retryable and attempt < MAX_RETRIES - 1:
                wait = min(delay or 20 * (attempt + 1), 90)
                print(f"  [Gemini] {type(err).__name__}; retrying in {wait:.0f}s "
                      f"({attempt + 1}/{MAX_RETRIES - 1})")
                time.sleep(wait)
                continue
            raise err from None  # "from None" keeps SDK internals out of tracebacks


def ask_json(prompt: str, system: str = "", max_tokens: int = 4096) -> dict:
    text = ask(prompt + "\n\nRespond with valid JSON only.", system, max_tokens,
               json_mode=True)
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        raise ProviderError(
            "Gemini returned invalid JSON (often a truncated reply). "
            "Try again or raise max_tokens."
        ) from None