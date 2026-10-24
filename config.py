import os
from dotenv import load_dotenv

load_dotenv()


def _clean(value: str | None) -> str:
    return (value or "").strip().strip('"').strip("'")


GEMINI_API_KEY = _clean(os.getenv("GEMINI_API_KEY"))
TAVILY_API_KEY = _clean(os.getenv("TAVILY_API_KEY"))
GEMINI_MODEL = _clean(os.getenv("GEMINI_MODEL")) or "gemini-2.5-flash"

USER_AGENT = "SEOBlogAgentBot/1.0 (educational project)"
REQUEST_TIMEOUT = 15
SCRAPE_DELAY_SECONDS = 2
MAX_COMPETITORS = 5
MAX_TEXT_CHARS = 15000
MIN_USABLE_WORDS = 150     # pages thinner than this are ignored in analysis
EXCERPT_CHARS = 1500       # per-page text excerpt sent to Gemini
MAX_H2_IN_PROMPT = 20
MAX_H3_IN_PROMPT = 12
# Part 5-6
MAX_GAPS_IN_OUTLINE = 10
BATCH_WORD_LIMIT = 900          # target words written per Gemini call
WRITER_MAX_TOKENS = 8192
LLM_CALL_DELAY_SECONDS = 4      # pause between writer calls (free-tier friendliness)

_PLACEHOLDER_PREFIXES = ("your_", "xxx", "changeme")


def secrets() -> list[str]:
    """Used by the log redactor. Never print these."""
    return [GEMINI_API_KEY, TAVILY_API_KEY]


def validate_keys(need_search: bool = True) -> None:
    from utils.errors import ConfigError  # lazy import avoids a circular import

    required = [("GEMINI_API_KEY", GEMINI_API_KEY)]
    if need_search:
        required.append(("TAVILY_API_KEY", TAVILY_API_KEY))

    bad = [name for name, val in required
           if not val or val.lower().startswith(_PLACEHOLDER_PREFIXES)]
    if bad:
        raise ConfigError(
            "Missing or placeholder value in .env for: " + ", ".join(bad) +
            ". Copy .env.example to .env and add your real keys."
        )

# Part 7-9
HUMANIZE_BATCH_WORDS = 1000
MAX_CLAIMS_TO_VERIFY = 5
MAX_LINKS_TO_CHECK = 10


def search_key_ok() -> bool:
    return bool(TAVILY_API_KEY) and not TAVILY_API_KEY.lower().startswith(_PLACEHOLDER_PREFIXES)