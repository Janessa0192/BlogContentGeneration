import time
from urllib.parse import urlparse

from tavily import TavilyClient

import config
from config import MAX_COMPETITORS
from models.schemas import UserInput, SearchResult, SEOBrief
from utils.errors import (ProviderError, InvalidKeyError, RateLimitError,
                          QuotaExhaustedError, redact)
from utils.llm import ask_json

BLOCKED_DOMAINS = {
    "wikipedia.org", "reddit.com", "quora.com", "facebook.com", "instagram.com",
    "twitter.com", "x.com", "linkedin.com", "youtube.com", "pinterest.com",
    "amazon.com", "amazon.in", "flipkart.com", "ebay.com", "medium.com",
    "glassdoor.com", "indeed.com", "naukri.com", "tripadvisor.com",
}
BLOG_HINTS = ("/blog", "/resources", "/articles", "/article", "/guide", "/learn", "/insights")


def get_domain(url: str) -> str:
    host = urlparse(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def _is_blocked(domain: str) -> bool:
    return any(domain == b or domain.endswith("." + b) for b in BLOCKED_DOMAINS)


_tavily = None


def _tavily_client() -> TavilyClient:
    global _tavily
    if _tavily is None:
        _tavily = TavilyClient(api_key=config.TAVILY_API_KEY)
    return _tavily


def _classify_tavily(e: Exception) -> tuple[ProviderError, bool]:
    """Returns (error_to_raise, retryable). Matches on class name so SDK versions don't matter."""
    name = type(e).__name__
    low = redact(e).lower()

    if name in ("InvalidAPIKeyError", "MissingAPIKeyError", "ForbiddenError") \
            or "unauthorized" in low or "invalid api key" in low:
        return InvalidKeyError(
            "Tavily rejected TAVILY_API_KEY (invalid or missing). "
            "Check it at https://app.tavily.com and update .env."
        ), False

    if name == "UsageLimitExceededError" or "usage limit" in low or "rate limit" in low \
            or "too many requests" in low:
        if "rate limit" in low or "too many" in low:
            return RateLimitError(
                "Tavily rate limit hit (too many requests too quickly). "
                "Wait a moment and try again."
            ), True
        return QuotaExhaustedError(
            "Tavily free-tier credits are used up (or plan limit reached). "
            "Check usage at https://app.tavily.com; credits reset on your plan's cycle."
        ), False

    if name == "TimeoutError" or "timed out" in low:
        return ProviderError("Tavily request timed out. Try again."), True

    return ProviderError(f"Tavily error ({name}): {redact(e)[:300]}"), False


def tavily_search(query: str, num: int = 10) -> list[SearchResult]:
    for attempt in range(3):
        try:
            resp = _tavily_client().search(
                query=query, max_results=num, search_depth="basic"
            )
            break
        except Exception as e:
            err, retryable = _classify_tavily(e)
            if retryable and attempt < 2:
                wait = 5 * (attempt + 1)
                print(f"  [Tavily] {type(err).__name__}; retrying in {wait}s")
                time.sleep(wait)
                continue
            raise err from None

    return [
        SearchResult(
            title=item.get("title", "") or "",
            url=item.get("url", "") or "",
            snippet=item.get("content", "") or "",
            position=i,            # Tavily gives relevance order, so rank = index
        )
        for i, item in enumerate(resp.get("results", []), start=1)
    ]

def suggest_keyword(user: UserInput) -> str:
    data = ask_json(
        f"Company: {user.company_name}\nTopic: {user.topic}\nAudience: {user.audience}\n\n"
        'Suggest the single best primary SEO keyword (2-5 words) for a blog post. '
        'Return JSON: {"keyword": "..."}'
    )
    return data["keyword"]


def find_competitors(user: UserInput, keyword: str) -> list[SearchResult]:
    own_domain = get_domain(user.website) if user.website else ""
    queries = [
        keyword,
        f"{keyword} blog",
        f"{user.company_name} competitors {user.topic}",
    ]

    candidates: list[SearchResult] = []
    for q in queries:
        try:
            candidates.extend(tavily_search(q))
        except Exception as e:
            print(f"  Search failed for '{q}': {e}")

    seen_domains = set()
    own_name = user.company_name.lower().replace(" ", "")
    scored = []
    for r in candidates:
        domain = get_domain(r.url)
        if (not domain or _is_blocked(domain) or domain == own_domain
                or own_name in domain.replace("-", "") or r.url.lower().endswith(".pdf")):
            continue
        if domain in seen_domains:
            continue
        seen_domains.add(domain)
        blog_like = any(h in r.url.lower() for h in BLOG_HINTS)
        scored.append((0 if blog_like else 1, r.position, r))

    scored.sort(key=lambda x: (x[0], x[1]))
    return [r for _, _, r in scored[:MAX_COMPETITORS]]


def generate_brief(user: UserInput, keyword: str, competitors: list[SearchResult]) -> SEOBrief:
    comp_text = "\n".join(f"- {c.title} ({c.url}): {c.snippet}" for c in competitors)
    data = ask_json(
        f"Create an SEO brief for a blog post.\n\n"
        f"Company: {user.company_name}\nTopic: {user.topic}\nAudience: {user.audience}\n"
        f"Tone: {user.tone}\nWord count: {user.word_count}\nPrimary keyword: {keyword}\n\n"
        f"Top ranking pages:\n{comp_text}\n\n"
        "Return JSON with keys: primary_keyword (string), secondary_keywords (5-8 strings), "
        "search_intent (informational|commercial|transactional|navigational), "
        "content_angle (1-2 sentences on how to stand out), suggested_title (string), "
        "questions_to_answer (5-8 strings readers would ask)."
    )
    return SEOBrief(**data)


def run_seo_research(user: UserInput):
    keyword = user.target_keyword
    if not keyword:
        print("No keyword given, asking the LLM to suggest one...")
        keyword = suggest_keyword(user)
        print(f"  Suggested keyword: {keyword}")

    print("Searching for competitor pages...")
    competitors = find_competitors(user, keyword)
    for c in competitors:
        print(f"  - {c.title} | {c.url}")

    print("Generating SEO brief...")
    brief = generate_brief(user, keyword, competitors)
    return brief, competitors