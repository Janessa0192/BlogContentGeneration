import time
import requests
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser
from bs4 import BeautifulSoup

from config import USER_AGENT, REQUEST_TIMEOUT, SCRAPE_DELAY_SECONDS, MAX_TEXT_CHARS
from models.schemas import ScrapedPage, SearchResult

_robots_cache: dict[str, RobotFileParser | None] = {}


def _get_robots(base: str) -> RobotFileParser | None:
    """Returns a parser, or None if robots.txt could not be read safely."""
    if base in _robots_cache:
        return _robots_cache[base]

    rp = RobotFileParser()
    try:
        resp = requests.get(f"{base}/robots.txt", headers={"User-Agent": USER_AGENT},
                            timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200:
            rp.parse(resp.text.splitlines())
        elif resp.status_code in (401, 403):
            rp.disallow_all = True           # access denied -> treat as disallowed
        elif resp.status_code >= 500:
            rp = None                        # server error -> be conservative
        else:
            rp.parse([])                     # 404 etc. -> no rules, everything allowed
    except requests.RequestException:
        rp = None

    _robots_cache[base] = rp
    return rp


def is_allowed(url: str) -> bool:
    parsed = urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    rp = _get_robots(base)
    if rp is None:
        return False
    return rp.can_fetch(USER_AGENT, url)


def _clean_texts(tags) -> list[str]:
    return [t.get_text(" ", strip=True) for t in tags if t.get_text(strip=True)]


def parse_html(url: str, html: str) -> ScrapedPage:
    soup = BeautifulSoup(html, "html.parser")

    title = soup.title.get_text(strip=True) if soup.title else None
    desc_tag = soup.find("meta", attrs={"name": "description"})
    canon_tag = soup.find("link", rel="canonical")

    for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()

    main = soup.find("article") or soup.find("main") or soup.body or soup
    paragraphs = _clean_texts(main.find_all(["p", "li"]))
    text = "\n".join(p for p in paragraphs if len(p.split()) > 3)

    return ScrapedPage(
        url=url,
        status="ok",
        title=title,
        meta_description=desc_tag.get("content") if desc_tag else None,
        canonical=canon_tag.get("href") if canon_tag else None,
        h1=_clean_texts(main.find_all("h1")) or _clean_texts(soup.find_all("h1")),
        h2=_clean_texts(main.find_all("h2")),
        h3=_clean_texts(main.find_all("h3")),
        text=text[:MAX_TEXT_CHARS],
        word_count=len(text.split()),
    )


def scrape_page(url: str) -> ScrapedPage:
    if not is_allowed(url):
        return ScrapedPage(url=url, status="blocked_by_robots",
                           error="Disallowed or unreadable robots.txt")
    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        if "text/html" not in resp.headers.get("Content-Type", ""):
            return ScrapedPage(url=url, status="failed", error="Not an HTML page")
        return parse_html(url, resp.text)
    except requests.RequestException as e:
        return ScrapedPage(url=url, status="failed", error=str(e))


def scrape_competitors(competitors: list[SearchResult]) -> list[ScrapedPage]:
    pages = []
    for i, c in enumerate(competitors):
        print(f"Scraping {c.url}")
        page = scrape_page(c.url)
        print(f"  -> {page.status}" + (f" ({page.word_count} words)" if page.status == "ok" else ""))
        pages.append(page)
        if i < len(competitors) - 1:
            time.sleep(SCRAPE_DELAY_SECONDS)
    return pages