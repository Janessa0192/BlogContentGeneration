import re


def s(v, default: str = "") -> str:
    return v.strip() if isinstance(v, str) else default


def as_list(v) -> list[str]:
    return [x.strip() for x in v if isinstance(x, str) and x.strip()] if isinstance(v, list) else []


def dedupe(items: list[str], cap: int) -> list[str]:
    seen, out = set(), []
    for x in items:
        k = x.lower()
        if k not in seen:
            seen.add(k)
            out.append(x)
    return out[:cap]


def to_int(v, default: int) -> int:
    if isinstance(v, bool):
        return default
    if isinstance(v, (int, float)):
        return int(v)
    if isinstance(v, str):
        m = re.search(r"\d+", v)
        if m:
            return int(m.group())
    return default


def count_words(text: str) -> int:
    return len(re.findall(r"\w[\w'’-]*", text))


def slugify(text: str, max_len: int = 60) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    if len(slug) > max_len:
        cut = slug[:max_len]
        slug = cut.rsplit("-", 1)[0] if "-" in cut else cut
    return slug.strip("-")