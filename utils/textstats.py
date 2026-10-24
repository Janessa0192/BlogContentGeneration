import re
import statistics
from collections import Counter

from models.schemas import StyleMetrics
from utils.clean import count_words

CLICHES = [
    "in today's fast-paced", "in today's digital", "ever-evolving", "it's important to note",
    "it is important to note", "it's worth noting", "it is worth noting", "game-changer",
    "game changer", "unlock the power", "unlock the full potential", "delve", "tapestry",
    "navigate the complexities", "navigating the complexities", "in conclusion",
    "let's dive in", "dive into", "look no further", "at the end of the day",
    "plays a crucial role", "plays a vital role", "a testament to", "to the next level",
    "in the realm of", "embark on", "harness the power", "treasure trove", "whether you're a",
]

_STOP = {"the", "a", "an", "of", "to", "and", "in", "on", "for", "is", "are", "that", "this",
         "with", "it", "you", "your", "can", "be", "as", "at", "by", "or"}

_ABBR = re.compile(r"\b(e\.g|i\.e|vs|approx|Mr|Mrs|Dr|Inc|Ltd)\.", re.I)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[^\w\s]?[A-Z0-9])")


def norm_text(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()


def prose_text(md: str) -> str:
    """Body paragraphs only: no headings, lists, quotes, tables, code, placeholders or link URLs."""
    lines, in_code = [], False
    for ln in md.splitlines():
        if ln.strip().startswith("```"):
            in_code = not in_code
            continue
        st = ln.strip()
        if in_code or not st or st.startswith(("#", ">", "|")) or re.match(r"^([-*+]|\d+[.)])\s+", st):
            lines.append("")
            continue
        lines.append(st)
    text = "\n".join(lines)
    text = re.sub(r"\[\[[^\]]*\]\]", "", text)
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)
    return re.sub(r"[*`]+", "", text)


def paragraphs(md: str) -> list[str]:
    return [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", prose_text(md)) if p.strip()]


def split_sentences(text: str) -> list[str]:
    t = _ABBR.sub(lambda m: m.group(1).replace(".", ""), text)
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return []
    return [p.strip() for p in _SENT_SPLIT.split(t) if count_words(p) >= 2]


def sentence_lengths(md: str) -> list[int]:
    return [count_words(s) for s in split_sentences(prose_text(md))]


def paragraph_sentence_counts(md: str) -> list[int]:
    return [n for n in (len(split_sentences(p)) for p in paragraphs(md)) if n > 0]


def count_syllables(word: str) -> int:
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    if len(w) <= 3:
        return 1
    w = re.sub(r"(?:[^laeiouy]es|ed|[^laeiouy]e)$", "", w)
    w = re.sub(r"^y", "", w)
    return max(1, len(re.findall(r"[aeiouy]{1,2}", w)))


def flesch_reading_ease(text: str) -> float:
    sents = split_sentences(text)
    words = re.findall(r"[A-Za-z][A-Za-z'’-]*", text)
    if not sents or not words:
        return 0.0
    syll = sum(count_syllables(w) for w in words)
    return round(206.835 - 1.015 * (len(words) / len(sents)) - 84.6 * (syll / len(words)), 1)


def repeated_openers(sents: list[str], min_count: int = 4) -> list[str]:
    firsts = [" ".join(re.findall(r"[A-Za-z']+", s)[:2]).lower() for s in sents]
    out = [f'"{k}..." x{c}' for k, c in Counter(f for f in firsts if f).most_common() if c >= min_count]
    first_words = [(f.split(" ")[0] if f else "") for f in firsts]
    run = 1
    for i in range(1, len(first_words)):
        if first_words[i] and first_words[i] == first_words[i - 1]:
            run += 1
            if run == 3:
                out.append(f'3 sentences in a row start with "{first_words[i]}"')
        else:
            run = 1
    return out


def repeated_phrases(text: str, keyword: str = "", min_count: int = 3, top: int = 5) -> list[str]:
    words = [w.lower().replace("’", "'") for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9'’-]*", text)]
    kw = keyword.lower().strip()
    grams = Counter(" ".join(words[i:i + 4]) for i in range(len(words) - 3))
    chosen: list[str] = []
    for gram, count in grams.most_common():
        if count < min_count:
            break
        parts = gram.split()
        if kw and (kw in gram or gram in kw):
            continue
        if sum(1 for w in parts if w in _STOP) >= 3 or len(set(parts)) < 3:
            continue
        if any(parts[1:] == c.split()[:-1] or c.split()[1:] == parts[:-1] for c in chosen):
            continue
        chosen.append(gram)
        if len(chosen) >= top:
            break
    return chosen


def style_metrics(markdown: str, keyword: str = "") -> StyleMetrics:
    text = prose_text(markdown)
    sents = split_sentences(text)
    lens = [count_words(s) for s in sents]
    per_para = paragraph_sentence_counts(markdown)
    low = text.lower().replace("’", "'")
    return StyleMetrics(
        words=count_words(text),
        sentences=len(sents),
        avg_sentence_len=round(statistics.mean(lens), 1) if lens else 0.0,
        sentence_len_stdev=round(statistics.pstdev(lens), 1) if len(lens) > 1 else 0.0,
        avg_paragraph_sentences=round(statistics.mean(per_para), 1) if per_para else 0.0,
        repeated_openers=repeated_openers(sents),
        repeated_phrases=repeated_phrases(text, keyword),
        cliches_found=[c for c in CLICHES if c in low],
        em_dash_count=markdown.count("\u2014"),
    )