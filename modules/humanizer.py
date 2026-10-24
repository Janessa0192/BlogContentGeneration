import re
import time

import config
from models.schemas import Outline, UserInput, Draft, HumanizeResult, StyleMetrics
from modules.writer import find_placeholders, strip_fences, make_batches
from utils.clean import count_words
from utils.errors import RateLimitError, QuotaExhaustedError
from utils.llm import ask
from utils.textstats import style_metrics, norm_text

SYSTEM_PROMPT = """You are a careful human editor improving a draft blog article. You rewrite for natural flow and readability while keeping every fact exactly as it is.

Rules:
1. Keep EVERY fact, number, name and claim exactly as given. Never add facts, statistics, quotes, examples with numbers, or claims.
2. Keep all headings exactly: same text, same # level, same order. Keep [[...]] placeholders and markdown links exactly unchanged.
3. Keep the primary keyword and the secondary keyword usage as is. Do not add more.
4. Improve flow: natural transitions, remove repeated wording and repeated sentence openers, mix short and long sentences, prefer concrete wording. Contractions are fine if the tone suits.
5. Remove stock phrases and filler. Do not write "In conclusion" or "Let's dive in". Use em dashes sparingly.
6. Keep each chunk within about 10% of its original length. Do not summarize or cut content.
7. Output format: for every chunk, a line "===CHUNK n===" (same n as the input) followed by the full rewritten chunk in markdown. Nothing else before, between or after.
"""

_LIST_MARK = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+", re.M)
_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
_CHUNK_RE = re.compile(r"^={3}CHUNK\s+(\d+)={3}\s*$", re.M)


# ---------- chunking & guards (pure functions) ----------

def split_chunks(markdown: str) -> list[str]:
    chunks, cur, in_code = [], [], False
    for ln in markdown.strip().splitlines():
        if ln.strip().startswith("```"):
            in_code = not in_code
        if not in_code and ln.startswith("## ") and cur:
            chunks.append("\n".join(cur).strip())
            cur = []
        cur.append(ln)
    if cur:
        chunks.append("\n".join(cur).strip())
    return [c for c in chunks if c]


def heading_signature(text: str) -> list[tuple[int, str]]:
    sig, in_code = [], False
    for ln in text.splitlines():
        if ln.strip().startswith("```"):
            in_code = not in_code
            continue
        m = None if in_code else re.match(r"^(#{1,6})[ \t]+(.+?)[ \t]*$", ln)
        if m:
            sig.append((len(m.group(1)), norm_text(m.group(2))))
    return sig


def numbers_in(text: str) -> set[str]:
    t = _LIST_MARK.sub("", text)
    t = re.sub(r"\[\[[^\]]*\]\]", "", t)
    t = re.sub(r"\]\([^)]*\)", "]", t)
    return {n.replace(",", "") for n in _NUM.findall(t)}


def links_in(text: str) -> list[str]:
    return sorted(re.findall(r"\]\(([^)\s]+)\)", text))


def validate_chunk(orig: str, new: str | None, keyword: str) -> list[str]:
    if not new or not new.strip():
        return ["empty reply"]
    problems = []
    if heading_signature(orig) != heading_signature(new):
        problems.append("headings changed (text, level or order)")
    if sorted(find_placeholders(orig)) != sorted(find_placeholders(new)):
        problems.append("placeholders changed")
    if numbers_in(orig) != numbers_in(new):
        problems.append("numbers changed (added or removed)")
    if links_in(orig) != links_in(new):
        problems.append("links changed")
    ow, nw = count_words(orig), count_words(new)
    lo, hi = (0.8, 1.2) if ow >= 80 else (0.5, 1.6)
    if ow and not (lo <= nw / ow <= hi):
        problems.append(f"length changed too much ({ow} -> {nw} words)")
    kw = keyword.lower().strip()
    if kw and kw in orig.lower() and kw not in new.lower():
        problems.append("primary keyword dropped")
    return problems


def parse_chunks(raw: str, expected_ids: set[int]) -> dict[int, str]:
    raw = strip_fences(raw)
    marks = list(_CHUNK_RE.finditer(raw))
    if not marks:
        return {next(iter(expected_ids)): raw.strip()} if len(expected_ids) == 1 and raw.strip() else {}
    out = {}
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(raw)
        out[int(m.group(1))] = raw[m.end():end].strip()
    return {k: v for k, v in out.items() if k in expected_ids}


# ---------- prompting ----------

def _issue_hint(m: StyleMetrics) -> str:
    bits = []
    if m.cliches_found:
        bits.append("- Stock phrases to remove: " + ", ".join(m.cliches_found))
    if m.repeated_openers:
        bits.append("- Repetitive sentence openers: " + "; ".join(m.repeated_openers))
    if m.repeated_phrases:
        bits.append("- Phrases repeated too often: " + "; ".join(m.repeated_phrases))
    if m.sentence_len_stdev and m.sentence_len_stdev < 6:
        bits.append("- Sentence lengths are too uniform; mix short and long sentences.")
    if m.em_dash_count > 3:
        bits.append(f"- {m.em_dash_count} em dashes; use far fewer.")
    return "\n".join(bits)


def _prompt(user: UserInput, outline: Outline, hint: str, pending: list[dict],
            notes: dict[int, list[str]]) -> str:
    parts = [f"Audience: {user.audience}\nTone: {user.tone}\n"
             f"Primary keyword (keep as is): {outline.primary_keyword}\n"]
    if hint:
        parts.append("Problems detected in the draft that you should fix:\n" + hint + "\n")
    if notes:
        parts.append("Your previous attempt was rejected for these reasons; fix them:\n"
                     + "\n".join(f"- chunk {cid}: {'; '.join(p)}" for cid, p in notes.items()) + "\n")
    parts.append("Rewrite each chunk below.\n")
    for it in pending:
        parts.append(f"===CHUNK {it['id']}===\n{it['text']}\n")
    return "\n".join(parts)


# ---------- main ----------

def run_humanizer(user: UserInput, outline: Outline, draft: Draft) -> HumanizeResult:
    kw = outline.primary_keyword
    before = style_metrics(draft.markdown, kw)
    chunks = split_chunks(draft.markdown)
    items = [{"id": i + 1, "text": c, "words": count_words(c)} for i, c in enumerate(chunks)]
    hint = _issue_hint(before)
    results: dict[int, str] = {}
    warnings: list[str] = []
    aborted = False

    batches = make_batches(items, config.HUMANIZE_BATCH_WORDS)
    for bi, batch in enumerate(batches):
        if aborted:
            break
        print(f"Humanizing part {bi + 1}/{len(batches)}...")
        pending, notes = batch, {}
        for attempt in range(2):
            try:
                raw = ask(_prompt(user, outline, hint, pending, notes), SYSTEM_PROMPT,
                          max_tokens=config.WRITER_MAX_TOKENS)
            except (RateLimitError, QuotaExhaustedError) as e:
                warnings.append(f"Humanizing stopped early ({type(e).__name__}); remaining "
                                f"sections keep the original text. {e}")
                aborted = True
                break
            parsed = parse_chunks(raw, {it["id"] for it in pending})
            failed, notes = [], {}
            for it in pending:
                new = parsed.get(it["id"])
                probs = validate_chunk(it["text"], new, kw) if new else ["missing from the reply"]
                if probs:
                    failed.append(it)
                    notes[it["id"]] = probs
                else:
                    results[it["id"]] = new
            pending = failed
            if not pending:
                break
            if attempt == 0:
                print(f"  {len(pending)} section(s) failed safety checks; retrying once.")
                time.sleep(config.LLM_CALL_DELAY_SECONDS)
        if not aborted:
            for it in pending:
                warnings.append(f'Kept original for section "{it["text"].splitlines()[0][:50]}": '
                                + "; ".join(notes.get(it["id"], ["failed checks"])))
        if bi < len(batches) - 1 and not aborted:
            time.sleep(config.LLM_CALL_DELAY_SECONDS)

    kept = [it["id"] for it in items if it["id"] not in results]
    for it in items:
        results.setdefault(it["id"], it["text"])

    markdown = "\n\n".join(results[it["id"]] for it in items).strip() + "\n"
    markdown = re.sub(r"\n{3,}", "\n\n", markdown)
    after = style_metrics(markdown, kw)

    if items and len(kept) == len(items):
        warnings.append("No section could be rewritten safely; the text is unchanged.")
    if after.cliches_found:
        warnings.append("Stock phrases still present: " + ", ".join(after.cliches_found))

    return HumanizeResult(
        markdown=markdown, word_count=count_words(markdown), before=before, after=after,
        chunks_total=len(items), chunks_rewritten=len(items) - len(kept),
        chunks_kept_original=len(kept), claims_to_verify=draft.claims_to_verify, warnings=warnings,
    )


def print_humanize_summary(r: HumanizeResult) -> None:
    b, a = r.before, r.after
    print("\n=== Humanized draft ===")
    print(f"Words: {r.word_count} | sections rewritten: {r.chunks_rewritten}/{r.chunks_total}, "
          f"kept original: {r.chunks_kept_original}")
    print(f"Avg sentence length: {b.avg_sentence_len} -> {a.avg_sentence_len} | "
          f"variety (stdev): {b.sentence_len_stdev} -> {a.sentence_len_stdev}")
    print(f"Stock phrases: {len(b.cliches_found)} -> {len(a.cliches_found)} | repeated phrases: "
          f"{len(b.repeated_phrases)} -> {len(a.repeated_phrases)} | repeated openers: "
          f"{len(b.repeated_openers)} -> {len(a.repeated_openers)}")
    for w in r.warnings:
        print(f"WARNING: {w}")