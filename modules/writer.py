import re
import time

import config
from models.schemas import Outline, UserInput, Draft, KeywordReport
from utils.clean import count_words
from utils.errors import ProviderError
from utils.llm import ask

FAQ_HEADING = "Frequently Asked Questions"

SYSTEM_PROMPT = """You are an experienced human writer producing a genuinely useful blog article.

Hard rules:
- Write original prose. Never mention or imitate competitors.
- Do NOT invent statistics, studies, quotes, customer names, prices, dates, product features or URLs. If a number would help and you do not know it reliably, describe the point qualitatively instead.
- Say nothing about the company beyond its name. Where real company proof is needed (case study, original data, product detail, expert quote), insert a placeholder like [[COMPANY INPUT: short description]] instead of making something up. Use at most one placeholder per section.
- Mention the company name at most twice in the whole article, only where natural, never as a sales pitch.
- Plain, concrete language for the stated audience. Short paragraphs (1-4 sentences). Vary sentence length. Use specific examples and actionable steps. Use lists only where they genuinely help.
- Avoid filler openers ("In today's fast-paced world"), clichés, and phrases like "In conclusion" or "Let's dive in".
- Use the primary keyword naturally; never stuff it. Variations are fine.
- Markdown only: ## for H2, ### for H3. No other heading levels except the single H1 when told to write it. No URLs.
"""


# ---------- batching ----------

def _blocks(o: Outline) -> list[dict]:
    blocks = [{"kind": "intro", "words": o.intro.target_words, "data": o.intro}]
    blocks += [{"kind": "section", "words": sec.target_words, "data": sec} for sec in o.sections]
    if o.faqs:
        blocks.append({"kind": "faq", "words": o.faq_words, "data": o.faqs})
    blocks.append({"kind": "conclusion", "words": o.conclusion.target_words, "data": o.conclusion})
    return blocks


def make_batches(blocks: list[dict], limit: int) -> list[list[dict]]:
    batches, cur, cur_words = [], [], 0
    for b in blocks:
        if cur and cur_words + b["words"] > limit:
            batches.append(cur)
            cur, cur_words = [], 0
        cur.append(b)
        cur_words += b["words"]
    if cur:
        batches.append(cur)
    return batches


def _block_text(b: dict, o: Outline) -> str:
    kind, d, words = b["kind"], b["data"], b["words"]
    if kind == "intro":
        return (
            f"BLOCK: Introduction (no heading; it sits directly under the H1)\n"
            f"Length: about {words} words\nPurpose: {d.purpose}\n"
            f"Key points: {'; '.join(d.key_points)}\n"
            f'Use the primary keyword "{o.primary_keyword}" naturally within the first 100 words.'
        )
    if kind == "section":
        h3 = "\n".join(f"  - ### {h.heading}: {h.notes}" for h in d.h3s) or "  (none)"
        kws = ", ".join(d.secondary_keywords) or "(none)"
        return (
            f"BLOCK: ## {d.h2}\nLength: about {words} words\nPurpose: {d.purpose}\n"
            f"Key points: {'; '.join(d.key_points)}\nSubheadings (use as ### if listed):\n{h3}\n"
            f"Work these secondary keywords in naturally: {kws}"
        )
    if kind == "faq":
        qs = "\n".join(f"  - {f.question} (notes: {f.answer_notes})" for f in d)
        return (
            f"BLOCK: ## {FAQ_HEADING}\nLength: about {words} words in total\n"
            f"For each question use a ### heading with the question text, then a direct "
            f"40-80 word answer.\n{qs}"
        )
    return (
        f"BLOCK: ## {d.heading}\nLength: about {words} words\nPurpose: {d.purpose}\n"
        f"Key points: {'; '.join(d.key_points)}\n"
        f'Mention "{o.primary_keyword}" once. End with a helpful, low-pressure next step.'
    )


def _batch_prompt(user: UserInput, o: Outline, batch: list[dict], first: bool,
                  headings_so_far: list[str], tail: str) -> str:
    placeholders = "; ".join(o.needs_company_input) or "(none listed)"
    parts = [
        f"Company: {user.company_name}\nAudience: {user.audience}\nTone: {user.tone}\n"
        f"Primary keyword: {o.primary_keyword}\nSearch intent: {o.search_intent}\n"
        f"Angle: {o.differentiation_angle}\n"
        f"Things only the company can supply (use placeholders for these): {placeholders}\n",
        f"Write ONLY the following blocks, in order, using the headings exactly as given.\n",
    ]
    if first:
        parts.append(f"Start with the H1 line exactly: # {o.h1}\n")
    else:
        parts.append("Do NOT write an H1 or repeat earlier content.\n")
    if headings_so_far:
        parts.append("Already written (do not repeat): " + " | ".join(headings_so_far) + "\n")
    if tail:
        parts.append(f"The article so far ends with:\n---\n{tail}\n---\nContinue smoothly from there.\n")
    parts.append("\n\n".join(_block_text(b, o) for b in batch))
    parts.append(
        "\n\nOutput format: the markdown only. Then, on its own line, write ===CLAIMS=== followed by "
        "a bullet list of every specific factual claim you made (numbers, named tools, laws, dates, "
        "'studies show' type statements) that a fact-checker should verify, or '- none'."
    )
    return "\n".join(parts)


# ---------- parsing & checks ----------

_CLAIMS_RE = re.compile(r"^\s*={3,}\s*CLAIMS\s*={3,}\s*$", re.I | re.M)


def strip_fences(raw: str) -> str:
    return re.sub(r"^```(?:markdown|md)?\s*\n|\n?```\s*$", "", raw.strip())


def split_claims(raw: str) -> tuple[str, list[str]]:
    parts = _CLAIMS_RE.split(strip_fences(raw), maxsplit=1)
    body = parts[0].strip()
    claims: list[str] = []
    if len(parts) > 1:
        for line in parts[1].splitlines():
            line = line.strip().lstrip("-*• ").strip()
            if line and line.lower() not in ("none", "none.", "n/a"):
                claims.append(line)
    return body, claims


def _norm(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()


def heading_set(text: str) -> set[str]:
    return {_norm(re.sub(r"^\s*#+\s*", "", ln)) for ln in text.splitlines() if ln.lstrip().startswith("#")}


def missing_headings(text: str, required: list[str]) -> list[str]:
    have = heading_set(text)
    return [h for h in required if _norm(h) not in have]


def _required(batch: list[dict], o: Outline, first: bool) -> list[str]:
    req = [o.h1] if first else []
    for b in batch:
        if b["kind"] == "section":
            req.append(b["data"].h2)
        elif b["kind"] == "faq":
            req.append(FAQ_HEADING)
        elif b["kind"] == "conclusion":
            req.append(b["data"].heading)
    return req


def find_placeholders(text: str) -> list[str]:
    return re.findall(r"\[\[[^\]]*\]\]", text)


def keyword_report(markdown: str, o: Outline) -> KeywordReport:
    kw = o.primary_keyword.lower()
    lines = markdown.splitlines()
    h1 = next((l[2:] for l in lines if l.startswith("# ")), "")
    body = " ".join(l for l in lines if not l.lstrip().startswith("#"))
    first100 = " ".join(body.split()[:100]).lower()
    h2_idx = [i for i, l in enumerate(lines) if l.startswith("## ")]
    conclusion = " ".join(lines[h2_idx[-1] + 1:]).lower() if h2_idx else ""

    low = markdown.lower()
    count = low.count(kw) if kw else 0
    total = max(count_words(markdown), 1)
    return KeywordReport(
        primary_keyword=o.primary_keyword,
        primary_count=count,
        density_pct=round(count * max(len(kw.split()), 1) / total * 100, 2),
        in_h1=kw in h1.lower(),
        in_first_100_words=kw in first100,
        in_h2_count=sum(1 for i in h2_idx if kw in lines[i].lower()),
        in_conclusion=kw in conclusion,
        secondary_used={k: low.count(k.lower()) for k in o.keyword_plan.secondary_map},
    )


# ---------- writing ----------

def _write_batch(user, o, batch, first, headings_so_far, tail) -> tuple[str, list[str], list[str]]:
    prompt = _batch_prompt(user, o, batch, first, headings_so_far, tail)
    required = _required(batch, o, first)
    body, claims, missing = "", [], []
    for attempt in range(2):
        raw = ask(prompt, SYSTEM_PROMPT, max_tokens=config.WRITER_MAX_TOKENS)
        body, claims = split_claims(raw)
        missing = missing_headings(body, required)
        if not missing:
            break
        if attempt == 0:
            print(f"  Missing headings {missing}; retrying this part once.")
            prompt += ("\n\nIMPORTANT: your previous attempt left out these headings: "
                       f"{missing}. Include every heading exactly as given.")
            time.sleep(config.LLM_CALL_DELAY_SECONDS)

    warns = []
    if missing:
        warns.append("Some headings were still missing after a retry: " + "; ".join(missing))
    target = sum(b["words"] for b in batch)
    if count_words(body) < target * 0.5:
        warns.append(f"One part came out much shorter than planned ({count_words(body)} vs ~{target} words).")
    return body, claims, warns


def run_writer(user: UserInput, o: Outline) -> Draft:
    batches = make_batches(_blocks(o), config.BATCH_WORD_LIMIT)
    texts: list[str] = []
    claims: list[str] = []
    warnings: list[str] = []
    headings_so_far: list[str] = []

    for i, batch in enumerate(batches):
        print(f"Writing part {i + 1}/{len(batches)}...")
        tail = texts[-1][-800:] if texts else ""
        body, c, w = _write_batch(user, o, batch, i == 0, headings_so_far, tail)
        texts.append(body)
        claims.extend(c)
        warnings.extend(w)
        headings_so_far.extend(_required(batch, o, False))
        if i < len(batches) - 1:
            time.sleep(config.LLM_CALL_DELAY_SECONDS)

    markdown = re.sub(r"\n{3,}", "\n\n", "\n\n".join(texts)).strip() + "\n"
    h1_lines = [l for l in markdown.splitlines() if re.match(r"^#\s", l)]
    if not h1_lines:
        markdown = f"# {o.h1}\n\n" + markdown
        warnings.append("The draft had no H1, so the planned one was added.")
    elif len(h1_lines) > 1:
        warnings.append(f"The draft has {len(h1_lines)} H1 headings; there should be exactly one.")

    wc = count_words(markdown)
    if o.target_word_count and abs(wc - o.target_word_count) / o.target_word_count > 0.15:
        warnings.append(f"Length is {wc} words versus a target of {o.target_word_count} (more than 15% off).")

    report = keyword_report(markdown, o)
    if not report.in_h1:
        warnings.append("Primary keyword is missing from the H1.")
    if not report.in_first_100_words:
        warnings.append("Primary keyword is missing from the first 100 words.")
    if report.primary_count < 3:
        warnings.append(f"Primary keyword appears only {report.primary_count} time(s).")
    if report.density_pct > 2.5:
        warnings.append(f"Keyword density is {report.density_pct}%, which may read as stuffing.")

    placeholders = find_placeholders(markdown)
    if placeholders:
        warnings.append(f"{len(placeholders)} placeholder(s) need real company input before publishing.")

    return Draft(
        title=o.h1, markdown=markdown, word_count=wc, target_word_count=o.target_word_count,
        keyword_report=report, claims_to_verify=claims, placeholders=placeholders, warnings=warnings,
    )


def print_draft_summary(d: Draft) -> None:
    k = d.keyword_report
    print("\n=== Draft ===")
    print(f"{d.title}")
    print(f"Words: {d.word_count} (target {d.target_word_count})")
    print(f'Keyword "{k.primary_keyword}": {k.primary_count}x, density {k.density_pct}%, '
          f"H1={k.in_h1}, first100={k.in_first_100_words}, in {k.in_h2_count} H2(s), conclusion={k.in_conclusion}")
    print(f"Claims to verify: {len(d.claims_to_verify)} | Placeholders: {len(d.placeholders)}")
    for w in d.warnings:
        print(f"WARNING: {w}")