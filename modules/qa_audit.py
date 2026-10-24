import ipaddress
import re
import statistics
from collections import Counter
from urllib.parse import urlparse

import requests

import config
from models.schemas import (Check, QAReport, HumanizeResult, Outline, UserInput, MetaTags,
                            ClaimResult)
from modules.fact_check import verify_claims
from modules.seo_research import get_domain
from modules.writer import keyword_report, find_placeholders, FAQ_HEADING
from utils.clean import count_words
from utils.textstats import (prose_text, flesch_reading_ease, sentence_lengths,
                             paragraph_sentence_counts, style_metrics, norm_text)

KW, STRUCT, CONTENT, READ, META, TRUST = ("Keyword", "Structure", "Content",
                                          "Readability", "Meta", "Trust")


def _c(id, cat, name, pts, status, detail="", suggestion="") -> Check:
    return Check(id=id, category=cat, name=name, points=pts, status=status, detail=detail,
                 suggestion=suggestion if status in ("warn", "fail") else "")


def parse_headings(md: str) -> list[tuple[int, str]]:
    out, in_code = [], False
    for ln in md.splitlines():
        if ln.strip().startswith("```"):
            in_code = not in_code
            continue
        m = None if in_code else re.match(r"^(#{1,6})[ \t]+(.*?)[ \t]*$", ln)
        if m:
            out.append((len(m.group(1)), m.group(2).strip()))
    return out


# ---------- keyword ----------

def keyword_checks(md: str, outline: Outline, heads) -> list[Check]:
    r = keyword_report(md, outline)
    kw = outline.primary_keyword
    n_h2 = sum(1 for lvl, _ in heads if lvl == 2)
    d = r.density_pct

    if r.in_h2_count == 0:
        h2_status = "fail"
    elif n_h2 >= 4 and r.in_h2_count > 0.6 * n_h2:
        h2_status = "warn"
    else:
        h2_status = "pass"

    if 0.5 <= d <= 2.0:
        dens = "pass"
    elif 0.3 <= d < 0.5 or 2.0 < d <= 2.5:
        dens = "warn"
    else:
        dens = "fail"

    out = [
        _c("kw_h1", KW, "Primary keyword in H1", 5, "pass" if r.in_h1 else "fail",
           "Found in the H1." if r.in_h1 else "Not in the H1.",
           f'Put "{kw}" in the H1, ideally near the start.'),
        _c("kw_first100", KW, "Keyword in first 100 words", 4,
           "pass" if r.in_first_100_words else "fail",
           "Found." if r.in_first_100_words else "Missing from the opening.",
           f'Mention "{kw}" naturally in the first paragraph.'),
        _c("kw_h2", KW, "Keyword in an H2", 4, h2_status,
           f"In {r.in_h2_count} of {n_h2} H2 headings.",
           f'Use "{kw}" or a close variation in one or two H2s, not all of them.'),
        _c("kw_density", KW, "Keyword density", 7, dens,
           f"{r.primary_count} mentions, {d}% density (target 0.5-2.0%).",
           "Add a few natural mentions or variations." if d < 0.5
           else "Reduce exact repeats and use synonyms; it may read as stuffing."),
        _c("kw_conclusion", KW, "Keyword in conclusion", 2,
           "pass" if r.in_conclusion else "warn",
           "Found." if r.in_conclusion else "Not in the final section.",
           f'Mention "{kw}" once in the conclusion.'),
    ]
    mapped = list(outline.keyword_plan.secondary_map)
    if mapped:
        used = sum(1 for k in mapped if r.secondary_used.get(k, 0) > 0)
        ratio = used / len(mapped)
        out.append(_c("kw_secondary", KW, "Secondary keywords used", 3,
                      "pass" if ratio >= 0.5 else "warn" if ratio >= 0.25 else "fail",
                      f"{used} of {len(mapped)} planned secondary keywords appear.",
                      "Work the missing secondary keywords into their planned sections: "
                      + ", ".join(k for k in mapped if r.secondary_used.get(k, 0) == 0)))
    else:
        out.append(_c("kw_secondary", KW, "Secondary keywords used", 3, "skip",
                      "No secondary keywords were planned."))
    return out


# ---------- structure ----------

def structure_checks(heads, outline: Outline) -> list[Check]:
    h1s = [t for lvl, t in heads if lvl == 1]
    out = [_c("one_h1", STRUCT, "Exactly one H1", 5, "pass" if len(h1s) == 1 else "fail",
              f"Found {len(h1s)} H1 heading(s).", "Use exactly one H1 for the article title.")]

    if len(h1s) == 1:
        n = len(h1s[0])
        out.append(_c("h1_length", STRUCT, "H1 length", 3,
                      "pass" if n <= 65 else "warn" if n <= 80 else "fail",
                      f"{n} characters (aim for 65 or fewer).",
                      "Shorten the H1 so it fits comfortably in search results and on mobile."))
    else:
        out.append(_c("h1_length", STRUCT, "H1 length", 3, "skip", "Needs exactly one H1."))

    problems, prev = [], 0
    for lvl, text in heads:
        if not text:
            problems.append("an empty heading")
        if lvl > prev + 1:
            problems.append(f'skips from H{prev} to H{lvl} at "{text[:40]}"')
        if lvl >= 4:
            problems.append(f'uses H{lvl} ("{text[:40]}")')
        prev = lvl
    out.append(_c("hierarchy", STRUCT, "Heading hierarchy", 5, "pass" if not problems else "fail",
                  "Clean H1 > H2 > H3 structure." if not problems else "; ".join(problems[:3]),
                  "Fix the heading levels so they nest in order (H1, H2, H3) with no empty headings."))

    n_h2 = sum(1 for lvl, _ in heads if lvl == 2)
    out.append(_c("h2_count", STRUCT, "Enough H2 sections", 3,
                  "pass" if n_h2 >= 3 else "warn" if n_h2 == 2 else "fail",
                  f"{n_h2} H2 headings.", "Break the article into at least 3 clear H2 sections."))

    norms = Counter(norm_text(t) for lvl, t in heads if lvl >= 2)
    dups = [k for k, v in norms.items() if v > 1]
    out.append(_c("dup_headings", STRUCT, "No duplicate headings", 2, "pass" if not dups else "warn",
                  "All headings are unique." if not dups else "Repeated: " + ", ".join(dups[:3]),
                  "Make each heading unique and specific."))

    if outline.faqs:
        h3_n = sum(1 for lvl, _ in heads if lvl == 3)
        has = any(norm_text(t) == norm_text(FAQ_HEADING) for lvl, t in heads if lvl == 2)
        ok = has and h3_n >= min(3, len(outline.faqs))
        out.append(_c("faq", STRUCT, "FAQ section present", 2, "pass" if ok else "warn",
                      "FAQ section found." if ok else "FAQ section missing or too thin.",
                      "Keep a FAQ section with each question as its own H3."))
    else:
        out.append(_c("faq", STRUCT, "FAQ section present", 2, "skip", "No FAQs were planned."))
    return out


# ---------- content & links ----------

def _public_http(url: str) -> bool:
    p = urlparse(url)
    host = (p.hostname or "").lower()
    if p.scheme not in ("http", "https") or not host or host == "localhost" or host.endswith(".local"):
        return False
    try:
        return ipaddress.ip_address(host).is_global
    except ValueError:
        return True


def _check_url(url: str) -> str:
    headers = {"User-Agent": config.USER_AGENT}
    try:
        resp = requests.head(url, headers=headers, allow_redirects=True, timeout=config.REQUEST_TIMEOUT)
        if resp.status_code in (403, 405, 501) or resp.status_code >= 500:
            resp = requests.get(url, headers=headers, stream=True, allow_redirects=True,
                                timeout=config.REQUEST_TIMEOUT)
            resp.close()
        code = resp.status_code
    except requests.exceptions.Timeout:
        return "unclear"
    except requests.exceptions.ConnectionError:
        return "broken"
    except requests.RequestException:
        return "unclear"
    if code in (404, 410):
        return "broken"
    return "unclear" if code >= 400 else "ok"


def link_checks(md: str, user: UserInput, online: bool) -> list[Check]:
    links = re.findall(r"(?<!!)\[[^\]]*\]\(([^)\s]+)\)", md)
    if not links:
        return [_c("links", CONTENT, "Links", 3, "warn", "No links in the article.",
                   "Add 2-3 internal links to your own pages and 1-2 reputable external sources.")]
    ext = [u for u in links if u.startswith("http")]
    if not online or not ext:
        return [_c("links", CONTENT, "Links", 3, "pass",
                   f"{len(links)} link(s) found" + ("" if online else " (not checked online)") + ".")]
    results = {u: _check_url(u) for u in ext[:config.MAX_LINKS_TO_CHECK] if _public_http(u)}
    broken = [u for u, r in results.items() if r == "broken"]
    unclear = [u for u, r in results.items() if r == "unclear"]
    if broken:
        return [_c("links", CONTENT, "Links", 3, "fail", "Broken: " + ", ".join(broken[:3]),
                   "Fix or remove the broken links.")]
    if unclear:
        return [_c("links", CONTENT, "Links", 3, "warn",
                   "Could not confirm: " + ", ".join(unclear[:3]),
                   "Open these links by hand to confirm they work.")]
    return [_c("links", CONTENT, "Links", 3, "pass", f"{len(links)} link(s), none broken.")]


def content_checks(md: str, outline: Outline, user: UserInput, heads, online: bool) -> list[Check]:
    wc = count_words(md)
    target = outline.target_word_count or wc
    diff = abs(wc - target) / max(target, 1)
    wc_check = _c("word_count", CONTENT, "Word count near target", 4,
                  "pass" if diff <= 0.10 else "warn" if diff <= 0.20 else "fail",
                  f"{wc} words versus a target of {target}.",
                  f"Add about {target - wc} words of useful detail." if wc < target
                  else f"Trim about {wc - target} words of filler.")

    have = {norm_text(t) for _, t in heads}
    required = [sec.h2 for sec in outline.sections] + [outline.conclusion.heading]
    missing = [h for h in required if norm_text(h) not in have]
    cov = _c("outline_coverage", CONTENT, "Outline sections present", 3,
             "pass" if not missing else "warn" if len(missing) == 1 else "fail",
             "All planned sections are present." if not missing else "Missing: " + "; ".join(missing[:3]),
             "Restore the missing sections or update the outline to match the article.")
    return [wc_check, cov] + link_checks(md, user, online)


# ---------- readability ----------

def readability_checks(md: str) -> list[Check]:
    lens = sentence_lengths(md)
    if not lens:
        return [_c(i, READ, n, p, "skip", "No sentences found.")
                for i, n, p in (("flesch", "Reading ease", 6), ("sent_len", "Sentence length", 4),
                                ("paragraphs", "Paragraph length", 3), ("variety", "Sentence variety", 2))]
    avg = statistics.mean(lens)
    long_pct = 100 * sum(1 for n in lens if n > 30) / len(lens)
    fre = flesch_reading_ease(prose_text(md))
    paras = paragraph_sentence_counts(md)
    long_para_pct = 100 * sum(1 for n in paras if n > 5) / len(paras) if paras else 0.0
    sd = statistics.pstdev(lens) if len(lens) > 1 else 0.0

    return [
        _c("flesch", READ, "Reading ease (Flesch)", 6,
           "pass" if fre >= 50 else "warn" if fre >= 35 else "fail",
           f"Flesch score {fre} (50+ suits most general audiences).",
           "Use shorter sentences and simpler words unless your audience is highly technical."),
        _c("sent_len", READ, "Sentence length", 4,
           "pass" if avg <= 20 and long_pct <= 10 else "warn" if avg <= 25 and long_pct <= 20 else "fail",
           f"Average {avg:.1f} words; {long_pct:.0f}% of sentences exceed 30 words.",
           "Split the longest sentences."),
        _c("paragraphs", READ, "Paragraph length", 3,
           "pass" if long_para_pct <= 10 else "warn" if long_para_pct <= 25 else "fail",
           f"{long_para_pct:.0f}% of paragraphs have more than 5 sentences.",
           "Break long paragraphs into 2-4 sentence chunks."),
        _c("variety", READ, "Sentence variety", 2,
           "pass" if sd >= 5.0 else "warn" if sd >= 3.5 else "fail",
           f"Sentence-length spread (stdev) {sd:.1f}.",
           "Mix short, punchy sentences with longer ones."),
    ]


# ---------- trust ----------

def claims_check(flagged: list[str], results: list[ClaimResult], online: bool) -> Check:
    if not flagged:
        return _c("claims", TRUST, "Factual claims verified", 5, "skip",
                  "The writer flagged no factual claims. Still fact-check any numbers or names yourself.")
    if not online:
        return _c("claims", TRUST, "Factual claims verified", 5, "skip",
                  f"{len(flagged)} claim(s) flagged; verification skipped (--no-verify).")
    bad = [r for r in results if r.verdict == "contradicted"]
    unv = [r for r in results if r.verdict == "unverified"]
    if bad:
        return _c("claims", TRUST, "Factual claims verified", 5, "fail",
                  f"{len(bad)} claim(s) contradicted by sources: " + "; ".join(r.claim[:80] for r in bad[:2]),
                  "Correct or remove the contradicted claims.")
    if unv:
        return _c("claims", TRUST, "Factual claims verified", 5, "warn",
                  f"{len(unv)} of {len(results)} claim(s) could not be confirmed automatically.",
                  "Check the unverified claims against primary sources, or remove them.")
    return _c("claims", TRUST, "Factual claims verified", 5, "pass",
              f"All {len(results)} checked claim(s) are supported by sources.")


def trust_checks(md: str, kw: str) -> list[Check]:
    ph = find_placeholders(md)
    m = style_metrics(md, kw)
    n_rep = len(m.repeated_phrases) + len(m.repeated_openers)
    return [
        _c("placeholders", TRUST, "No leftover placeholders", 5, "pass" if not ph else "fail",
           "None." if not ph else f"{len(ph)} left: " + "; ".join(ph[:3]),
           "Replace every [[COMPANY INPUT: ...]] with real company material before publishing."),
        _c("ai_isms", TRUST, "No stock AI phrases", 3,
           "pass" if not m.cliches_found else "warn" if len(m.cliches_found) <= 2 else "fail",
           "None found." if not m.cliches_found else "Found: " + ", ".join(m.cliches_found[:4]),
           "Rewrite the stock phrases in plain, specific language."),
        _c("repetition", TRUST, "No robotic repetition", 2,
           "pass" if n_rep == 0 else "warn" if n_rep <= 2 else "fail",
           "None found." if n_rep == 0 else "; ".join((m.repeated_phrases + m.repeated_openers)[:3]),
           "Vary the repeated phrases and sentence openers."),
    ]


# ---------- meta (used by Part 9) ----------

def meta_checks(meta: MetaTags, outline: Outline) -> list[Check]:
    kw = outline.primary_keyword.lower()
    t, d = meta.meta_title, meta.meta_description
    lt, ld = len(t), len(d)
    slug_ok = bool(re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", meta.slug)) and len(meta.slug) <= 60
    return [
        _c("meta_title_len", META, "Meta title length", 4,
           "pass" if 30 <= lt <= 60 else "warn" if 20 <= lt <= 70 else "fail",
           f"{lt} characters (aim for 30-60).", "Rewrite the meta title to 30-60 characters."),
        _c("meta_title_kw", META, "Keyword in meta title", 3, "pass" if kw in t.lower() else "fail",
           "Included." if kw in t.lower() else "Missing.",
           f'Include "{outline.primary_keyword}" in the meta title, near the start.'),
        _c("meta_desc_len", META, "Meta description length", 4,
           "pass" if 120 <= ld <= 160 else "warn" if 90 <= ld <= 175 else "fail",
           f"{ld} characters (aim for 120-160).", "Rewrite the meta description to 120-160 characters."),
        _c("meta_desc_kw", META, "Keyword in meta description", 2,
           "pass" if kw in d.lower() else "fail", "Included." if kw in d.lower() else "Missing.",
           f'Mention "{outline.primary_keyword}" once in the meta description.'),
        _c("slug", META, "URL slug", 2, "pass" if slug_ok else "warn", f"/{meta.slug}",
           "Use a short lowercase slug with hyphens (60 characters or fewer)."),
    ]


# ---------- scoring ----------

def compute_score(checks: list[Check]) -> tuple[int, dict[str, int]]:
    earned = possible = 0.0
    cats: dict[str, list[float]] = {}
    for c in checks:
        if c.status == "skip":
            continue
        got = c.points if c.status == "pass" else c.points / 2 if c.status == "warn" else 0.0
        possible += c.points
        earned += got
        e = cats.setdefault(c.category, [0.0, 0.0])
        e[0] += got
        e[1] += c.points
    score = round(earned / possible * 100) if possible else 0
    return score, {k: round(v[0] / v[1] * 100) for k, v in cats.items() if v[1]}


def grade(score: int) -> str:
    return "A" if score >= 90 else "B" if score >= 80 else "C" if score >= 70 else "D" if score >= 60 else "F"


# ---------- main ----------

def run_audit(user: UserInput, outline: Outline, human: HumanizeResult, online: bool = True) -> QAReport:
    md = human.markdown
    heads = parse_headings(md)
    warnings: list[str] = []

    claim_results: list[ClaimResult] = []
    if human.claims_to_verify:
        if online:
            print("Verifying claims...")
            claim_results, vw = verify_claims(human.claims_to_verify, config.MAX_CLAIMS_TO_VERIFY)
            warnings += vw
        else:
            claim_results = [ClaimResult(claim=c, note="Not verified (--no-verify).")
                             for c in human.claims_to_verify]

    checks = (keyword_checks(md, outline, heads) + structure_checks(heads, outline)
              + content_checks(md, outline, user, heads, online) + readability_checks(md)
              + trust_checks(md, outline.primary_keyword)
              + [claims_check(human.claims_to_verify, claim_results, online)])

    kr = keyword_report(md, outline)
    lens = sentence_lengths(md)
    stats = {
        "words": float(count_words(md)),
        "flesch": flesch_reading_ease(prose_text(md)),
        "avg_sentence_len": round(statistics.mean(lens), 1) if lens else 0.0,
        "keyword_mentions": float(kr.primary_count),
        "keyword_density_pct": kr.density_pct,
        "h2_count": float(sum(1 for lvl, _ in heads if lvl == 2)),
    }
    return QAReport(checks=checks, claim_results=claim_results, stats=stats, warnings=warnings)


def print_audit_summary(r: QAReport) -> None:
    counts = Counter(c.status for c in r.checks)
    print("\n=== QA audit ===")
    print(f"Checks: {counts['pass']} pass, {counts['warn']} warn, {counts['fail']} fail, "
          f"{counts['skip']} skipped")
    for c in r.checks:
        if c.status in ("fail", "warn"):
            print(f"  [{c.status.upper()}] {c.name}: {c.detail}")
    for w in r.warnings:
        print(f"WARNING: {w}")