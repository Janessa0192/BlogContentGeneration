import json

from models.schemas import (Outline, UserInput, HumanizeResult, QAReport, MetaTags, FinalOutput,
                            Check)
from modules.qa_audit import compute_score, grade, meta_checks
from utils.clean import count_words, dedupe, s, slugify
from utils.llm import ask_json
from utils.textstats import prose_text, paragraphs
from utils.errors import ProviderError

BLOCKER_IDS = {"placeholders", "claims", "one_h1", "links"}
CATEGORY_ORDER = ["Keyword", "Structure", "Content", "Readability", "Meta", "Trust"]
MARK = {"pass": "[PASS]", "warn": "[WARN]", "fail": "[FAIL]", "skip": "[SKIP]"}

META_SYSTEM = ("You write SEO meta tags. Use only information present in the article. Do not invent "
               "claims, numbers, prices or offers.")


# ---------- meta ----------

def fit(text: str, limit: int) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].rstrip(",;:-. ")


def _clean_meta(v) -> str:
    return " ".join(s(v).strip("\"'`“”").split())


def _meta_problems(title: str, desc: str, kw: str) -> list[str]:
    p = []
    if not 30 <= len(title) <= 60:
        p.append(f"meta_title is {len(title)} characters (must be 30-60)")
    if kw not in title.lower():
        p.append("meta_title must include the primary keyword")
    if not 120 <= len(desc) <= 160:
        p.append(f"meta_description is {len(desc)} characters (must be 120-160)")
    if kw not in desc.lower():
        p.append("meta_description must include the primary keyword")
    return p


def fallback_meta(outline: Outline, markdown: str) -> tuple[str, str]:
    paras = paragraphs(markdown)
    return fit(outline.h1, 60), fit(paras[0] if paras else outline.h1, 155)


def generate_meta(user: UserInput, outline: Outline, markdown: str) -> MetaTags:
    kw = outline.primary_keyword.lower()
    base = (f"Article H1: {outline.h1}\nPrimary keyword: {outline.primary_keyword}\n"
            f"Audience: {user.audience}\nTone: {user.tone}\n\n"
            f"Article start:\n{prose_text(markdown)[:1800]}\n\n"
            'Return JSON: {"meta_title": str, "meta_description": str}. The title is 30-60 '
            "characters and includes the primary keyword near the start. The description is "
            "130-155 characters, includes the primary keyword once, and ends with a soft call to "
            "action. No quotation marks and no emoji.")
    title = desc = ""
    prompt = base
    for _ in range(2):
        try:
            raw = ask_json(prompt, META_SYSTEM, max_tokens=1024)
        except ProviderError:
            raise
        raw = raw if isinstance(raw, dict) else {}
        title, desc = _clean_meta(raw.get("meta_title")), _clean_meta(raw.get("meta_description"))
        problems = _meta_problems(title, desc, kw)
        if not problems:
            break
        prompt = base + "\n\nYour previous attempt had these problems; fix them: " + "; ".join(problems)

    source = "gemini"
    fb_title, fb_desc = fallback_meta(outline, markdown)
    if not title:
        title, source = fb_title, "fallback"
    if not desc:
        desc, source = fb_desc, "fallback"
    slug = outline.keyword_plan.suggested_slug or slugify(outline.primary_keyword)
    return MetaTags(meta_title=title, meta_description=desc, slug=slug, source=source)


# ---------- suggestions & blockers ----------

def compute_blockers(checks: list[Check]) -> list[str]:
    return [f"{c.name}: {c.detail}" for c in checks if c.id in BLOCKER_IDS and c.status == "fail"]


def build_suggestions(checks: list[Check], outline: Outline) -> list[str]:
    ranked = sorted((c for c in checks if c.status in ("warn", "fail") and c.suggestion),
                    key=lambda c: -(c.points if c.status == "fail" else c.points / 2))
    out = [c.suggestion for c in ranked]
    if outline.needs_company_input:
        out.append("Add real material only your company has (this builds trust): "
                   + "; ".join(outline.needs_company_input[:4]) + ".")
    if outline.faqs:
        out.append("Mark up the FAQ section with FAQPage schema if your CMS supports it.")
    out.append("Add a relevant featured image with descriptive alt text, compressed for fast loading.")
    out.append("Have a subject-matter expert read the final text; add a byline and a last-updated date.")
    return dedupe(out, 12)


# ---------- assembly ----------

def build_final(user: UserInput, outline: Outline, human: HumanizeResult, qa: QAReport) -> FinalOutput:
    print("Generating meta tags...")
    meta = generate_meta(user, outline, human.markdown)
    checks = qa.checks + meta_checks(meta, outline)
    score, cats = compute_score(checks)
    blockers = compute_blockers(checks)
    warnings = list(qa.warnings) + list(human.warnings)
    if meta.source == "fallback":
        warnings.append("Meta tags used a plain fallback; review them before publishing.")
    return FinalOutput(
        title=outline.h1, meta=meta, article_markdown=human.markdown,
        word_count=count_words(human.markdown), seo_score=score, grade=grade(score),
        category_scores=cats, checks=checks, claim_results=qa.claim_results,
        suggestions=build_suggestions(checks, outline), ready_to_publish=not blockers,
        blockers=blockers, warnings=warnings,
    )


def front_matter(meta: MetaTags) -> str:
    return ("---\n"
            f"title: {json.dumps(meta.meta_title, ensure_ascii=False)}\n"
            f"description: {json.dumps(meta.meta_description, ensure_ascii=False)}\n"
            f"slug: {meta.slug}\n---\n\n")


def render_report(f: FinalOutput) -> str:
    L = [f"# SEO Report: {f.title}\n",
         f"**SEO score: {f.seo_score}/100 (grade {f.grade})**  ",
         "_A heuristic on-page checklist. It is not a Google score or a ranking prediction._\n",
         f"**Ready to publish:** {'YES' if f.ready_to_publish else 'NO'}"]
    if f.blockers:
        L += ["", "Blockers to fix first:"] + [f"- {b}" for b in f.blockers]
    L += ["", f"**Meta title:** {f.meta.meta_title}", f"**Meta description:** {f.meta.meta_description}",
          f"**Slug:** /{f.meta.slug}", f"**Words:** {f.word_count}", "",
          "## Category scores", "", "| Category | Score |", "|---|---|"]
    L += [f"| {c} | {f.category_scores[c]}% |" for c in CATEGORY_ORDER if c in f.category_scores]
    L += ["", "## Checks"]
    for cat in CATEGORY_ORDER:
        rows = [c for c in f.checks if c.category == cat]
        if rows:
            L += ["", f"### {cat}"] + [f"- {MARK[c.status]} **{c.name}** ({c.points} pts): {c.detail}" for c in rows]
    if f.claim_results:
        L += ["", "## Claim verification"]
        for r in f.claim_results:
            src = f" Sources: {', '.join(r.sources)}" if r.sources else ""
            L.append(f"- **{r.verdict}**: {r.claim} {r.note}{src}")
    L += ["", "## Improvement suggestions"] + [f"{i}. {t}" for i, t in enumerate(f.suggestions, 1)]
    if f.warnings:
        L += ["", "## Warnings"] + [f"- {w}" for w in f.warnings]
    return "\n".join(L) + "\n"


def print_final_summary(f: FinalOutput) -> None:
    print("\n=== Final output ===")
    print(f"SEO score: {f.seo_score}/100 (grade {f.grade}) | words: {f.word_count}")
    print(f"Meta title ({len(f.meta.meta_title)}): {f.meta.meta_title}")
    print(f"Meta description ({len(f.meta.meta_description)}): {f.meta.meta_description}")
    print("Ready to publish: " + ("YES" if f.ready_to_publish else "NO"))
    for b in f.blockers:
        print(f"  BLOCKER: {b}")
    print("Top suggestions:")
    for t in f.suggestions[:4]:
        print(f"  - {t}")