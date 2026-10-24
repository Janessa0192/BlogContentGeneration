import config
from typing import Dict
from models.schemas import (
    UserInput, SEOBrief, GapAnalysis, Outline, OutlineSection, OutlineH3, FAQItem,
    IntroPlan, ConclusionPlan, KeywordPlan,
)
from utils.clean import s, as_list, dedupe, to_int, slugify
from utils.errors import ProviderError
from utils.llm import ask_json


SYSTEM_PROMPT = (
    "You are a senior SEO content strategist who plans articles that satisfy search intent "
    "and read naturally. Plan only from the material provided. Never invent facts, statistics "
    "or claims about the company."
)


# ---------- deterministic helpers ----------

def normalize_budget(raw: list[int], total: int, floor: int = 40) -> list[int]:
    """Rescale word targets so they sum to `total` (rounded to 10s, none below `floor`)."""
    raw = [max(r, floor) for r in raw]
    scale = total / sum(raw)
    scaled = [max(floor, int(round(r * scale / 10.0)) * 10) for r in raw]
    diff = total - sum(scaled)
    i = scaled.index(max(scaled))
    scaled[i] = max(floor, scaled[i] + diff)
    return scaled


def build_keyword_plan(brief: SEOBrief, sections: list[OutlineSection], word_count: int) -> KeywordPlan:
    secondary_map: dict[str, str] = {}
    for sec in sections:
        for k in sec.secondary_keywords:
            secondary_map.setdefault(k, sec.h2)
    unplaced = [k for k in brief.secondary_keywords if k not in secondary_map]

    target = max(3, round(word_count / 250))
    slug = slugify(brief.primary_keyword)
    pk = brief.primary_keyword
    rules = [
        f'H1: include "{pk}" once, ideally near the start.',
        f'Introduction: use "{pk}" within the first 100 words.',
        f'H2 headings: include "{pk}" or a close variation in 1-2 headings, not all of them.',
        f'Body: about {target} natural mentions across the article. Variations are fine; '
        "never force the exact phrase.",
        f'Conclusion: mention "{pk}" once.',
        f"URL slug: /{slug}",
        "Meta title and description (generated in Part 9): lead with the primary keyword.",
    ]
    return KeywordPlan(primary_keyword=pk, target_mentions=target, suggested_slug=slug,
                       placement_rules=rules, secondary_map=secondary_map,
                       unplaced_secondary=unplaced)


# ---------- prompt ----------

def _build_prompt(user: UserInput, brief: SEOBrief, gap: GapAnalysis, gaps: list, n_sections: int) -> str:
    gap_lines = "\n".join(
        f"[{i}] ({g.priority}) {g.topic}: {g.why_it_matters} | How: {g.how_to_cover}"
        for i, g in enumerate(gaps, start=1)
    ) or "(none)"
    return (
        f"Company: {user.company_name}\nTopic: {user.topic}\nAudience: {user.audience}\n"
        f"Tone: {user.tone}\nTotal article length: {user.word_count} words\n"
        f"Primary keyword: {brief.primary_keyword}\n"
        f"Secondary keywords (choose from these only): {', '.join(brief.secondary_keywords)}\n"
        f"Search intent: {gap.search_intent} ({gap.intent_notes})\n"
        f"Differentiation angle: {gap.differentiation_angle}\n\n"
        f"Topics every competitor covers (cover them, but briefly): "
        f"{'; '.join(t.topic for t in gap.common_topics) or '(none)'}\n"
        f"Subtopics to include: {'; '.join(gap.subtopics_to_include) or '(none)'}\n"
        f"Competitor weaknesses to beat: {'; '.join(gap.competitor_weaknesses) or '(none)'}\n"
        f"Content gaps (numbered):\n{gap_lines}\n"
        f"FAQ candidates: {'; '.join(gap.faq_candidates) or '(none)'}\n\n"
        f"Plan the article and return JSON with exactly these keys:\n"
        f'- "h1": under 65 characters, includes the primary keyword, promises a clear benefit\n'
        f'- "intro": {{"purpose": str, "key_points": [str], "target_words": int}}\n'
        f'- "sections": about {n_sections} objects, in logical reading order, each '
        f'{{"h2": str, "purpose": str, "key_points": [3-5 str], '
        f'"h3s": [{{"heading": str, "notes": str}}] (0-3 items, only if the section needs them), '
        f'"target_words": int, "secondary_keywords": [0-3 from the provided list], '
        f'"covers_gaps": [gap numbers from above]}}\n'
        f'- "faq_words": int\n'
        f'- "faqs": 4-6 objects {{"question": str, "answer_notes": str}}\n'
        f'- "conclusion": {{"heading": str, "purpose": str, "key_points": [str], "target_words": int}}\n\n'
        "Rules: make headings specific and descriptive rather than generic; put the primary "
        "keyword (or a close variation) in only one or two H2s; cover all high-priority gaps "
        "somewhere; match the search intent; do not make claims about the company beyond its name."
    )


# ---------- cleaning LLM output ----------

def build_outline_from_llm(raw: dict, user: UserInput, brief: SEOBrief, gap: GapAnalysis,
                           warnings: list[str]) -> Outline:
    gaps = gap.content_gaps[:config.MAX_GAPS_IN_OUTLINE]
    canon = {k.lower(): k for k in brief.secondary_keywords}

    sections: list[OutlineSection] = []
    seen_h2: set[str] = set()
    for item in raw.get("sections", []) if isinstance(raw.get("sections"), list) else []:
        if not isinstance(item, dict) or not s(item.get("h2")):
            continue
        h2 = s(item["h2"])
        if h2.lower() in seen_h2:
            warnings.append(f'Duplicate H2 "{h2}" was dropped.')
            continue
        seen_h2.add(h2.lower())

        h3s = []
        for h in item.get("h3s", []) if isinstance(item.get("h3s"), list) else []:
            if isinstance(h, dict) and s(h.get("heading")):
                h3s.append(OutlineH3(heading=s(h["heading"]), notes=s(h.get("notes"))))
            elif isinstance(h, str) and h.strip():
                h3s.append(OutlineH3(heading=h.strip()))

        kws = dedupe([canon[k.lower()] for k in as_list(item.get("secondary_keywords"))
                      if k.lower() in canon], 4)

        covers: list[str] = []
        for n in item.get("covers_gaps", []) if isinstance(item.get("covers_gaps"), list) else []:
            if isinstance(n, int) and not isinstance(n, bool) and 1 <= n <= len(gaps):
                if gaps[n - 1].topic not in covers:
                    covers.append(gaps[n - 1].topic)

        sections.append(OutlineSection(
            h2=h2, purpose=s(item.get("purpose")),
            key_points=as_list(item.get("key_points"))[:6], h3s=h3s[:4],
            target_words=to_int(item.get("target_words"), 150),
            secondary_keywords=kws, covers_gaps=covers,
        ))

    if len(sections) < 2:
        raise ProviderError("Gemini returned too few usable outline sections. Please run again.")

    # FAQs: accept objects or plain strings, force a trailing "?", de-duplicate
    faqs: list[FAQItem] = []
    seen_q: set[str] = set()
    for f in raw.get("faqs", []) if isinstance(raw.get("faqs"), list) else []:
        q = s(f.get("question")) if isinstance(f, dict) else s(f)
        notes = s(f.get("answer_notes")) if isinstance(f, dict) else ""
        if not q:
            continue
        if not q.endswith("?"):
            q += "?"
        if q.lower() not in seen_q:
            seen_q.add(q.lower())
            faqs.append(FAQItem(question=q, answer_notes=notes))
    faqs = faqs[:6]
    if not faqs:
        warnings.append("Gemini returned no FAQs; using candidates from the gap analysis.")
        faqs = [FAQItem(question=q if q.endswith("?") else q + "?") for q in gap.faq_candidates[:5]]

    intro_raw = raw.get("intro") if isinstance(raw.get("intro"), dict) else {}
    concl_raw = raw.get("conclusion") if isinstance(raw.get("conclusion"), dict) else {}

    h1 = s(raw.get("h1"))
    if not h1:
        h1 = brief.suggested_title
        warnings.append("Gemini gave no H1; used the suggested title from the SEO brief.")

    # word budget: [intro] + sections + [faq] + [conclusion] rescaled to the requested length
    raw_budget = ([to_int(intro_raw.get("target_words"), 120)]
                  + [sec.target_words for sec in sections]
                  + [to_int(raw.get("faq_words"), 150) if faqs else 0, to_int(concl_raw.get("target_words"), 100)])
    scaled = normalize_budget(raw_budget, user.word_count)
    intro_words, sec_words, faq_words, concl_words = scaled[0], scaled[1:-2], scaled[-2], scaled[-1]
    for sec, w in zip(sections, sec_words):
        sec.target_words = w

    covered = {t for sec in sections for t in sec.covers_gaps}
    uncovered = [g.topic for g in gaps if g.priority == "high" and g.topic not in covered]
    if uncovered:
        warnings.append("High-priority gaps not covered by any section: " + "; ".join(uncovered))

    pk = brief.primary_keyword.lower()
    if pk not in h1.lower():
        warnings.append("The H1 does not contain the exact primary keyword.")
    if not any(pk in sec.h2.lower() for sec in sections):
        warnings.append("No H2 contains the primary keyword; consider adding it to one heading.")
    if gap.based_on == "snippets_only":
        warnings.append("Gap analysis was based on search snippets only, so this outline is lower confidence.")

    plan = build_keyword_plan(brief, sections, user.word_count)
    if plan.unplaced_secondary:
        warnings.append("Secondary keywords not assigned to any section: " + ", ".join(plan.unplaced_secondary))

    return Outline(
        primary_keyword=brief.primary_keyword,
        search_intent=gap.search_intent,
        differentiation_angle=gap.differentiation_angle,
        h1=h1,
        target_word_count=user.word_count,
        intro=IntroPlan(purpose=s(intro_raw.get("purpose")),
                        key_points=as_list(intro_raw.get("key_points"))[:5],
                        target_words=intro_words),
        sections=sections,
        faq_words=faq_words,
        faqs=faqs,
        conclusion=ConclusionPlan(heading=s(concl_raw.get("heading")) or "Conclusion",
                                  purpose=s(concl_raw.get("purpose")),
                                  key_points=as_list(concl_raw.get("key_points"))[:4],
                                  target_words=concl_words),
        keyword_plan=plan,
        uncovered_gaps=uncovered,
        needs_company_input=gap.needs_company_input[:6],
        warnings=warnings,
    )


# ---------- entry points ----------

def run_outline(user: UserInput, brief: SEOBrief, gap: GapAnalysis) -> Outline:
    n_sections = max(3, min(9, round(user.word_count / 250)))
    gaps = gap.content_gaps[:config.MAX_GAPS_IN_OUTLINE]
    print(f"Generating outline (~{n_sections} sections) with Gemini...")
    raw = ask_json(_build_prompt(user, brief, gap, gaps, n_sections), SYSTEM_PROMPT, max_tokens=8192)
    if not isinstance(raw, dict):
        raise ProviderError("Gemini returned an unexpected structure for the outline. Try again.")
    return build_outline_from_llm(raw, user, brief, gap, [])


def print_outline_summary(o: Outline) -> None:
    print("\n=== Outline ===")
    print(f"H1: {o.h1}   (slug: /{o.keyword_plan.suggested_slug})")
    print(f"  Intro ({o.intro.target_words}w)")
    for sec in o.sections:
        print(f"  H2: {sec.h2} ({sec.target_words}w)")
        for h in sec.h3s:
            print(f"      H3: {h.heading}")
    print(f"  FAQ ({o.faq_words}w): " + " | ".join(f.question for f in o.faqs))
    print(f"  {o.conclusion.heading} ({o.conclusion.target_words}w)")
    print(f"Primary keyword target: ~{o.keyword_plan.target_mentions} mentions")
    for w in o.warnings:
        print(f"WARNING: {w}")