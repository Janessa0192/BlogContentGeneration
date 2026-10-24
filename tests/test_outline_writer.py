import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from models.schemas import SEOBrief, GapAnalysis, ContentGap, UserInput
from modules.outline import normalize_budget, build_outline_from_llm
from modules.writer import (make_batches, split_claims, strip_fences, missing_headings,
                            find_placeholders, keyword_report)
from utils.clean import count_words, slugify
from utils.errors import ProviderError

# --- budget ---
assert sum(normalize_budget([100, 300, 300, 100, 100, 100], 1200)) == 1200
assert sum(normalize_budget([150] * 7, 1200)) == 1200
assert all(x >= 40 for x in normalize_budget([50] * 10, 300))

# --- small utils ---
assert count_words("Don't stop-believing now") == 3
assert slugify("Email Marketing Tips for Small Businesses!") == "email-marketing-tips-for-small-businesses"

# --- outline cleaning with messy LLM output ---
brief = SEOBrief(primary_keyword="email marketing",
                 secondary_keywords=["email list", "subject lines", "open rate"],
                 search_intent="informational", content_angle="x",
                 suggested_title="Email Marketing Guide", questions_to_answer=[])
gap = GapAnalysis(primary_keyword="email marketing", search_intent="informational",
                  based_on="full_pages",
                  content_gaps=[ContentGap(topic="Deliverability", priority="high"),
                                ContentGap(topic="Templates", priority="medium"),
                                ContentGap(topic="Compliance", priority="high")],
                  faq_candidates=["How often should I send emails?", "Is email marketing worth it?"],
                  needs_company_input=["case study"], differentiation_angle="practical")
user = UserInput(company_name="Acme", topic="email", audience="owners", word_count=1200)

raw = {
    "h1": "Email Marketing for Small Businesses: A Practical Guide",
    "intro": {"purpose": "hook", "key_points": ["why"], "target_words": 100},
    "sections": [
        {"h2": "Build Your Email List", "purpose": "p", "key_points": ["a"],
         "h3s": [{"heading": "Lead magnets", "notes": "n"}, "Opt-in forms"],
         "target_words": 300, "secondary_keywords": ["Email List", "bogus kw"],
         "covers_gaps": [1, 99, True]},
        {"h2": "Write Subject Lines", "target_words": "250 words",
         "secondary_keywords": ["subject lines"], "covers_gaps": [2]},
        {"h2": "build your email list", "target_words": 100},   # duplicate, dropped
        "junk",
    ],
    "faq_words": 150,
    "faqs": [{"question": "How often should I send emails", "answer_notes": "n"},
             "Is email marketing worth it?",
             {"question": "how often should i send emails"}],
    "conclusion": {"heading": "", "target_words": 100},
}
o = build_outline_from_llm(raw, user, brief, gap, [])
assert [s.h2 for s in o.sections] == ["Build Your Email List", "Write Subject Lines"]
assert o.sections[0].secondary_keywords == ["email list"]
assert o.sections[0].covers_gaps == ["Deliverability"]
assert [h.heading for h in o.sections[0].h3s] == ["Lead magnets", "Opt-in forms"]
assert o.sections[1].covers_gaps == ["Templates"]
assert o.uncovered_gaps == ["Compliance"]
assert [f.question for f in o.faqs] == ["How often should I send emails?", "Is email marketing worth it?"]
assert o.conclusion.heading == "Conclusion"
assert o.intro.target_words + sum(s.target_words for s in o.sections) + o.faq_words \
    + o.conclusion.target_words == 1200
assert o.keyword_plan.unplaced_secondary == ["open rate"]
assert o.keyword_plan.target_mentions == 5

o2 = build_outline_from_llm({**raw, "faqs": []}, user, brief, gap, [])
assert len(o2.faqs) == 2                      # fell back to gap FAQ candidates

try:
    build_outline_from_llm({"sections": ["junk"]}, user, brief, gap, [])
    raise SystemExit("expected ProviderError")
except ProviderError:
    pass

# --- batching ---
blocks = [{"words": w} for w in [120, 360, 360, 120, 120, 120]]
assert [len(b) for b in make_batches(blocks, 900)] == [3, 3]

# --- parsing ---
body, claims = split_claims("# T\n\ntext\n\n===CLAIMS===\n- Open rates average 20%\n- none\n")
assert body == "# T\n\ntext" and claims == ["Open rates average 20%"]
assert strip_fences("```markdown\n# Hi\n```") == "# Hi"
assert missing_headings("# My Title\n\n## Build Your List\n", ["My Title", "Build your list!"]) == []
assert missing_headings("# My Title\n", ["My Title", "Other"]) == ["Other"]
assert len(find_placeholders("a [[COMPANY INPUT: case study]] b [[X]]")) == 2

# --- keyword report ---
md = ("# Email Marketing for Small Businesses\n\n"
      "Email marketing helps small teams reach customers. We cover the basics here.\n\n"
      "## Build Your Email List\n\n"
      "Grow your email list with a simple form. Email marketing works when lists are healthy.\n\n"
      "## Conclusion\n\nStart small with email marketing today.\n")
r = keyword_report(md, o)
assert r.primary_count == 4 and r.in_h1 and r.in_first_100_words and r.in_conclusion
assert r.in_h2_count == 0 and r.secondary_used["email list"] == 2

print("All outline/writer tests passed.")