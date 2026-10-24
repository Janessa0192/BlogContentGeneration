import re
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import config
config.LLM_CALL_DELAY_SECONDS = 0

from models.schemas import (Outline, IntroPlan, OutlineSection, ConclusionPlan, KeywordPlan, FAQItem,
                            Draft, KeywordReport, HumanizeResult, StyleMetrics, MetaTags, UserInput,
                            Check, SearchResult)
from utils import textstats as ts
from utils.clean import count_words
from modules import humanizer as hz
from modules import fact_check as fc
from modules import qa_audit as qa
from modules import final_output as fo

SAMPLE = """# Email Marketing for Small Business: A Simple Guide

Email marketing is the cheapest way for a small shop to stay in touch with customers. This guide shows you how to start without a big budget or a design team.

## Build Your List

Start with an email list of people who actually want to hear from you. Put a simple sign-up form on your website and offer something useful in return. Keep it short. People rarely fill out long forms.

## Write Better Subject Lines

A good subject line is short and clear. Tell readers what they get when they open the message. Test two versions and keep the one that wins.

## Frequently Asked Questions

### How often should I send?

Once a week is a good start for most small businesses.

### Is it worth it?

Yes, when you keep messages useful and send them to people who asked for them.

## Conclusion

Email marketing rewards patience. Start small, learn from each send, and improve a little every week.
"""

outline = Outline(
    primary_keyword="email marketing", h1="Email Marketing for Small Business: A Simple Guide",
    target_word_count=count_words(SAMPLE), intro=IntroPlan(),
    sections=[OutlineSection(h2="Build Your List"), OutlineSection(h2="Write Better Subject Lines")],
    faqs=[FAQItem(question="How often should I send?")], conclusion=ConclusionPlan(),
    keyword_plan=KeywordPlan(primary_keyword="email marketing", target_mentions=3,
                             suggested_slug="email-marketing-small-business",
                             secondary_map={"email list": "Build Your List"}),
)
user = UserInput(company_name="Acme", topic="email", audience="owners", word_count=100)

# ---------- textstats ----------
sents = ts.split_sentences("This is one. Here is another one! Is this the third? Yes, e.g. this works. Price is 3.5 dollars.")
assert len(sents) == 5, sents
easy = ts.flesch_reading_ease("The cat sat on the mat. It was a big cat. The dog ran to the cat. They had fun.")
hard = ts.flesch_reading_ease("Notwithstanding considerable institutional complexities, organizational stakeholders "
                              "systematically operationalize interdisciplinary methodologies throughout multinational infrastructures.")
assert easy > 70 and hard < easy - 50, (easy, hard)
m = ts.style_metrics("# T\n\nLet's dive in and delve into it. It's important to note this.\n", "x")
assert "delve" in m.cliches_found and "let's dive in" in m.cliches_found and "it's important to note" in m.cliches_found
op = ts.repeated_openers(["You can do this.", "You can try that.", "You can see it.", "You can go."])
assert any("you can" in x for x in op), op
rp = ts.repeated_phrases("We recommend starting small. We recommend starting small again. "
                         "Also we recommend starting small today.")
assert "we recommend starting small" in rp, rp

# ---------- humanizer guards ----------
orig = "## Step One\n\nSend 3 emails per week to 1,000 subscribers. See [guide](https://x.com/a). [[COMPANY INPUT: case]]\n"
good = "## Step One\n\nEach week, send 3 emails to your 1,000 subscribers. Here's the [guide](https://x.com/a). [[COMPANY INPUT: case]]\n"
assert hz.validate_chunk(orig, good, "") == []
assert any("numbers" in p for p in hz.validate_chunk(orig, good.replace("3 emails", "4 emails"), ""))
assert any("headings" in p for p in hz.validate_chunk(orig, good.replace("## Step One\n\n", ""), ""))
assert any("placeholders" in p for p in hz.validate_chunk(orig, good.replace("[[COMPANY INPUT: case]]", ""), ""))
assert any("links" in p for p in hz.validate_chunk(orig, good.replace("x.com/a", "x.com/b"), ""))
assert hz.validate_chunk(orig, "", "") == ["empty reply"]
kw_orig = "## A\n\nUse email marketing well and often with care today for real results."
assert any("keyword" in p for p in hz.validate_chunk(kw_orig, "## A\n\nDo it well and often with care today for real results.", "email marketing"))

p = hz.parse_chunks("===CHUNK 1===\n# T\n\nhello\n===CHUNK 2===\n## A\n\nworld\n", {1, 2})
assert p == {1: "# T\n\nhello", 2: "## A\n\nworld"}, p
assert hz.parse_chunks("just text", {1}) == {1: "just text"}
assert hz.parse_chunks("just text", {1, 2}) == {}

md = "# Title\n\nIntro text.\n\n## A\n\nBody A.\n\n### Sub\n\nMore.\n\n## B\n\nBody B.\n"
ch = hz.split_chunks(md)
assert len(ch) == 3 and ch[1].startswith("## A") and "### Sub" in ch[1]
assert "\n\n".join(ch) + "\n" == md

# ---------- humanizer pipeline with a fake model ----------
def fake_identity(prompt, system="", max_tokens=0, json_mode=False):
    found = re.findall(r"===CHUNK (\d+)===\n(.*?)\n(?====CHUNK|\Z)", prompt, re.S)
    return "\n".join(f"===CHUNK {n}===\n{t}\n" for n, t in found)

def fake_bad(prompt, system="", max_tokens=0, json_mode=False):
    return fake_identity(prompt).replace("## Build Your List\n\n", "")

draft = Draft(title="x", markdown=SAMPLE, word_count=count_words(SAMPLE), target_word_count=100,
              keyword_report=KeywordReport(primary_keyword="email marketing", primary_count=0, density_pct=0.0,
                                           in_h1=False, in_first_100_words=False, in_h2_count=0, in_conclusion=False),
              claims_to_verify=["Claim A"])
hz.ask = fake_identity
res = hz.run_humanizer(user, outline, draft)
assert res.markdown.strip() == SAMPLE.strip()
assert res.chunks_kept_original == 0 and res.chunks_rewritten == res.chunks_total
assert res.claims_to_verify == ["Claim A"]

hz.ask = fake_bad
res = hz.run_humanizer(user, outline, draft)
assert res.chunks_kept_original == 1 and "## Build Your List" in res.markdown
assert any("Build Your List" in w for w in res.warnings), res.warnings

# ---------- fact check cleaning ----------
claims = ["A is true", "B is false", "C unknown"]
evidence = {1: [SearchResult(title="t", url="https://a.com", snippet="s")],
            2: [SearchResult(title="t2", url="https://b.com", snippet="s")], 3: []}
raw = {"results": [{"claim_id": 1, "verdict": "SUPPORTED", "note": "ok", "evidence": [1]},
                   {"claim_id": 2, "verdict": "contradicted", "note": "x", "evidence": [5]},
                   {"claim_id": 3, "verdict": "weird"},
                   {"claim_id": 99, "verdict": "supported"}]}
res_c = fc.clean_results(raw, claims, evidence)
assert [r.verdict for r in res_c] == ["supported", "unverified", "unverified"]
assert res_c[0].sources == ["https://a.com"]
assert fc.verify_claims([], 5) == ([], [])

# ---------- QA audit ----------
def human_from(markdown, claims=None):
    return HumanizeResult(markdown=markdown, word_count=count_words(markdown), before=StyleMetrics(),
                          after=StyleMetrics(), chunks_total=1, chunks_rewritten=1,
                          chunks_kept_original=0, claims_to_verify=claims or [])

rep = qa.run_audit(user, outline, human_from(SAMPLE), online=False)
by = {c.id: c for c in rep.checks}
for cid, want in [("one_h1", "pass"), ("kw_h1", "pass"), ("kw_first100", "pass"), ("kw_conclusion", "pass"),
                  ("placeholders", "pass"), ("hierarchy", "pass"), ("h2_count", "pass"), ("faq", "pass"),
                  ("outline_coverage", "pass"), ("word_count", "pass"), ("links", "warn"), ("claims", "skip")]:
    assert by[cid].status == want, (cid, by[cid])

rep2 = qa.run_audit(user, outline, human_from(SAMPLE + "\n[[COMPANY INPUT: case study]]\n"), online=False)
assert {c.id: c for c in rep2.checks}["placeholders"].status == "fail"
assert fo.compute_blockers(rep2.checks)

rep3 = qa.run_audit(user, outline, human_from(SAMPLE + "\nSee [guide](https://example.com).\n"), online=False)
assert {c.id: c for c in rep3.checks}["links"].status == "pass"

assert qa._public_http("https://example.com") and qa._public_http("http://8.8.8.8")
assert not qa._public_http("http://localhost/x") and not qa._public_http("http://192.168.1.1")
assert qa.parse_headings("# A\n```\n# not heading\n```\n## B\n") == [(1, "A"), (2, "B")]

# ---------- scoring ----------
cs = [Check(id="a", category="X", name="a", status="pass", points=10),
      Check(id="b", category="X", name="b", status="warn", points=10),
      Check(id="c", category="Y", name="c", status="fail", points=10),
      Check(id="d", category="Y", name="d", status="skip", points=10)]
score, cat = qa.compute_score(cs)
assert score == 50 and cat == {"X": 75, "Y": 0}, (score, cat)
assert qa.grade(95) == "A" and qa.grade(59) == "F"
total = sum(c.points for c in rep.checks) + 15
assert total == 100, total        # article checks (85) + meta checks (15)

# ---------- meta & final helpers ----------
desc = ("email marketing " * 10)[:140]
mt = MetaTags(meta_title="Email Marketing for Small Business: Start Here", meta_description=desc,
              slug="email-marketing-small-business")
assert all(c.status == "pass" for c in qa.meta_checks(mt, outline)), qa.meta_checks(mt, outline)
bad = qa.meta_checks(MetaTags(meta_title="x" * 80, meta_description="short", slug="Bad Slug"), outline)
assert [c.status for c in bad].count("fail") >= 3
assert len(fo.fit("word " * 50, 30)) <= 30 and fo.fit("short", 30) == "short"
assert 'title: "Email Marketing' in fo.front_matter(mt)
sugg = fo.build_suggestions(bad, outline)
assert any("image" in x for x in sugg)

print("All Part 7-9 tests passed.")