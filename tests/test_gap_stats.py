import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from models.schemas import ScrapedPage, SEOBrief
from modules.gap_analysis import (compute_page_stats, aggregate_stats, select_usable,
                                  build_analysis_from_llm)

def page(url, words, h2, text="email marketing tips " * 20, title="Email Marketing Guide"):
    return ScrapedPage(url=url, status="ok", title=title, meta_description="email marketing basics",
                       h1=["Email Marketing Guide"], h2=h2, h3=[], text=text, word_count=words)

pages = [
    page("https://a.com/1", 1000, ["Email marketing basics", "Build your list"]),
    page("https://b.com/2", 1400, ["Build your list", "Subject lines"]),
    ScrapedPage(url="https://c.com/3", status="blocked_by_robots"),
    page("https://d.com/4", 80, ["Tiny"]),   # too thin, must be excluded
]

usable = select_usable(pages)
assert [p.url for p in usable] == ["https://a.com/1", "https://b.com/2"], usable

stats = [compute_page_stats(p, "email marketing") for p in usable]
assert stats[0].keyword_in_title and stats[0].keyword_in_h1 and stats[0].keyword_in_meta
assert stats[0].keyword_in_h2_count == 1
assert stats[0].keyword_count == 20

agg = aggregate_stats(usable, stats)
assert agg.median_word_count == 1200 and agg.recommended_word_count == 1300, agg
assert "list" in agg.top_heading_terms          # appears in both pages' headings

brief = SEOBrief(primary_keyword="email marketing", secondary_keywords=[], search_intent="informational",
                 content_angle="x", suggested_title="x", questions_to_answer=["What is email marketing?"])
messy = {
    "search_intent": "WEIRD",                                   # invalid -> falls back to brief
    "common_topics": [{"topic": "List building", "covered_by": [1, 2, 99, True, "x"]}, "junk"],
    "content_gaps": [{"topic": "Deliverability", "priority": "HIGH"},
                     {"topic": "Templates", "priority": "bogus"}, {"no_topic": 1}],
    "faq_candidates": ["What is email marketing?", "How often to send?"],   # duplicate of brief's
    "competitor_weaknesses": "not a list",
}
res = build_analysis_from_llm(messy, brief, stats, agg, [p.url for p in usable], "full_pages", [])
assert res.search_intent == "informational"
assert res.common_topics[0].covered_by == ["https://a.com/1", "https://b.com/2"]
assert [g.priority for g in res.content_gaps] == ["high", "medium"]
assert len(res.faq_candidates) == 2 and res.competitor_weaknesses == []
print("All gap-analysis tests passed.")