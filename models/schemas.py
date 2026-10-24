from pydantic import BaseModel, Field
from typing import Optional, List
from pydantic import BaseModel, Field
from typing import Optional, List, Dict

class UserInput(BaseModel):
    company_name: str
    website: Optional[str] = None
    target_keyword: Optional[str] = None
    topic: str
    audience: str
    tone: str = "professional but friendly"
    word_count: int = Field(default=1200, ge=300, le=5000)

class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str = ""
    position: int = 0

class SEOBrief(BaseModel):
    primary_keyword: str
    secondary_keywords: List[str]
    search_intent: str            # informational / commercial / transactional / navigational
    content_angle: str
    suggested_title: str
    questions_to_answer: List[str]

class ScrapedPage(BaseModel):
    url: str
    status: str                   # ok | blocked_by_robots | failed
    error: Optional[str] = None
    title: Optional[str] = None
    meta_description: Optional[str] = None
    canonical: Optional[str] = None
    h1: List[str] = []
    h2: List[str] = []
    h3: List[str] = []
    text: str = ""
    word_count: int = 0

class ResearchResult(BaseModel):
    brief: SEOBrief
    competitors: List[SearchResult]
    pages: List[ScrapedPage]


class PageStats(BaseModel):
    url: str
    title: Optional[str] = None
    word_count: int = 0
    h2_count: int = 0
    h3_count: int = 0
    keyword_in_title: bool = False
    keyword_in_h1: bool = False
    keyword_in_meta: bool = False
    keyword_in_h2_count: int = 0
    keyword_count: int = 0
    keyword_density_pct: float = 0.0

class AggregateStats(BaseModel):
    pages_analyzed: int
    median_word_count: int
    min_word_count: int
    max_word_count: int
    avg_h2_count: float
    recommended_word_count: int      # informational; user's word_count stays authoritative
    top_heading_terms: List[str]

class TopicCoverage(BaseModel):
    topic: str
    covered_by: List[str] = []       # competitor URLs

class ContentGap(BaseModel):
    topic: str
    why_it_matters: str = ""
    how_to_cover: str = ""
    priority: str = "medium"         # high | medium | low

class GapAnalysis(BaseModel):
    primary_keyword: str
    search_intent: str
    intent_notes: str = ""
    based_on: str                    # "full_pages" | "snippets_only"
    page_stats: List[PageStats] = []
    aggregate: Optional[AggregateStats] = None
    common_topics: List[TopicCoverage] = []
    content_gaps: List[ContentGap] = []
    competitor_weaknesses: List[str] = []
    subtopics_to_include: List[str] = []
    faq_candidates: List[str] = []
    differentiation_angle: str = ""
    needs_company_input: List[str] = []
    warnings: List[str] = []

class OutlineH3(BaseModel):
    heading: str
    notes: str = ""

class OutlineSection(BaseModel):
    h2: str
    purpose: str = ""
    key_points: List[str] = []
    h3s: List[OutlineH3] = []
    target_words: int = 150
    secondary_keywords: List[str] = []
    covers_gaps: List[str] = []

class FAQItem(BaseModel):
    question: str
    answer_notes: str = ""

class IntroPlan(BaseModel):
    purpose: str = ""
    key_points: List[str] = []
    target_words: int = 120

class ConclusionPlan(BaseModel):
    heading: str = "Conclusion"
    purpose: str = ""
    key_points: List[str] = []
    target_words: int = 100

class KeywordPlan(BaseModel):
    primary_keyword: str
    target_mentions: int
    suggested_slug: str
    placement_rules: List[str] = []
    secondary_map: Dict[str, str] = {}      # secondary keyword -> H2 where it goes
    unplaced_secondary: List[str] = []

class Outline(BaseModel):
    primary_keyword: str
    search_intent: str = ""
    differentiation_angle: str = ""
    h1: str
    target_word_count: int
    intro: IntroPlan
    sections: List[OutlineSection]
    faq_words: int = 150
    faqs: List[FAQItem] = []
    conclusion: ConclusionPlan
    keyword_plan: KeywordPlan
    uncovered_gaps: List[str] = []
    needs_company_input: List[str] = []
    warnings: List[str] = []

class KeywordReport(BaseModel):
    primary_keyword: str
    primary_count: int
    density_pct: float
    in_h1: bool
    in_first_100_words: bool
    in_h2_count: int
    in_conclusion: bool
    secondary_used: Dict[str, int] = {}

class Draft(BaseModel):
    title: str
    markdown: str
    word_count: int
    target_word_count: int
    keyword_report: KeywordReport
    claims_to_verify: List[str] = []
    placeholders: List[str] = []
    warnings: List[str] = []

class StyleMetrics(BaseModel):
    words: int = 0
    sentences: int = 0
    avg_sentence_len: float = 0.0
    sentence_len_stdev: float = 0.0
    avg_paragraph_sentences: float = 0.0
    repeated_openers: List[str] = []
    repeated_phrases: List[str] = []
    cliches_found: List[str] = []
    em_dash_count: int = 0

class HumanizeResult(BaseModel):
    markdown: str
    word_count: int
    before: StyleMetrics
    after: StyleMetrics
    chunks_total: int
    chunks_rewritten: int
    chunks_kept_original: int
    claims_to_verify: List[str] = []
    warnings: List[str] = []

class ClaimResult(BaseModel):
    claim: str
    verdict: str = "unverified"        # supported | contradicted | unverified
    note: str = ""
    sources: List[str] = []

class Check(BaseModel):
    id: str
    category: str
    name: str
    status: str                        # pass | warn | fail | skip
    points: int
    detail: str = ""
    suggestion: str = ""

class QAReport(BaseModel):
    checks: List[Check]
    claim_results: List[ClaimResult] = []
    stats: Dict[str, float] = {}
    warnings: List[str] = []

class MetaTags(BaseModel):
    meta_title: str
    meta_description: str
    slug: str
    source: str = "gemini"             # gemini | fallback

class FinalOutput(BaseModel):
    title: str
    meta: MetaTags
    article_markdown: str
    word_count: int
    seo_score: int
    grade: str
    category_scores: Dict[str, int]
    checks: List[Check]
    claim_results: List[ClaimResult]
    suggestions: List[str]
    ready_to_publish: bool
    blockers: List[str]
    warnings: List[str]