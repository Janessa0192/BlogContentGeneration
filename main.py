import sys
from pathlib import Path

import config
from models.schemas import (UserInput, ResearchResult, GapAnalysis, Outline, Draft,
                            HumanizeResult, QAReport)
from modules.user_input import collect_user_input
from modules.seo_research import run_seo_research
from modules.scraper import scrape_competitors
from modules.gap_analysis import run_gap_analysis, print_summary
from modules.outline import run_outline, print_outline_summary
from modules.writer import run_writer, print_draft_summary
from modules.humanizer import run_humanizer, print_humanize_summary
from modules.qa_audit import run_audit, print_audit_summary
from modules.final_output import build_final, front_matter, render_report, print_final_summary
from utils.errors import ProviderError, ConfigError, redact
from utils.files import read_text, write_text

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)
INPUT_FILE = OUT / "user_input.json"
RESEARCH_FILE = OUT / "research.json"
GAP_FILE = OUT / "gap_analysis.json"
OUTLINE_FILE = OUT / "outline.json"
DRAFT_MD, DRAFT_JSON = OUT / "draft.md", OUT / "draft.json"
HUMAN_MD, HUMAN_JSON = OUT / "humanized.md", OUT / "humanized.json"
QA_JSON = OUT / "qa_report.json"
FINAL_MD, FINAL_JSON, REPORT_MD = OUT / "final_article.md", OUT / "final_report.json", OUT / "seo_report.md"

STAGES = ["research", "analysis", "outline", "write", "humanize", "audit", "final"]
FILES = {"research": RESEARCH_FILE, "analysis": GAP_FILE, "outline": OUTLINE_FILE,
         "write": DRAFT_JSON, "humanize": HUMAN_JSON, "audit": QA_JSON}
LEGACY = {"--skip-research": "analysis", "--skip-analysis": "outline", "--skip-outline": "write"}
KNOWN_FLAGS = {"--reuse", "--from", "--stop-after", "--outline-only", "--no-verify"} | set(LEGACY)


def _stage_arg(args: list[str], flag: str):
    if flag not in args:
        return None
    i = args.index(flag)
    value = args[i + 1] if i + 1 < len(args) else None
    if value not in STAGES:
        raise ConfigError(f"{flag} needs one of: {', '.join(STAGES)}")
    return value


def run():
    args = sys.argv[1:]
    for a in args:
        if a.startswith("--") and a not in KNOWN_FLAGS:
            print(f"Note: unknown flag {a} was ignored.")

    start = 0
    for flag, stage in LEGACY.items():
        if flag in args:
            start = max(start, STAGES.index(stage))
    from_stage = _stage_arg(args, "--from")
    if from_stage:
        start = max(start, STAGES.index(from_stage))
    stop_stage = _stage_arg(args, "--stop-after") or ("outline" if "--outline-only" in args else None)
    if stop_stage and STAGES.index(stop_stage) < start:
        raise ConfigError("--stop-after comes before the stage you are starting from.")
    online = "--no-verify" not in args

    missing = [FILES[st].name for st in STAGES[:start] if st in FILES and not FILES[st].exists()]
    if missing:
        raise ConfigError("Starting from '" + STAGES[start] + "' needs saved files: " + ", ".join(missing)
                          + ". Run the earlier stages first.")

    config.validate_keys(need_search=(start == 0))

    if INPUT_FILE.exists() and ("--reuse" in args or start > 0):
        user = UserInput.model_validate_json(read_text(INPUT_FILE))
        print(f"Reusing saved input for {user.company_name}")
    else:
        user = collect_user_input()
        write_text(INPUT_FILE, user.model_dump_json(indent=2))

    ctx: dict = {}

    def get(name, path, model):
        if name not in ctx:
            ctx[name] = model.model_validate_json(read_text(path))
        return ctx[name]

    def on(stage):
        return STAGES.index(stage) >= start

    def halt(stage):
        if stop_stage == stage:
            print(f"\nStopped after '{stage}'. Resume with: python main.py --from {STAGES[STAGES.index(stage) + 1]}")
            return True
        return False

    if on("research"):
        brief, competitors = run_seo_research(user)
        pages = scrape_competitors(competitors)
        ctx["research"] = ResearchResult(brief=brief, competitors=competitors, pages=pages)
        write_text(RESEARCH_FILE, ctx["research"].model_dump_json(indent=2))
        ok = sum(1 for p in pages if p.status == "ok")
        print(f"\nResearch done. {ok}/{len(pages)} pages scraped. Saved to {RESEARCH_FILE}")
        if halt("research"):
            return

    if on("analysis"):
        research = get("research", RESEARCH_FILE, ResearchResult)
        ctx["gap"] = run_gap_analysis(user, research)
        write_text(GAP_FILE, ctx["gap"].model_dump_json(indent=2))
        print_summary(ctx["gap"])
        if halt("analysis"):
            return

    if on("outline"):
        research = get("research", RESEARCH_FILE, ResearchResult)
        gap = get("gap", GAP_FILE, GapAnalysis)
        ctx["outline"] = run_outline(user, research.brief, gap)
        write_text(OUTLINE_FILE, ctx["outline"].model_dump_json(indent=2))
        print_outline_summary(ctx["outline"])
        print(f"\nSaved to {OUTLINE_FILE}")
        if halt("outline"):
            return

    outline = get("outline", OUTLINE_FILE, Outline)

    if on("write"):
        ctx["draft"] = run_writer(user, outline)
        write_text(DRAFT_MD, ctx["draft"].markdown)
        write_text(DRAFT_JSON, ctx["draft"].model_dump_json(indent=2))
        print_draft_summary(ctx["draft"])
        if halt("write"):
            return

    if on("humanize"):
        draft = get("draft", DRAFT_JSON, Draft)
        ctx["human"] = run_humanizer(user, outline, draft)
        write_text(HUMAN_MD, ctx["human"].markdown)
        write_text(HUMAN_JSON, ctx["human"].model_dump_json(indent=2))
        print_humanize_summary(ctx["human"])
        if halt("humanize"):
            return

    human = get("human", HUMAN_JSON, HumanizeResult)

    if on("audit"):
        ctx["qa"] = run_audit(user, outline, human, online=online)
        write_text(QA_JSON, ctx["qa"].model_dump_json(indent=2))
        print_audit_summary(ctx["qa"])
        if halt("audit"):
            return

    qa = get("qa", QA_JSON, QAReport)
    final = build_final(user, outline, human, qa)
    write_text(FINAL_MD, front_matter(final.meta) + final.article_markdown)
    write_text(FINAL_JSON, final.model_dump_json(indent=2))
    write_text(REPORT_MD, render_report(final))
    print_final_summary(final)
    print(f"\nSaved: {FINAL_MD}, {REPORT_MD}, {FINAL_JSON}")


if __name__ == "__main__":
    try:
        run()
    except ProviderError as e:
        print(f"\nERROR [{type(e).__name__}]: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {redact(e)[:300]}")
        sys.exit(1)