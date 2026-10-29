import html
import json
from pathlib import Path
from urllib.parse import urlparse

from crewai import Agent, Crew, Process, Task

from .fact_checker import fact_check_with_one_correction
from .llm_config import get_llm

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ANALYSIS_FILE = PROJECT_ROOT / "critical_analysis.json"
INSIGHTS_FILE = PROJECT_ROOT / "insights.json"
MANIFEST_FILE = PROJECT_ROOT / "visualization_manifest.json"

CHART_DIR = (PROJECT_ROOT / "generated_charts").resolve()

REPORT_MD_FILE = PROJECT_ROOT / "final_research_report.md"
REPORT_HTML_FILE = PROJECT_ROOT / "final_research_report.html"

MAX_SOURCES = 8
MAX_SOURCE_CHARS = 1800
MAX_REPORT_INPUT_CHARS = 16000


def load_json(path: Path) -> dict:
    """Load and validate a JSON object."""
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(f"{path.name} must contain a JSON object.")

    return data


def build_source_registry(analysis_data: dict) -> dict[str, dict]:
    """Build a source registry for report citations."""
    registry = {}

    for source in analysis_data.get("sources", []):
        if not isinstance(source, dict):
            continue

        source_id = str(source.get("source_id", "")).strip()

        if source_id:
            registry[source_id] = source

    return registry


def prepare_sources(registry: dict[str, dict]) -> list[dict]:
    """Limit source content included in the LLM prompt."""
    prepared = []

    for source_id, source in list(registry.items())[:MAX_SOURCES]:
        prepared.append(
            {
                "source_id": source_id,
                "title": str(source.get("title", "")),
                "url": str(source.get("url", "")),
                "content": str(source.get("content", ""))[:MAX_SOURCE_CHARS],
            }
        )

    return prepared


def get_validated_charts(
    manifest: dict,
    source_registry: dict[str, dict],
) -> list[dict]:
    """
    Return only manifest-approved charts whose files exist
    inside generated_charts and whose metadata is valid.
    """
    if manifest.get("status") != "completed":
        return []

    raw_charts = manifest.get("charts", [])

    if not isinstance(raw_charts, list):
        return []

    validated = []
    seen_paths = set()

    for chart in raw_charts:
        if not isinstance(chart, dict):
            continue

        relative_path = str(chart.get("path", "")).strip()

        if not relative_path:
            continue

        # Reject absolute paths and path traversal.
        candidate = (PROJECT_ROOT / relative_path).resolve()

        try:
            candidate.relative_to(CHART_DIR)
        except ValueError:
            continue

        if candidate.suffix.lower() != ".html":
            continue

        if not candidate.is_file():
            continue

        if candidate in seen_paths:
            continue

        if chart.get("chart_type") not in {"bar", "line", "scatter"}:
            continue

        try:
            observation_count = int(chart.get("observation_count", 0))
        except (TypeError, ValueError):
            continue

        if observation_count < 2:
            continue

        units = chart.get("units", [])
        if not isinstance(units, list) or len(units) != 1:
            continue

        source_ids = chart.get("source_ids", [])
        if not isinstance(source_ids, list) or not source_ids:
            continue

        # Do not cite unknown source IDs from the manifest.
        if any(source_id not in source_registry for source_id in source_ids):
            continue

        # Prevent embedding a file that is not an HTML document.
        try:
            with candidate.open("r", encoding="utf-8") as file:
                beginning = file.read(2048).lower()
        except (OSError, UnicodeError):
            continue

        if "<html" not in beginning and "<!doctype html" not in beginning:
            continue

        item = dict(chart)
        item["path"] = candidate.relative_to(PROJECT_ROOT).as_posix()
        item["source_ids"] = sorted(set(source_ids))
        item["units"] = units
        item["observation_count"] = observation_count

        validated.append(item)
        seen_paths.add(candidate)

    return validated


def build_citation_guide(source_registry: dict[str, dict]) -> str:
    """Build a source list the report agent can cite."""
    entries = []

    for source_id, source in source_registry.items():
        title = str(source.get("title", "Untitled source")).strip()
        url = str(source.get("url", "")).strip()

        entries.append(
            {
                "source_id": source_id,
                "title": title,
                "url": url,
            }
        )

    return json.dumps(entries, ensure_ascii=False, indent=2)

def detect_insufficient_evidence(analysis: dict) -> bool:
    """Detect whether Critical Analysis explicitly reports insufficient evidence."""

    if analysis.get("evidence_sufficient") is False:
        return True

    assessment = str(
        analysis.get("overall_assessment", "")
    ).casefold()

    indicators = (
        "insufficient evidence",
        "no evidence",
        "lack of evidence",
        "cannot be verified",
        "could not be verified",
        "does not establish",
        "do not establish",
        "not enough evidence",
        "unsupported by the evidence",
        "unable to verify",
    )

    if any(term in assessment for term in indicators):
        return True

    uncertain_claims = analysis.get("uncertain_claims", [])
    research_gaps = analysis.get("research_gaps", [])

    limitation_text = " ".join(
        str(item)
        for item in uncertain_claims + research_gaps
    ).casefold()

    return any(
        term in limitation_text
        for term in (
            "no evidence",
            "insufficient evidence",
            "cannot be verified",
            "could not be verified",
            "not established",
        )
    )

def create_report_agent() -> Agent:
    return Agent(
        role="Research Report Builder",
        goal=(
            "Create a structured, evidence-based research report with "
            "traceable source citations and clear distinctions between "
            "findings, interpretations, and limitations."
        ),
        backstory=(
            "You are a careful research editor. You cite supplied source IDs, "
            "avoid unsupported claims, report uncertainty honestly, and never "
            "invent sources, statistics, quotations, or research results."
        ),
        llm=get_llm(),
        max_iter=1,
        allow_delegation=False,
        verbose=False,
    )


def generate_report_text(
    question: str,
    critical_analysis: dict,
    insights: dict,
    sources: list[dict],
    citation_guide: str,
) -> str:
    """Generate the narrative portion of the report."""
    agent = create_report_agent()

    task = Task(
        description="""
Write the narrative for a final research report.

Research question:
{question}

Critical analysis JSON:
{analysis_json}

Insight generation JSON:
{insights_json}

Retrieved source excerpts:
{sources_json}

Allowed source citation registry:
{citation_guide}

Requirements:
1. Use Markdown headings.
2. Include an executive summary, methodology, key findings,
   critical analysis, insights, limitations, and conclusion.
3. Cite factual claims using source IDs in this exact style: [S1].
4. Use only IDs present in the allowed source citation registry.
5. Do not invent URLs, quotations, numbers, sources, or statistics.
6. Distinguish source-supported findings from interpretations
   and hypotheses.
7. Discuss contradictions and missing evidence where supplied.
8. If evidence is insufficient, explicitly state the limitation.
9. Do not claim that a generated chart exists. Chart sections and
   embeds will be added separately by the report builder.
10. Do not fabricate chart data or treat a hypothesis as a fact.
11. Do not create a separate fabricated bibliography.
12. Return only the report body in Markdown.
13. CENTRAL CLAIM VERIFICATION: Determine whether the supplied
    evidence directly supports the central research question.
    Do not treat related background information as proof of the claim.

14. If Critical Analysis identifies insufficient evidence, state
    prominently that the central claim could not be established
    from the available evidence. Preserve that conclusion throughout
    the executive summary, findings, and conclusion.

15. Separate directly relevant findings from background context.
    Label unrelated developments "Background context" and explain
    that they do not establish the central claim.

16. Do not present an unsupported hypothesis as a conclusion.
    Do not invent explanations for why evidence is missing, and do
    not suggest secret or unofficial activity without direct support.

17. Absence of evidence does not prove that a claim is false.
    State only that the available sources do not establish it.

18. If the Insight Generation output indicates insufficient evidence
    or contains no hypotheses because they were suppressed, do not
    generate new hypotheses to replace them.

19. Do not turn a source's publication date into proof that an
    event happened on that date. Preserve uncertainty about dates
    when the supplied metadata is unverified.
""",
        expected_output=(
            "A structured Markdown research report with source-ID citations."
        ),
        agent=agent,
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
        tracing=False,
    )

    try:
        result = crew.kickoff(
            inputs={
                "question": question,
                "analysis_json": json.dumps(critical_analysis, ensure_ascii=False)[
                    :MAX_REPORT_INPUT_CHARS
                ],
                "insights_json": json.dumps(insights, ensure_ascii=False)[
                    :MAX_REPORT_INPUT_CHARS
                ],
                "sources_json": json.dumps(sources, ensure_ascii=False)[
                    :MAX_REPORT_INPUT_CHARS
                ],
                "citation_guide": citation_guide,
            }
        )
    except Exception as exc:
        raise RuntimeError(
            "Report generation failed because the LLM request failed. "
            "Check OpenRouter availability, model ID, and rate limits, "
            "then retry."
        ) from exc

    # report_text = str(result.raw).strip()

    # if not report_text:
    #     raise RuntimeError("The report agent returned an empty report.")

    # return report_text

    report_text = str(result.raw or "").strip()

    if not report_text:
        raise RuntimeError(
            "Report generation failed: the LLM returned an empty response. "
            "Check OpenRouter rate limits and retry."
        )

    if len(report_text) < 200:
        raise RuntimeError(
            "Report generation failed: the response is unexpectedly short. "
            "Check the OpenRouter response and retry."
        )

    return report_text


def build_bibliography(source_registry: dict[str, dict]) -> str:
    """Create a bibliography from the retrieved source registry."""
    lines = ["## References", ""]

    for source_id, source in source_registry.items():
        title = str(source.get("title", "Untitled source")).strip()
        url = str(source.get("url", "")).strip()

        if urlparse(url).scheme not in {"http", "https"}:
            url = ""

        if url:
            lines.append(f"- [{source_id}] [{title}]({url})")
        else:
            lines.append(f"- [{source_id}] {title}")

    return "\n".join(lines)


# def build_chart_section(charts: list[dict]) -> str:
#     """Add links and HTML embeds only for approved charts."""
#     if not charts:
#         return (
#             "## Validated Data Visualizations\n\n"
#             "No chart passed the visualization manifest and file checks. "
#             "No charts have been embedded in this report.\n"
#         )

#     sections = ["## Validated Data Visualizations", ""]

#     for index, chart in enumerate(charts, start=1):
#         title = str(chart.get("title", f"Chart {index}"))
#         relative_path = chart["path"]
#         chart_filename = Path(relative_path).name
#         chart_url = relative_path.replace("\\", "/")

#         source_ids = ", ".join(chart["source_ids"])
#         units = ", ".join(str(unit) for unit in chart["units"])

#         sections.extend(
#             [
#                 f"### {index}. {title}",
#                 "",
#                 (
#                     f"- Observations: {chart['observation_count']}\n"
#                     f"- Unit: {units}\n"
#                     f"- Source IDs: {source_ids}\n"
#                     f"- [Open chart: {chart_filename}]({chart_url})"
#                 ),
#                 "",
#                 (
#                     f'<iframe src="{html.escape(chart_url, quote=True)}" '
#                     'width="100%" height="560" '
#                     'style="border:0;" loading="lazy" '
#                     f'title="{html.escape(title, quote=True)}"></iframe>'
#                 ),
#                 "",
#             ]
#         )

#     return "\n".join(sections)



def build_chart_section(charts: list[dict]) -> str:
    """Create readable Markdown metadata for validated charts."""

    if not charts:
        return (
            "## Validated Data Visualizations\n\n"
            "No charts passed validation for this research run.\n"
        )

    sections = ["## Validated Data Visualizations", ""]

    for index, chart in enumerate(charts, start=1):
        title = str(chart.get("title", f"Chart {index}"))
        units = ", ".join(str(unit) for unit in chart.get("units", []))
        source_ids = ", ".join(chart.get("source_ids", []))

        sections.extend(
            [
                f"### {index}. {title}",
                "",
                f"- Observations: {chart.get('observation_count', 0)}",
                f"- Units: {units or 'Not specified'}",
                f"- Source IDs: {source_ids or 'Not specified'}",
                "- View the interactive chart in the Charts tab of the app.",
                "",
            ]
        )

    return "\n".join(sections)


def build_html_report(markdown_report: str) -> str:
    """
    Create a simple HTML report.

    The Markdown narrative is escaped for safe display; chart embeds
    are inserted separately from the trusted manifest-approved list.
    """
    escaped_report = html.escape(markdown_report)

    # Keep a readable HTML version without requiring another package.
    paragraphs = []
    for block in escaped_report.split("\n\n"):
        block = block.strip()
        if not block:
            continue

        if block.startswith("### "):
            paragraphs.append(f"<h3>{block[4:]}</h3>")
        elif block.startswith("## "):
            paragraphs.append(f"<h2>{block[3:]}</h2>")
        elif block.startswith("# "):
            paragraphs.append(f"<h1>{block[2:]}</h1>")
        else:
            paragraphs.append(f"<p>{block.replace(chr(10), '<br>')}</p>")

    body = "\n".join(paragraphs)

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Research Report</title>
<style>
body {{ font-family: Arial, sans-serif; max-width: 1000px;
       margin: 40px auto; padding: 0 20px; line-height: 1.65; }}
iframe {{ max-width: 100%; }}
</style>
</head>
<body>
{body}
</body>
</html>
"""


def run_report_builder() -> dict:
    analysis_data = load_json(ANALYSIS_FILE)
    insights_data = load_json(INSIGHTS_FILE)
    manifest = load_json(MANIFEST_FILE)

    question = str(analysis_data.get("research_question", "")).strip()
    if not question:
        raise ValueError(
            "research_question is missing from critical_analysis.json."
        )

    source_registry = build_source_registry(analysis_data)
    if not source_registry:
        raise ValueError("No retrieved sources were found for report citations.")

    sources = prepare_sources(source_registry)
    citation_guide = build_citation_guide(source_registry)

    # Only manifest-approved charts can enter the final report.
    validated_charts = get_validated_charts(manifest, source_registry)

    narrative = generate_report_text(
        question=question,
        critical_analysis=analysis_data.get("critical_analysis", {}),
        insights=insights_data.get("insight_generation", {}),
        sources=sources,
        citation_guide=citation_guide,
    )

    # Verify factual claims and allow at most one correction attempt.
    narrative, fact_check_audit = fact_check_with_one_correction(
        report_text=narrative,
        research_question=question,
        source_registry=source_registry,
    )

    print("Fact-check status:", fact_check_audit["report_status"])
    print("Correction attempts:", fact_check_audit["correction_attempts"])

    if fact_check_audit["report_status"] == "needs_review":
        narrative += (
            "\n\n## Fact-Checking Notice\n\n"
            "The final verification identified claims that still need "
            "manual review. See `fact_check_report.json` for details. "
            "Do not treat unresolved claims as verified facts.\n"
        )

        # Enforce the evidence limitation in the final report.
    critical_analysis = analysis_data.get("critical_analysis", {})
    insufficient_evidence = detect_insufficient_evidence(
        critical_analysis
    )

    if insufficient_evidence:
        evidence_notice = (
            "## Evidence Sufficiency Notice\n\n"
            "**Status: INSUFFICIENT EVIDENCE.** The available sources "
            "do not establish the central claim in the research question. "
            "Related findings, if included, are background context only "
            "and must not be interpreted as proof of that claim. "
            "This does not prove the claim is false; it remains unverified "
            "based on the evidence available in this research run.\n"
        )

        if "## Evidence Sufficiency Notice" not in narrative:
            narrative = narrative.rstrip() + "\n\n" + evidence_notice

    chart_section = build_chart_section(validated_charts)
    bibliography = build_bibliography(source_registry)

    final_markdown = "\n\n".join(
        [
            f"# Research Report\n\n**Research question:** {question}",
            narrative,
            chart_section,
            bibliography,
        ]
    )

    REPORT_MD_FILE.write_text(final_markdown, encoding="utf-8")

    # Save an HTML report and insert only manifest-approved chart embeds.
    html_document = build_html_report(narrative + "\n\n" + bibliography)
    chart_html = build_chart_section(validated_charts)
    html_document = html_document.replace(
        "</body>",
        f"{chart_html}\n</body>",
    )
    REPORT_HTML_FILE.write_text(html_document, encoding="utf-8")

    result = {
        "status": "completed",
        "evidence_sufficient": not insufficient_evidence,
        "report_markdown": str(REPORT_MD_FILE.relative_to(PROJECT_ROOT)),
        "report_html": str(REPORT_HTML_FILE.relative_to(PROJECT_ROOT)),
        "validated_charts_embedded": len(validated_charts),
        "embedded_chart_paths": [chart["path"] for chart in validated_charts],
        "message": (
            "Report generated with manifest-approved charts."
            if validated_charts
            else "Report generated without charts because no charts "
            "passed manifest and file validation."
        ),
        "fact_check_status": fact_check_audit["report_status"],
        "fact_check_verdict": fact_check_audit["final_verdict"],
        "correction_attempts": fact_check_audit["correction_attempts"],
        "fact_check_file": "fact_check_report.json",
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    run_report_builder()
