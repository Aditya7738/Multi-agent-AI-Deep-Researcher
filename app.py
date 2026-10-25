
import json
import subprocess
import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from src.multi_agent_ai_deep_researcher.query_planner import plan_research
from src.multi_agent_ai_deep_researcher.contextual_retriever import (
    retrieve_sources,
    search_budget_status,
)
from src.multi_agent_ai_deep_researcher.critical_analysis import (
    analyze_evidence,
)
from src.multi_agent_ai_deep_researcher.data_visualization import (
    run_visualization,
)
from src.multi_agent_ai_deep_researcher.report_builder import (
    run_report_builder,
)
from src.multi_agent_ai_deep_researcher.research_with_web import (
    is_valid_research_question,
)


PROJECT_ROOT = Path(__file__).resolve().parent
PACKAGE_MODULE = "src.multi_agent_ai_deep_researcher"

ANALYSIS_FILE = PROJECT_ROOT / "critical_analysis.json"
INSIGHTS_FILE = PROJECT_ROOT / "insights.json"
CHART_DATA_FILE = PROJECT_ROOT / "chart_data.json"
MANIFEST_FILE = PROJECT_ROOT / "visualization_manifest.json"
FACT_CHECK_FILE = PROJECT_ROOT / "fact_check_report.json"
REPORT_MD_FILE = PROJECT_ROOT / "final_research_report.md"
CHART_DIR = PROJECT_ROOT / "generated_charts"


def display_critical_analysis(analysis: dict) -> None:
    """Render critical analysis as readable research findings."""

    if not isinstance(analysis, dict):
        st.info("Critical analysis is not available.")
        return

    overall = analysis.get("overall_assessment")

    if overall:
        st.subheader("Overall Assessment")
        st.write(overall)

    findings = analysis.get("findings", [])

    if findings:
        st.subheader("Key Findings")

        for index, finding in enumerate(findings, start=1):
            if not isinstance(finding, dict):
                continue

            claim = finding.get("claim", "Finding")
            with st.container(border=True):
                st.markdown(f"**Finding {index}: {claim}**")

                if finding.get("evidence_summary"):
                    st.write(finding["evidence_summary"])

                col1, col2 = st.columns(2)

                with col1:
                    confidence = finding.get("confidence")
                    if confidence:
                        st.caption(f"Confidence: {confidence}")

                with col2:
                    source_ids = finding.get("source_ids", [])
                    if source_ids:
                        st.caption(
                            "Supporting sources: "
                            + ", ".join(map(str, source_ids))
                        )

                if finding.get("limitations"):
                    st.caption(
                        f"Limitations: {finding['limitations']}"
                    )

    contradictions = analysis.get("contradictions", [])

    if contradictions:
        st.subheader("Contradictions in the Evidence")

        for item in contradictions:
            if not isinstance(item, dict):
                continue

            with st.expander(
                str(item.get("claim", "Evidence conflict"))
            ):
                st.write(
                    item.get(
                        "explanation",
                        item.get("description", "No explanation provided."),
                    )
                )

                source_ids = item.get("source_ids", [])
                if source_ids:
                    st.caption(
                        "Sources: " + ", ".join(map(str, source_ids))
                    )

    uncertain = (
        analysis.get("uncertain_claims")
        or analysis.get("uncertain_findings")
        or []
    )

    if uncertain:
        st.subheader("Uncertain Claims")

        for item in uncertain:
            if isinstance(item, dict):
                st.markdown(
                    f"- **{item.get('claim', 'Uncertain claim')}**"
                )
                explanation = item.get("explanation")
                if explanation:
                    st.write(explanation)
            else:
                st.markdown(f"- {item}")

    gaps = analysis.get("research_gaps", [])

    if gaps:
        st.subheader("Research Gaps")
        for gap in gaps:
            if isinstance(gap, dict):
                st.markdown(
                    f"- {gap.get('description', gap.get('gap', str(gap)))}"
                )
            else:
                st.markdown(f"- {gap}")


def display_insights(insight_data: dict) -> None:
    """Display insights and hypotheses as readable cards."""

    if not isinstance(insight_data, dict):
        st.info("Insight data is not available.")
        return

    insights = insight_data.get("insights", [])

    if insights:
        st.subheader("Generated Insights")

        for index, item in enumerate(insights, start=1):
            if not isinstance(item, dict):
                continue

            title = item.get("title") or item.get(
                "insight", f"Insight {index}"
            )

            with st.container(border=True):
                st.markdown(f"**{index}. {title}**")

                description = item.get("description") or item.get("reasoning")
                if description:
                    st.write(description)

                source_ids = item.get("source_ids", [])
                if source_ids:
                    st.caption(
                        "Evidence sources: "
                        + ", ".join(map(str, source_ids))
                    )

                confidence = item.get("confidence")
                if confidence:
                    st.caption(f"Confidence: {confidence}")

    hypotheses = insight_data.get("hypotheses", [])

    if hypotheses:
        st.subheader("Hypotheses to Investigate")

        for index, item in enumerate(hypotheses, start=1):
            if not isinstance(item, dict):
                continue

            title = item.get("hypothesis") or item.get(
                "statement", f"Hypothesis {index}"
            )

            with st.container(border=True):
                st.markdown(f"**Hypothesis {index}**")
                st.write(title)

                rationale = item.get("rationale") or item.get("reasoning")
                if rationale:
                    st.write(rationale)

                source_ids = item.get("source_ids", [])
                if source_ids:
                    st.caption(
                        "Related sources: "
                        + ", ".join(map(str, source_ids))
                    )

    if not insights and not hypotheses:
        st.info(
            "No structured insights or hypotheses were found. "
            "Expand the technical details below to inspect the saved output."
        )

        with st.expander("Technical insight output"):
            st.json(insight_data)


st.set_page_config(
    page_title="Multi-Agent AI Deep Researcher",
    page_icon="🔎",
    layout="wide",
)


def read_json(path: Path) -> dict:
    """Read a JSON file safely and return its contents."""
    if not path.exists():
        raise FileNotFoundError(f"Required output file not found: {path.name}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(f"{path.name} must contain a JSON object.")

    return data


def write_json(path: Path, data: dict) -> None:
    """Write a JSON object to the project root."""
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def display_fact_check(audit: dict | None) -> None:
    """Present the fact-check audit as a readable summary."""

    st.subheader("Fact-Checking Results")

    if not audit:
        st.info("No fact-check audit is available for this run.")
        return

    status = audit.get("report_status", "unknown")
    verdict = audit.get("final_verdict", "unknown")
    initial_verdict = audit.get("initial_verdict", "unknown")
    attempts = audit.get("correction_attempts", 0)
    issues = audit.get("final_issues", [])

    if status == "fact_checked" and verdict == "pass":
        st.success(
            "No material factual issues were identified in the final check."
        )
    else:
        st.warning(
            "Some claims need further review. Read the issues below "
            "before relying on the report."
        )

    col1, col2, col3 = st.columns(3)

    col1.metric("Initial check", str(initial_verdict).replace("_", " ").title())
    col2.metric("Final check", str(verdict).replace("_", " ").title())
    col3.metric("Correction attempts", attempts)

    summary = audit.get("final_summary")
    if summary:
        st.write(summary)

    st.markdown("**Remaining issues**")

    if not issues:
        st.write("No issues were returned by the final fact-check.")
        return

    for index, issue in enumerate(issues, start=1):
        if not isinstance(issue, dict):
            continue

        claim = issue.get("claim", f"Issue {index}")
        status_text = str(issue.get("status", "uncertain")).replace(
            "_", " "
        ).title()

        with st.container(border=True):
            st.markdown(f"**{index}. {claim}**")
            st.caption(f"Issue type: {status_text}")

            if issue.get("explanation"):
                st.write(issue["explanation"])

            source_ids = issue.get("source_ids", [])
            if source_ids:
                st.caption(
                    "Related sources: " + ", ".join(map(str, source_ids))
                )

            quote = issue.get("evidence_quote")
            if quote:
                st.markdown("**Evidence excerpt**")
                st.markdown(f"> {quote}")

    # with st.expander("Technical audit details"):
    #     st.json(audit)

    with st.expander("Detailed Fact-Check Audit"):
        st.markdown("### Verification Summary")

        st.write(f"**Research question:** {audit.get('research_question', 'Not available')}")
        st.write(f"**Initial verdict:** {str(audit.get('initial_verdict', 'unknown')).replace('_', ' ').title()}")
        st.write(f"**Final verdict:** {str(audit.get('final_verdict', 'unknown')).replace('_', ' ').title()}")
        st.write(
            f"**Correction attempts:** {audit.get('correction_attempts', 0)} "
            f"of {audit.get('max_correction_attempts', 1)}"
        )

        if audit.get("final_summary"):
            st.markdown("### Summary")
            st.write(audit["final_summary"])

        issues = audit.get("final_issues", [])

        st.markdown("### Remaining Issues")

        if not issues:
            st.success("No issues were returned by the final verification.")
        else:
            for index, issue in enumerate(issues, start=1):
                with st.container(border=True):
                    st.markdown(f"**{index}. {issue.get('claim', 'Unspecified claim')}**")
                    st.write(
                        f"**Issue type:** "
                        f"{str(issue.get('status', 'uncertain')).replace('_', ' ').title()}"
                    )
                    st.write(issue.get("explanation", "No explanation provided."))

                    source_ids = issue.get("source_ids", [])
                    if source_ids:
                        st.caption("Sources: " + ", ".join(source_ids))

                    if issue.get("evidence_quote"):
                        st.markdown("**Evidence excerpt**")
                        st.markdown(f"> {issue['evidence_quote']}")

        with st.expander("Developer Debug Data (JSON)"):
            st.json(audit)



def run_module(module_name: str) -> None:
    """Run an existing package module and surface errors in Streamlit."""
    completed = subprocess.run(
        [sys.executable, "-m", f"{PACKAGE_MODULE}.{module_name}"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    if completed.returncode != 0:
        details = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(
            f"Stage '{module_name}' failed.\n{details[-6000:]}"
        )

    # if completed.stdout.strip():
    #     st.code(completed.stdout[-4000:], language="text")


def run_research_workflow(question: str) -> dict:
    """Run the complete research pipeline from planning to final report."""

    is_valid, message = is_valid_research_question(question)

    if not is_valid:
        raise ValueError(message)
    
    progress = st.progress(0, text="Preparing research workflow...")
    stage_status = st.status("Running research agents...", expanded=True)

    try:    
        # 1. Query Planner Agent
        stage_status.write("1/8 — Planning research queries")
        queries = plan_research(question)

        if not queries:
            raise RuntimeError(
                "The Query Planner returned no usable research queries."
            )

        st.session_state["research_queries"] = queries
        progress.progress(10, text="Research queries planned")

        # 2. Cost Optimizer + Contextual Retriever Agent
        # retrieve_sources() calls optimize_queries() internally.
        stage_status.write("2/8 — Optimizing cost and retrieving sources")
        sources = retrieve_sources(queries)

        if not sources:
            raise RuntimeError(
                "No sources were retrieved. Check the Tavily key, search "
                "budget, and search results before continuing."
            )

        progress.progress(30, text="Sources retrieved")

        # 3. Critical Analysis Agent
        stage_status.write("3/8 — Analyzing retrieved evidence")
        critical_analysis = analyze_evidence(question, sources)

        write_json(
            ANALYSIS_FILE,
            {
                "research_question": question,
                "planned_queries": queries,
                "sources": sources,
                "critical_analysis": critical_analysis,
            },
        )

        progress.progress(45, text="Critical analysis completed")

        # 4. Insight Generator Agent
        stage_status.write("4/8 — Generating insights")
        run_module("insight_generator")

        if not INSIGHTS_FILE.exists():
            raise RuntimeError(
                "Insight Generator finished without creating insights.json."
            )

        progress.progress(58, text="Insights generated")

        # 5. Numerical Evidence Extractor
        stage_status.write("5/8 — Extracting source-supported numerical evidence")
        run_module("numerical_evidence_extractor")

        if not CHART_DATA_FILE.exists():
            raise RuntimeError(
                "Numerical extraction did not create chart_data.json."
            )

        progress.progress(68, text="Numerical evidence extracted")

        # 6. Data Visualization
        stage_status.write("6/8 — Validating evidence and generating charts")
        visualization_result = run_visualization()
        write_json(MANIFEST_FILE, visualization_result)

        progress.progress(77, text="Visualization stage completed")

        # 7. Report Builder + Fact-Checker
        # Your report_builder.py calls fact_check_with_one_correction().
        stage_status.write("7/8 — Building and fact-checking the report")
        report_result = run_report_builder()

        if not REPORT_MD_FILE.exists():
            raise RuntimeError(
                "Report Builder did not create final_research_report.md."
            )

        progress.progress(95, text="Report generated and fact-checked")

        # 8. Load final artifacts for the UI
        stage_status.write("8/8 — Loading report, sources, and charts")

        output = {
            "question": question,
            "queries": queries,
            "sources": sources,
            "critical_analysis": critical_analysis,
            "insights": read_json(INSIGHTS_FILE),
            "chart_data": read_json(CHART_DATA_FILE),
            "visualization": read_json(MANIFEST_FILE),
            "report": REPORT_MD_FILE.read_text(encoding="utf-8"),
            "report_result": report_result,
            "fact_check": (
                read_json(FACT_CHECK_FILE)
                if FACT_CHECK_FILE.exists()
                else None
            ),
            "budget": search_budget_status(),
        }

        progress.progress(100, text="Research workflow completed")
        stage_status.update(
            label="Research workflow completed",
            state="complete",
            expanded=False,
        )

        return output

    except Exception as exc:
        stage_status.update(
            label="Research workflow failed",
            state="error",
            expanded=True,
        )
        raise RuntimeError(str(exc)) from exc


def display_sources(sources: list[dict]) -> None:
    """Display retrieved sources and their metadata."""
    if not sources:
        st.info("No sources available.")
        return

    for source in sources:
        source_id = source.get("source_id", "Unknown")
        title = source.get("title", "Untitled source")
        url = source.get("url", "")
        content = source.get("content", "")
        score = source.get("relevance_score")

        with st.expander(f"[{source_id}] {title}"):
            if url:
                st.markdown(f"**Source URL:** [{url}]({url})")

            if score is not None:
                st.write(f"Relevance score: {score}")

            matched_queries = source.get("matched_queries", [])
            if matched_queries:
                st.write("Matched queries:", ", ".join(matched_queries))

            st.write(content or "No source excerpt was returned.")



def display_charts(manifest: dict) -> None:
    """Render validated chart files directly in Streamlit."""

    charts = manifest.get("charts", [])

    if manifest.get("status") != "completed" or not charts:
        st.info(
            "No validated charts were generated for this research question."
        )
        return

    chart_root = CHART_DIR.resolve()

    for index, chart in enumerate(charts, start=1):
        relative_path = str(chart.get("path", "")).strip()

        if not relative_path:
            continue

        chart_path = (PROJECT_ROOT / relative_path).resolve()

        try:
            chart_path.relative_to(chart_root)
        except ValueError:
            st.warning("Skipped a chart with an invalid file path.")
            continue

        if not chart_path.is_file() or chart_path.suffix.lower() != ".html":
            st.warning(f"Chart file is missing: {chart_path.name}")
            continue

        st.subheader(chart.get("title", f"Chart {index}"))

        st.caption(
            f"Observations: {chart.get('observation_count', 0)} | "
            f"Units: {', '.join(map(str, chart.get('units', [])))} | "
            f"Sources: {', '.join(chart.get('source_ids', []))}"
        )

        chart_html = chart_path.read_text(encoding="utf-8")

        components.html(
            chart_html,
            height=650,
            scrolling=True,
        )

        st.download_button(
            label="Download interactive chart",
            data=chart_path.read_bytes(),
            file_name=chart_path.name,
            mime="text/html",
            key=f"download_chart_{index}",
        )


st.title("Multi-Agent AI Deep Researcher")
st.write(
    "Research a question using coordinated AI agents, retrieved web evidence, "
    "source-supported numerical charts, and a fact-checked final report."
)

with st.sidebar:
    st.header("Research Settings")
    st.write("Search limits are controlled by your `.env` configuration.")

    try:
        budget = search_budget_status()
        st.metric("Estimated Tavily credits remaining", budget["credits_remaining"])
        st.caption(
            f"Estimated usage: {budget['credits_used']} / "
            f"{budget['monthly_limit']} for {budget['month']}"
        )
    except Exception as exc:
        st.warning(f"Could not load search budget: {exc}")

    st.caption(
        "The local budget is an application estimate, not a live Tavily "
        "account balance."
    )

question = st.text_area(
    "Enter your research question",
    placeholder=(
        "How is generative AI changing software engineering productivity, "
        "code quality, and employment?"
    ),
    height=110,
)

run_clicked = st.button(
    "Start Research",
    type="primary",
    disabled=not question.strip(),
)

# if run_clicked:
#     try:
#         with st.spinner("Running your multi-agent research workflow..."):



#             result = run_research_workflow(question.strip())

#         st.session_state["research_result"] = result
#         st.success("Research workflow completed.")

#     except Exception as exc:
#         st.error("The research workflow did not complete.")
#         st.exception(exc)


if run_clicked:
    st.session_state.pop("research_result", None)

    is_valid, validation_message = is_valid_research_question(
        question.strip()
    )

    if not is_valid:
        st.error(
            "The research workflow did not complete.\n\n"
            f"{validation_message}"
        )
    else:
        try:
            with st.spinner(
                "Running your multi-agent research workflow..."
            ):
                result = run_research_workflow(question.strip())

            st.session_state["research_result"] = result
            st.success("Research workflow completed.")

        except ValueError as exc:
            st.error(
                "The research workflow did not complete.\n\n"
                f"{exc}"
            )

        except Exception:
            logging.exception("Research workflow failed")
            st.error(
                "The research workflow did not complete. "
                "Please try again later."
            )



result = st.session_state.get("research_result")

if result:
    report_result = result["report_result"]
    fact_check = result["fact_check"]
    budget = result["budget"]

    st.divider()
    st.header("Research Results")

    col1, col2, col3 = st.columns(3)

    col1.metric("Retrieved sources", len(result["sources"]))
    col2.metric(
        "Validated charts",
        len(result["visualization"].get("charts", [])),
    )
    col3.metric(
        "Correction attempts",
        report_result.get("correction_attempts", 0),
    )

    st.caption(
        f"Estimated Tavily credits remaining: {budget['credits_remaining']}"
    )

    if fact_check:
        display_fact_check(fact_check)

    tabs = st.tabs(
        [
            "Final Report",
            "Charts",
            "Sources",
            "Research Plan",
            "Evidence & Insights",
        ]
    )

    with tabs[0]:
        st.markdown(result["report"])

        st.download_button(
            "Download Markdown report",
            data=result["report"],
            file_name="final_research_report.md",
            mime="text/markdown",
        )

        html_report_path = PROJECT_ROOT / "final_research_report.html"
        if html_report_path.exists():
            st.download_button(
                "Download HTML report",
                data=html_report_path.read_bytes(),
                file_name="final_research_report.html",
                mime="text/html",
            )

    with tabs[1]:
        display_charts(result["visualization"])

    with tabs[2]:
        display_sources(result["sources"])

    with tabs[3]:
        st.subheader("Planned research queries")
        for index, query in enumerate(result["queries"], start=1):
            st.write(f"{index}. {query}")

        st.subheader("Cost Optimizer")
        st.write(
            "The Contextual Retriever applies query deduplication, "
            "search-per-run limits, and monthly budget limits before "
            "executing Tavily searches."
        )

    with tabs[4]:
        analysis = result["critical_analysis"]

        display_critical_analysis(analysis)

        st.divider()

        insight_data = result["insights"].get(
            "insight_generation",
            result["insights"],
        )
        display_insights(insight_data)

        st.divider()
        st.subheader("Numerical Evidence")

        observations = result["chart_data"].get("observations", [])
        manifest = result["visualization"]

        col1, col2 = st.columns(2)

        col1.metric(
            "Accepted observations",
            manifest.get("validated_observations", len(observations)),
        )
        col2.metric(
            "Rejected observations",
            manifest.get("rejected_observations", 0),
        )

        if observations:
            with st.expander("Inspect extracted numerical observations"):
                st.dataframe(observations, use_container_width=True)
