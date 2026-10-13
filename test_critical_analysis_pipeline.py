import json

from src.multi_agent_ai_deep_researcher.query_planner import (
    plan_research,
)
from src.multi_agent_ai_deep_researcher.contextual_retriever import (
    retrieve_sources,
)
from src.multi_agent_ai_deep_researcher.critical_analysis import (
    analyze_evidence,
)


def main() -> None:
    question = input(
        "Enter your research question: "
    ).strip()

    if not question:
        print("Please enter a research question.")
        return

    try:
        # Stage 1: Plan the research.
        print("\n[1/3] Planning research...")
        queries = plan_research(question)

        print("\nPlanned queries:")

        for index, query in enumerate(queries, start=1):
            print(f"{index}. {query}")

        # Stage 2: Retrieve evidence within the configured budget.
        print("\n[2/3] Retrieving sources...")
        sources = retrieve_sources(queries)

        if not sources:
            print("No sources were retrieved. Analysis stopped.")
            return

        print(f"Retrieved {len(sources)} unique sources.")

        # Stage 3: Analyze the evidence.
        print("\n[3/3] Critically analyzing evidence...")

        analysis = analyze_evidence(
            research_question=question,
            sources=sources,
        )

        # Save the structured output for the next agent.
        output_file = "critical_analysis.json"

        with open(
            output_file,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                {
                    "research_question": question,
                    "planned_queries": queries,
                    "sources": sources,
                    "critical_analysis": analysis,
                },
                file,
                indent=2,
                ensure_ascii=False,
            )

        print("\nCRITICAL ANALYSIS")
        print("=" * 60)

        print("\nOverall assessment:")
        print(analysis["overall_assessment"])

        print("\nEvidence-supported findings:")

        for index, finding in enumerate(
            analysis["findings"],
            start=1,
        ):
            print(f"\n{index}. {finding['claim']}")
            print(f"Sources: {finding['source_ids']}")
            print(f"Confidence: {finding['confidence']}")
            print(f"Evidence: {finding['evidence_summary']}")

            if finding["limitations"]:
                print(f"Limitations: {finding['limitations']}")

        print("\nContradictions:")

        if not analysis["contradictions"]:
            print("No explicit contradictions identified.")

        for contradiction in analysis["contradictions"]:
            print(f"\nTopic: {contradiction['topic']}")
            print(
                f"Conflicting claims: "
                f"{contradiction['conflicting_claims']}"
            )
            print(f"Sources: {contradiction['source_ids']}")
            print(
                f"Possible explanation: "
                f"{contradiction['possible_explanation']}"
            )

        print("\nUncertain claims:")

        for claim in analysis["uncertain_claims"]:
            print(f"- {claim}")

        print("\nResearch gaps:")

        for gap in analysis["research_gaps"]:
            print(f"- {gap}")

        print(f"\nFull results saved to {output_file}")

    except (ValueError, RuntimeError) as error:
        print(f"\nResearch pipeline failed: {error}")

    except Exception as error:
        print(f"\nUnexpected error: {error}")


if __name__ == "__main__":
    main()