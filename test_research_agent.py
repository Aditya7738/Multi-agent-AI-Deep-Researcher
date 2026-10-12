from crewai import Crew, Process

from src.multi_agent_ai_deep_researcher.tasks import (
    create_research_task,
)


def main() -> None:
    research_question = input("Enter your research question: ").strip()

    if not research_question:
        print("Please enter a research question.")
        return

    task = create_research_task(research_question)

    crew = Crew(
        agents=[task.agent],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
        tracing=False,
    )

    print("\nResearching your question...\n")

    result = crew.kickoff()

    print("\n" + "=" * 60)
    print("RESEARCH SUMMARY")
    print("=" * 60)
    print(result.raw)


if __name__ == "__main__":
    main()
