from crewai import Task

from src.multi_agent_ai_deep_researcher.agents import (
    create_research_agent,
)


def create_research_task(research_question: str) -> Task:
    """Create a task to analyze a research question."""

    if not research_question or not research_question.strip():
        raise ValueError("Research question cannot be empty.")

    agent = create_research_agent()

    return Task(
        description=(
            "Analyze the following research question:\n\n"
            "{research_question}\n\n"
            "Produce a structured research summary containing:\n"
            "1. Executive summary\n"
            "2. Key concepts and findings\n"
            "3. Potential benefits\n"
            "4. Challenges and limitations\n"
            "5. Areas requiring further investigation\n\n"
            "Do not invent statistics, sources, or citations. "
            "Clearly identify uncertainty and information "
            "that requires external verification."
        ).format(research_question=research_question),
        expected_output=(
            "A structured research summary with clear headings, "
            "key findings, limitations, and areas for further "
            "investigation."
        ),
        agent=agent,
    )