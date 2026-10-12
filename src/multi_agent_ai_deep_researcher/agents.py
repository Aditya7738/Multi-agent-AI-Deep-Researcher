from crewai import Agent

from src.multi_agent_ai_deep_researcher.llm_config import get_llm


def create_research_agent() -> Agent:
    """Create the first research agent."""

    return Agent(
        role="Research Analyst",
        goal=(
            "Analyze the user's research question and produce "
            "a clear, accurate, structured summary."
        ),
        backstory=(
            "You are an experienced research analyst specializing "
            "in technology, science, and evidence-based reasoning. "
            "You distinguish established facts from uncertainty "
            "and avoid inventing statistics or citations."
        ),
        llm=get_llm(),
        verbose=True,
        allow_delegation=False,
        max_iter=1,
    )