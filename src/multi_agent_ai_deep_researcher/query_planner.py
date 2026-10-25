
import re

from crewai import Agent, Crew, Process, Task
from pydantic import BaseModel, Field

from .llm_config import get_llm

# from .research_with_web import is_valid_research_question
from .research_with_web import (
    is_valid_research_question,
    validate_research_scope,
)

MAX_SUBQUERIES = 4


class ResearchPlan(BaseModel):
    """Structured output produced by the Query Planner."""

    subqueries: list[str] = Field(
        description=(
            "A list of 2 to 4 distinct, focused research "
            "questions that collectively investigate the "
            "user's original research question."
        ),
        min_length=2,
        max_length=MAX_SUBQUERIES,
    )


def create_query_planner_agent() -> Agent:
    """Create the Query Planner Agent using OpenRouter."""

    return Agent(
        role="Research Query Planner",
        goal=(
            "Decompose complex research questions into a small "
            "set of focused, complementary search queries that "
            "cover the important dimensions of the topic."
        ),
        backstory=(
            "You are an experienced research strategist. You "
            "identify the main question, break it into distinct "
            "investigation areas, and formulate precise questions "
            "that can be answered using web sources. You avoid "
            "duplicate queries, irrelevant topics, and unnecessary "
            "searches."
        ),
        llm=get_llm(),
        allow_delegation=False,
        max_iter=1,
        verbose=False,
    )


def plan_research(question: str) -> list[str]:
    """
    Generate up to four focused research queries.

    Invalid input is rejected locally before any LLM request.
    """

    # is_valid, error_message = is_valid_research_question(question)

    # if not is_valid:
    #     raise ValueError(error_message)

    # question = question.strip()

    is_valid, error_message = is_valid_research_question(question)

    if not is_valid:
        raise ValueError(error_message)

    is_specific, scope_message = validate_research_scope(question)

    if not is_specific:
        raise ValueError(scope_message)

    question = question.strip()

    planner = create_query_planner_agent()

    task = Task(
        description=(
            "Create a focused research plan for the question below.\n\n"
            f"Original research question:\n{question}\n\n"
            "Instructions:\n"
            "1. Generate between 2 and 4 search queries.\n"
            "2. Cover distinct aspects of the original question.\n"
            "3. Prioritize factual questions answerable through web research.\n"
            "4. Include relevant comparisons, causes, effects, or risks "
            "when appropriate.\n"
            "5. Avoid duplicate or overlapping queries.\n"
            "6. Keep each query specific and concise.\n"
            "7. Do not answer the research question yourself.\n"
            "8. Do not invent facts or assume the conclusion.\n"
            "Return only the structured ResearchPlan output."
        ),
        expected_output=(
            "A ResearchPlan containing between 2 and 4 distinct "
            "research subqueries."
        ),
        output_pydantic=ResearchPlan,
        agent=planner,
    )

    crew = Crew(
        agents=[planner],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
        tracing=False,
    )

    result = crew.kickoff()

    if result.pydantic is None:
        raise RuntimeError(
            "The Query Planner did not return a valid research plan."
        )

    # Normalize and remove duplicate queries.
    queries = []
    seen = set()

    for query in result.pydantic.subqueries:
        query = query.strip()
        normalized = re.sub(r"\s+", " ", query).casefold()

        if query and normalized not in seen:
            queries.append(query)
            seen.add(normalized)

    if not queries:
        raise RuntimeError("The Query Planner returned no usable queries.")

    return queries[:MAX_SUBQUERIES]


def main() -> None:
    question = input(
        "Enter your research question: "
    ).strip()

    try:
        queries = plan_research(question)

        print("\nQUERY PLAN")
        print("=" * 50)

        for index, query in enumerate(queries, start=1):
            print(f"{index}. {query}")

        print(f"\nTotal planned queries: {len(queries)}")

    except ValueError as error:
        print(f"\nInvalid research question: {error}")

    except Exception as error:
        print(f"\nQuery planning failed: {error}")


if __name__ == "__main__":
    main()
