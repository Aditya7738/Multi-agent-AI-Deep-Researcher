import os

import re
from collections import Counter

from crewai import Agent, Crew, Process, Task
from dotenv import load_dotenv
from tavily import TavilyClient

from src.multi_agent_ai_deep_researcher.llm_config import get_llm

load_dotenv()


def search_web(question: str) -> list[dict]:
    """Perform one basic Tavily search for the research question."""

    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        raise RuntimeError("TAVILY_API_KEY is missing from the .env file.")

    client = TavilyClient(api_key=api_key)

    response = client.search(
        query=question,
        search_depth="basic",
        max_results=5,
        include_answer=False,
        include_raw_content=False,
    )

    results = response.get("results", [])

    # Keep only the fields needed for research.
    sources = []

    for index, item in enumerate(results, start=1):
        sources.append(
            {
                "source_id": f"S{index}",
                "title": item.get("title", "Untitled"),
                "url": item.get("url", ""),
                "content": item.get("content", ""),
            }
        )

    return sources


def create_research_agent() -> Agent:
    """Create a research analyst using OpenRouter."""

    return Agent(
        role="Research Analyst",
        goal=(
            "Analyze retrieved web evidence and produce an "
            "accurate, concise research summary."
        ),
        backstory=(
            "You are a careful research analyst. You distinguish "
            "evidence from assumptions, identify conflicting "
            "findings, and never invent facts or citations."
        ),
        llm=get_llm(),
        verbose=False,
        allow_delegation=False,
        max_iter=1,
    )


def is_valid_research_question(question: str) -> tuple[bool, str]:
    """Validate a research question locally without API calls."""

    question = question.strip()

    if not question:
        return False, "Please enter a research question."

    if len(question) < 15:
        return False, ("Your question is too short. Please provide more detail.")

    # Extract alphabetic words.
    words = re.findall(r"[a-zA-Z]+", question.lower())

    if len(words) < 3:
        return False, (
            "Please enter a meaningful research question containing "
            "at least three words."
        )

    # Reject extremely long, unbroken alphabetic strings.
    # These are unlikely to be ordinary research-question text.
    for word in words:
        if len(word) >= 30:
            return False, (
                "Your input appears to contain random text. "
                "Please enter a meaningful research question."
            )

    # Reject excessive numbers and symbols.
    alphanumeric_chars = [char for char in question if char.isalnum()]

    alphabetic_chars = [char for char in question if char.isalpha()]

    if not alphanumeric_chars:
        return False, "Please enter a meaningful research question."

    if len(alphabetic_chars) / len(alphanumeric_chars) < 0.5:
        return False, (
            "Your input contains too many numbers or symbols. "
            "Please enter a research question."
        )

    # Detect suspiciously long words with unusual consonant patterns.
    suspicious_words = []

    for word in words:
        if len(word) >= 10:
            vowel_count = sum(1 for char in word if char in "aeiouy")

            vowel_ratio = vowel_count / len(word)

            has_unusual_consonant_cluster = bool(
                re.search(r"[bcdfghjklmnpqrstvwxyz]{5,}", word)
            )

            has_excessive_repetition = bool(re.search(r"(.)\1{3,}", word))

            if (
                vowel_ratio < 0.15
                or has_unusual_consonant_cluster
                or has_excessive_repetition
            ):
                suspicious_words.append(word)

    # Require multiple suspicious words before rejecting based
    # on these less conclusive word-level patterns.
    if len(suspicious_words) >= 2:
        return False, (
            "Your input appears to contain random text. "
            "Please enter a meaningful research question."
        )

    # Reject excessive repetition of words.
    if len(words) >= 4:
        counts = Counter(words)
        most_common_count = counts.most_common(1)[0][1]

        if most_common_count / len(words) > 0.6:
            return False, (
                "Your question contains too much repetition. " "Please rephrase it."
            )

    return True, ""


def research(question: str) -> str:
    """Search the web once and summarize the retrieved evidence."""

    is_valid, error_message = is_valid_research_question(question)

    if not is_valid:
        print(f"\nInvalid research question: {error_message}")
        return error_message

    question = question.strip()

    print("\nSearching the web...")

    # One explicit Tavily search request.
    sources = search_web(question)

    if not sources:
        return "No relevant sources were found. Try another question."

    # Prepare evidence for the LLM.
    evidence = "\n\n".join(
        (
            f"Source ID: {source['source_id']}\n"
            f"Title: {source['title']}\n"
            f"URL: {source['url']}\n"
            f"Content: {source['content']}"
        )
        for source in sources
    )

    agent = create_research_agent()

    task = Task(
        description=(
            "Answer the user's research question using the "
            "retrieved evidence below.\n\n"
            f"Research question:\n{question}\n\n"
            f"Retrieved evidence:\n{evidence}\n\n"
            "Requirements:\n"
            "1. Write a concise executive summary.\n"
            "2. Identify the key findings supported by the evidence.\n"
            "3. Mention contradictions or limitations when apparent.\n"
            "4. Cite supporting sources using their source IDs, "
            "such as [S1] or [S2].\n"
            "5. Do not invent facts, statistics, or URLs.\n"
            "6. Clearly identify questions the evidence cannot answer."
        ),
        expected_output=(
            "A research summary with an executive summary, "
            "evidence-supported findings, limitations, and "
            "source ID citations."
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

    print("Analyzing retrieved evidence...")

    result = crew.kickoff()

    # Print the generated report once.
    print("\n" + "=" * 60)
    print("RESEARCH REPORT")
    print("=" * 60)
    print(result.raw)

    # Display the original source URLs.
    print("\n" + "=" * 60)
    print("SOURCES")
    print("=" * 60)

    for source in sources:
        print(f"\n[{source['source_id']}] {source['title']}")
        print(source["url"])

    return result.raw


def main() -> None:
    question = input("Enter your research question: ").strip()

    try:
        research(question)
    except Exception as error:
        print(f"\nResearch failed: {error}")


if __name__ == "__main__":
    main()
