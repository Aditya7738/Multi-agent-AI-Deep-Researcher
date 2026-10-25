import os
import re

from crewai import Agent, Crew, Process, Task
from dotenv import load_dotenv
from tavily import TavilyClient

from .llm_config import get_llm

# from collections import Counter



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


# A single term can be legitimate; multiple sequences together
# are a strong signal that the input is not a meaningful question.
_KEYBOARD_SEQUENCES = {
    "asdf",
    "qwer",
    "zxcv",
    "hjkl",
    "poiuy",
    "lkjh",
    "mnbv",
    "asdfgh",
    "qwerty",
    "zxcvbn",
}

# Questions that explicitly request an unlimited or excessively broad scope.
_BROAD_SCOPE_PATTERNS = (
    r"\beverything\s+about\b",
    r"\ball\s+about\b",
    r"\btell\s+me\s+everything\b",
    r"\bcover\s+everything\b",
    r"\bcomplete\s+overview\s+of\b",
    r"\beverything\s+there\s+is\s+to\s+know\b",
)

# Topics that are too general when submitted without a specific focus.
_BROAD_TOPICS = {
    "technology",
    "tech",
    "science",
    "history",
    "business",
    "education",
    "health",
    "artificial intelligence",
    "ai",
}


# def validate_research_scope(question: str) -> tuple[bool, str]:
#     """Reject excessively broad research requests without using an API."""

#     if not isinstance(question, str) or not question.strip():
#         return False, "Please enter a research question."

#     normalized = re.sub(r"\s+", " ", question.strip().lower())
#     normalized = re.sub(r"[.!?]+$", "", normalized).strip()

#     if any(re.search(pattern, normalized) for pattern in _BROAD_SCOPE_PATTERNS):
#         return (
#             False,
#             "Your research question is too broad. Please narrow the scope. "
#             "For example, specify a technology area, target group, "
#             "time period, or outcome you want to investigate.",
#         )

#     # Remove common introductory phrases to detect a topic-only request.
#     topic_only = re.sub(
#         r"^(please\s+)?(tell\s+me\s+about|explain|describe|research|"
#         r"investigate|study|analyze|analyse|what\s+is)\s+",
#         "",
#         normalized,
#     ).strip()

#     if topic_only in _BROAD_TOPICS:
#         return (
#             False,
#             "Your research question is too broad. Please choose a "
#             "specific aspect of the topic, such as its applications, "
#             "benefits, risks, or impact on a particular group.",
#         )

#     return (
#         True,
#         "",
#     )  # Questions that explicitly request an unlimited or excessively broad scope.


def validate_research_scope(question: str) -> tuple[bool, str]:
    """Reject excessively broad research requests without using an API."""

    if not isinstance(question, str) or not question.strip():
        return False, "Please enter a research question."

    normalized = re.sub(r"\s+", " ", question.strip().lower())
    normalized = re.sub(r"[.!?]+$", "", normalized).strip()

    if any(re.search(pattern, normalized) for pattern in _BROAD_SCOPE_PATTERNS):
        return (
            False,
            "Your research question is too broad. Please narrow the scope. "
            "For example, specify a technology area, target group, "
            "time period, or outcome you want to investigate.",
        )

    # Remove common introductory phrases to detect a topic-only request.
    topic_only = re.sub(
        r"^(please\s+)?(tell\s+me\s+about|explain|describe|research|"
        r"investigate|study|analyze|analyse|what\s+is)\s+",
        "",
        normalized,
    ).strip()

    if topic_only in _BROAD_TOPICS:
        return (
            False,
            "Your research question is too broad. Please choose a "
            "specific aspect of the topic, such as its applications, "
            "benefits, risks, or impact on a particular group.",
        )

    return True, ""


_BROAD_SCOPE_PATTERNS = (
    r"\beverything\s+about\b",
    r"\ball\s+about\b",
    r"\btell\s+me\s+everything\b",
    r"\bcover\s+everything\b",
    r"\bcomplete\s+overview\s+of\b",
    r"\beverything\s+there\s+is\s+to\s+know\b",
)

# Topics that are too general when submitted without a specific focus.
_BROAD_TOPICS = {
    "technology",
    "tech",
    "science",
    "history",
    "business",
    "education",
    "health",
    "artificial intelligence",
    "ai",
}




def is_valid_research_question(question: str) -> tuple[bool, str]:
    """Validate research input locally before making any API calls."""

    if not isinstance(question, str) or not question.strip():
        return False, "Please enter a research question."

    question = question.strip()
    words = re.findall(r"[a-zA-Z]+", question.lower())

    if len(question) < 15 or len(words) < 3:
        return False, "Please enter a specific, meaningful research question."

    # Detect multiple keyboard sequences, e.g.:
    # "asdf qwer zxcv random words"
    keyboard_matches = {word for word in words if word in _KEYBOARD_SEQUENCES}

    if len(keyboard_matches) >= 2:
        return (
            False,
            "Your input appears to contain random keyboard sequences. "
            "Please enter a meaningful research question.",
        )

    # Reject tokens that look like random character strings.
    suspicious_words = 0

    for word in words:
        if len(word) >= 30:
            return False, "Please remove unusually long random text."

        if len(word) >= 10:
            vowel_ratio = sum(char in "aeiou" for char in word) / len(word)

            if vowel_ratio < 0.15:
                suspicious_words += 1

            if re.search(r"(.)\1{4,}", word):
                suspicious_words += 1

    if suspicious_words >= 2:
        return (
            False,
            "Your input appears to contain random text. "
            "Please enter a meaningful research question.",
        )

    # Reject repeated-word spam.
    if words:
        most_common_count = max(words.count(word) for word in set(words))

        if most_common_count / len(words) > 0.6:
            return False, "Please enter a meaningful research question."

    return True, ""


# def is_valid_research_question(question: str) -> tuple[bool, str]:
#     """Validate a research question locally without API calls."""

#     question = question.strip()

#     if not question:
#         return False, "Please enter a research question."

#     if len(question) < 15:
#         return False, ("Your question is too short. Please provide more detail.")

#     # Extract alphabetic words.
#     words = re.findall(r"[a-zA-Z]+", question.lower())

#     if len(words) < 3:
#         return False, (
#             "Please enter a meaningful research question containing "
#             "at least three words."
#         )

#     # Reject extremely long, unbroken alphabetic strings.
#     # These are unlikely to be ordinary research-question text.
#     for word in words:
#         if len(word) >= 30:
#             return False, (
#                 "Your input appears to contain random text. "
#                 "Please enter a meaningful research question."
#             )

#     # Reject excessive numbers and symbols.
#     alphanumeric_chars = [char for char in question if char.isalnum()]

#     alphabetic_chars = [char for char in question if char.isalpha()]

#     if not alphanumeric_chars:
#         return False, "Please enter a meaningful research question."

#     if len(alphabetic_chars) / len(alphanumeric_chars) < 0.5:
#         return False, (
#             "Your input contains too many numbers or symbols. "
#             "Please enter a research question."
#         )

#     # Detect suspiciously long words with unusual consonant patterns.
#     suspicious_words = []

#     for word in words:
#         if len(word) >= 10:
#             vowel_count = sum(1 for char in word if char in "aeiouy")

#             vowel_ratio = vowel_count / len(word)

#             has_unusual_consonant_cluster = bool(
#                 re.search(r"[bcdfghjklmnpqrstvwxyz]{5,}", word)
#             )

#             has_excessive_repetition = bool(re.search(r"(.)\1{3,}", word))

#             if (
#                 vowel_ratio < 0.15
#                 or has_unusual_consonant_cluster
#                 or has_excessive_repetition
#             ):
#                 suspicious_words.append(word)

#     # Require multiple suspicious words before rejecting based
#     # on these less conclusive word-level patterns.
#     if len(suspicious_words) >= 2:
#         return False, (
#             "Your input appears to contain random text. "
#             "Please enter a meaningful research question."
#         )

#     # Reject excessive repetition of words.
#     if len(words) >= 4:
#         counts = Counter(words)
#         most_common_count = counts.most_common(1)[0][1]

#         if most_common_count / len(words) > 0.6:
#             return False, (
#                 "Your question contains too much repetition. " "Please rephrase it."
#             )

#     return True, ""


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
