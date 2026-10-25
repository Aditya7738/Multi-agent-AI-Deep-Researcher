import pytest

from src.multi_agent_ai_deep_researcher.research_with_web import (
    validate_research_scope,
)


@pytest.mark.parametrize(
    "question",
    [
        "Tell me everything about technology",
        "Tell me all about science",
        "Everything there is to know about history",
        "Explain technology",
        "What is artificial intelligence",
    ],
)
def test_broad_questions_are_rejected(question):
    is_specific, message = validate_research_scope(question)

    assert is_specific is False
    assert message


@pytest.mark.parametrize(
    "question",
    [
        "How is generative AI changing software engineering productivity?",
        "What cybersecurity risks affect small businesses using cloud services?",
        "How does AI-assisted coding affect code quality in Python projects?",
    ],
)
def test_focused_questions_are_accepted(question):
    is_specific, message = validate_research_scope(question)

    assert is_specific is True
    assert message == ""