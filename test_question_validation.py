from src.multi_agent_ai_deep_researcher.research_with_web import (
    is_valid_research_question,
    research,
)

from unittest.mock import patch


def test_random_text_is_rejected():
    question = (
        "hahriuehadakrkjhiueesjabfjksahrwiue "
        "dhfhekhkjhwk 664816416987 dakfkakljklajja"
    )

    is_valid, _ = is_valid_research_question(question)

    assert not is_valid


def test_empty_question_is_rejected():
    is_valid, _ = is_valid_research_question("   ")

    assert not is_valid


def test_meaningful_question_is_accepted():
    question = (
        "What are the benefits and limitations "
        "of generative AI for software developers?"
    )

    is_valid, message = is_valid_research_question(question)

    assert is_valid, message


def test_invalid_question_does_not_call_tavily():
    question = (
        "How is Generative AI Changing Software Engineering?"
    )

    with patch("src.multi_agent_ai_deep_researcher.research_with_web.search_web") as mock_search:
        result = research(question)

        mock_search.assert_not_called()

        assert "random text" in result.lower()


if __name__ == "__main__":
    test_random_text_is_rejected()
    test_empty_question_is_rejected()
    test_meaningful_question_is_accepted()
    test_invalid_question_does_not_call_tavily()

    print("All question validation tests passed.")
