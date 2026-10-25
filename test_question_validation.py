# from src.multi_agent_ai_deep_researcher.research_with_web import (
#     is_valid_research_question,
#     research,
# )

# from unittest.mock import patch


# def test_random_text_is_rejected():
#     question = (
#         "hahriuehadakrkjhiueesjabfjksahrwiue "
#         "dhfhekhkjhwk 664816416987 dakfkakljklajja"
#     )

#     is_valid, _ = is_valid_research_question(question)

#     assert not is_valid


# def test_empty_question_is_rejected():
#     is_valid, _ = is_valid_research_question("   ")

#     assert not is_valid


# def test_meaningful_question_is_accepted():
#     question = (
#         "What are the benefits and limitations "
#         "of generative AI for software developers?"
#     )

#     is_valid, message = is_valid_research_question(question)

#     assert is_valid, message


# def test_invalid_question_does_not_call_tavily():
#     question = (
#         "How is Generative AI Changing Software Engineering?"
#     )

#     with patch("src.multi_agent_ai_deep_researcher.research_with_web.search_web") as mock_search:
#         result = research(question)

#         mock_search.assert_not_called()

#         assert "random text" in result.lower()


# if __name__ == "__main__":
#     test_random_text_is_rejected()
#     test_empty_question_is_rejected()
#     test_meaningful_question_is_accepted()
#     test_invalid_question_does_not_call_tavily()

#     print("All question validation tests passed.")
#
#######################################################################3

from unittest.mock import patch
from app import run_research_workflow

from src.multi_agent_ai_deep_researcher.research_with_web import (
    is_valid_research_question,
)

def test_gibberish_does_not_start_research_workflow():
    from src.multi_agent_ai_deep_researcher.research_with_web import (
        is_valid_research_question,
    )

    question = "asdf qwer zxcv random words"
    valid, _ = is_valid_research_question(question)

    assert valid is False

    with patch("app.run_research_workflow") as mock_workflow:
        if valid:
            run_research_workflow(question)

        mock_workflow.assert_not_called()

def test_keyboard_gibberish_is_rejected():
    valid, message = is_valid_research_question(
        "asdf qwer zxcv random words"
    )

    assert valid is False
    assert "keyboard sequences" in message.lower()


def test_empty_input_is_rejected():
    valid, _ = is_valid_research_question("")
    assert valid is False


def test_meaningful_research_question_is_accepted():
    valid, message = is_valid_research_question(
        "How does generative AI affect software engineering productivity?"
    )

    assert valid is True
    assert message == ""


def test_single_keyboard_term_is_not_automatically_rejected():
    valid, _ = is_valid_research_question(
        "How does the QWERTY keyboard layout affect typing speed?"
    )

    assert valid is True
