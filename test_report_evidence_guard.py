from src.multi_agent_ai_deep_researcher.report_builder import (
    detect_insufficient_evidence,
)


def test_no_evidence_detected():
    analysis = {
        "overall_assessment": (
            "The supplied sources contain no evidence about "
            "the fictional Norvale Institute."
        ),
        "uncertain_claims": [],
        "research_gaps": [],
    }

    assert detect_insufficient_evidence(analysis) is True


def test_explicit_insufficient_evidence_flag():
    analysis = {
        "evidence_sufficient": False,
        "overall_assessment": "Preliminary analysis.",
        "uncertain_claims": [],
        "research_gaps": [],
    }

    assert detect_insufficient_evidence(analysis) is True


def test_sufficient_evidence_not_flagged():
    analysis = {
        "evidence_sufficient": True,
        "overall_assessment": (
            "Multiple sources directly support the finding."
        ),
        "uncertain_claims": [],
        "research_gaps": [],
    }

    assert detect_insufficient_evidence(analysis) is False


if __name__ == "__main__":
    test_no_evidence_detected()
    test_explicit_insufficient_evidence_flag()
    test_sufficient_evidence_not_flagged()

    print("All report evidence guard tests passed.")