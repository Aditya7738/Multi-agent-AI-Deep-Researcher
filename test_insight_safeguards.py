from src.multi_agent_ai_deep_researcher.insight_generator import (
    _indicates_insufficient_evidence,
    _extract_named_targets,
    _is_background_insight,
)


def test_detects_insufficient_evidence():
    analysis = {
        "overall_assessment": (
            "The supplied sources contain no evidence about "
            "the fictional Norvale Institute."
        ),
        "uncertain_claims": [],
        "research_gaps": [],
    }

    assert _indicates_insufficient_evidence(analysis)


def test_extracts_named_target():
    targets = _extract_named_targets(
        "Did the fictional Norvale Institute publish a secret "
        "AI regulation standard in October 2026?"
    )

    assert "Norvale Institute" in targets


def test_unrelated_regulatory_finding_is_background():
    question = (
        "Did the fictional Norvale Institute publish a secret "
        "AI regulation standard in October 2026?"
    )

    sources = [
        {
            "source_id": "S1",
            "title": "Italy's AI implementing decree",
            "url": "https://example.com/italy-ai",
            "content": (
                "Italy published an AI implementing decree."
            ),
        }
    ]

    insight = {
        "title": "National AI regulation developments",
        "description": (
            "Italy introduced additional AI regulatory requirements."
        ),
        "reasoning": "The source discusses Italy's rules.",
        "evidence_source_ids": ["S1"],
    }

    assert _is_background_insight(
        insight,
        question,
        sources,
    )


def test_directly_relevant_insight_is_not_background():
    question = (
        "Did the fictional Norvale Institute publish a secret "
        "AI regulation standard in October 2026?"
    )

    insight = {
        "title": "Norvale Institute publication claim",
        "description": (
            "The claim about the Norvale Institute remains unverified."
        ),
        "reasoning": "No direct evidence establishes the claim.",
        "evidence_source_ids": ["S1"],
    }

    assert not _is_background_insight(
        insight,
        question,
        [
            {
                "source_id": "S1",
                "title": "Norvale Institute mention",
                "content": "A source mentions the Norvale Institute.",
                "url": "https://example.com/norvale",
            }
        ],
    )


if __name__ == "__main__":
    test_detects_insufficient_evidence()
    test_extracts_named_target()
    test_unrelated_regulatory_finding_is_background()
    test_directly_relevant_insight_is_not_background()

    print("All insight safeguard tests passed.")