from src.multi_agent_ai_deep_researcher.critical_analysis import (
    enforce_evidence_sufficiency,
)


def test_one_source_is_insufficient():
    analysis = {
        "overall_assessment": "The claim may be true.",
        "findings": [
            {
                "claim": "An obscure claim",
                "source_ids": ["S1"],
                "evidence_summary": "One source mentions the claim.",
                "confidence": "low",
                "limitations": "",
            }
        ],
        "contradictions": [],
        "uncertain_claims": [],
        "research_gaps": [],
    }

    sources = [
        {
            "source_id": "S1",
            "title": "A single source",
            "content": "A brief, unverified mention of the claim.",
        }
    ]

    result = enforce_evidence_sufficiency(analysis, sources)

    assert result["evidence_sufficient"] is False
    assert "INSUFFICIENT EVIDENCE" in result["overall_assessment"]
    assert result["uncertain_claims"]
    assert result["research_gaps"]


def test_no_sources_is_insufficient():
    analysis = {
        "overall_assessment": "Analysis completed.",
        "findings": [],
        "contradictions": [],
        "uncertain_claims": [],
        "research_gaps": [],
    }

    result = enforce_evidence_sufficiency(analysis, [])

    assert result["evidence_sufficient"] is False
    assert "INSUFFICIENT EVIDENCE" in result["overall_assessment"]


def test_two_usable_sources_do_not_trigger_count_guard():
    analysis = {
        "overall_assessment": "Evidence supports a preliminary finding.",
        "findings": [
            {
                "claim": "A supported finding",
                "source_ids": ["S1", "S2"],
            }
        ],
        "contradictions": [],
        "uncertain_claims": [],
        "research_gaps": [],
    }

    sources = [
        {"source_id": "S1", "content": "Evidence from source one."},
        {"source_id": "S2", "content": "Evidence from source two."},
    ]

    result = enforce_evidence_sufficiency(analysis, sources)

    assert "evidence_sufficient" not in result
    assert result["overall_assessment"] != "INSUFFICIENT EVIDENCE"


if __name__ == "__main__":
    test_one_source_is_insufficient()
    test_no_sources_is_insufficient()
    test_two_usable_sources_do_not_trigger_count_guard()
    print("All insufficient-evidence tests passed.")