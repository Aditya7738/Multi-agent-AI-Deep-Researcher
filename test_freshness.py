from datetime import date

from src.multi_agent_ai_deep_researcher.freshness import (
    get_freshness_window,
    assess_publication_date,
    _parse_date,
)


def test_tavily_publication_dates():
    assert _parse_date(
        "Mon, 26 Oct 2026 00:00:00 GMT"
    ) == date(2026, 10, 26)

    assert _parse_date(
        "Wed, 21 Oct 2026 08:44:23 GMT"
    ) == date(2026, 10, 21)


def test_current_month_freshness():
    window = get_freshness_window(
        "What changed in AI regulation this month?",
        today=date(2026, 10, 28),
    )

    assert window["start_date"] == "2026-10-01"
    assert window["end_date"] == "2026-10-28"

    assert assess_publication_date(
        "Mon, 26 Oct 2026 00:00:00 GMT",
        window,
    ) == "recent_in_requested_window"

    assert assess_publication_date(
        "Wed, 21 Oct 2026 08:44:23 GMT",
        window,
    ) == "recent_in_requested_window"

    assert assess_publication_date(
        None,
        window,
    ) == "undated"


if __name__ == "__main__":
    test_tavily_publication_dates()
    test_current_month_freshness()
    print("All freshness tests passed.")