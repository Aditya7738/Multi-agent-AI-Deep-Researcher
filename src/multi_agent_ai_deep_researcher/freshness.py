"""Local helpers for detecting and evaluating time-sensitive research."""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from typing import Any
from email.utils import parsedate_to_datetime

_CURRENT_PATTERNS = (
    r"\bthis month\b",
    r"\bthis week\b",
    r"\btoday\b",
    r"\bthis year\b",
    r"\blatest\b",
    r"\bcurrent(?:ly)?\b",
    r"\brecent(?:ly)?\b",
    r"\bright now\b",
    r"\bas of now\b",
    r"\bwhat changed\b",
    r"\bnew developments?\b",
    r"\bbreaking news\b",
    r"\bupdat(?:e|es|ed|ing)\b",
)


def get_freshness_window(
    question: str,
    today: date | None = None,
) -> dict[str, Any]:
    """Determine the requested date range without network or LLM calls."""

    normalized = re.sub(
        r"\s+",
        " ",
        (question or "").casefold(),
    ).strip()

    current = today or date.today()

    needs_current = any(
        re.search(pattern, normalized)
        for pattern in _CURRENT_PATTERNS
    )

    if "this month" in normalized:
        start = current.replace(day=1)
        label = "this month"

    elif "this week" in normalized:
        start = current - timedelta(days=current.weekday())
        label = "this week"

    elif "today" in normalized:
        start = current
        label = "today"

    elif "this year" in normalized:
        start = current.replace(month=1, day=1)
        label = "this year"

    elif needs_current:
        start = current - timedelta(days=30)
        label = "last 30 days (default)"

    else:
        start = current - timedelta(days=30)
        label = "not applicable"

    return {
        "requires_current_information": needs_current,
        "start_date": start.isoformat(),
        "end_date": current.isoformat(),
        "label": label,
    }


# def _parse_date(value: Any) -> date | None:
#     """Parse common date strings without assuming missing dates are recent."""

#     if value is None:
#         return None

#     if isinstance(value, datetime):
#         return value.date()

#     if isinstance(value, date):
#         return value

#     text = str(value).strip()

#     if not text:
#         return None

#     try:
#         return datetime.fromisoformat(
#             text.replace("Z", "+00:00")
#         ).date()
#     except ValueError:
#         pass

#     formats = (
#         "%Y-%m-%d",
#         "%Y/%m/%d",
#         "%b %d, %Y",
#         "%B %d, %Y",
#         "%d %b %Y",
#         "%d %B %Y",
#     )

#     for date_format in formats:
#         try:
#             return datetime.strptime(text, date_format).date()
#         except ValueError:
#             continue

#     return None

def _parse_date(value: Any) -> date | None:
    """Parse ISO, RFC-style, and common publication date formats."""

    if value is None:
        return None

    if isinstance(value, datetime):
        return value.date()

    if isinstance(value, date):
        return value

    text = str(value).strip()

    if not text:
        return None

    # 1. ISO-8601 dates and timestamps.
    try:
        parsed = datetime.fromisoformat(
            text.replace("Z", "+00:00")
        )
        return parsed.date()
    except ValueError:
        pass

    # 2. RFC-style dates returned by Tavily, for example:
    # Mon, 26 Oct 2026 00:00:00 GMT
    try:
        parsed = parsedate_to_datetime(text)

        if parsed is not None:
            return parsed.date()
    except (TypeError, ValueError, OverflowError):
        pass

    # 3. Common date-only and human-readable formats.
    formats = (
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%b %d, %Y",
        "%B %d, %Y",
        "%d %b %Y",
        "%d %B %Y",
    )

    for date_format in formats:
        try:
            return datetime.strptime(text, date_format).date()
        except ValueError:
            continue

    return None


def assess_publication_date(
    value: Any,
    window: dict[str, Any],
) -> str:
    """Classify a source date against the requested date window."""

    if not window.get("requires_current_information"):
        return "not_checked"

    published = _parse_date(value)

    if published is None:
        return "undated"

    start = date.fromisoformat(window["start_date"])
    end = date.fromisoformat(window["end_date"])

    if published > end:
        return "future_date_check"

    if published < start:
        return "outdated_for_requested_window"

    return "recent_in_requested_window"