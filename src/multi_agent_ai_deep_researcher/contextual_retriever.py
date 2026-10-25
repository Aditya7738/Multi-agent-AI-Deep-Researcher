import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import (
    parse_qsl,
    urlencode,
    urlsplit,
    urlunsplit,
)
from .cost_optimizer import optimize_queries
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

# Project root: the directory containing pyproject.toml.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
USAGE_FILE = PROJECT_ROOT / ".tavily_usage.json"

MAX_SEARCHES_PER_RUN = int(os.getenv("MAX_TAVILY_SEARCHES_PER_RUN", "2"))
MAX_RESULTS_PER_SEARCH = int(os.getenv("MAX_TAVILY_RESULTS_PER_SEARCH", "5"))
MAX_UNIQUE_SOURCES = int(os.getenv("MAX_UNIQUE_SOURCES", "8"))
MAX_MONTHLY_CREDITS = int(os.getenv("MAX_TAVILY_CREDITS_MONTH", "900"))

# Basic-search credit estimate. Verify against your Tavily plan.
CREDITS_PER_SEARCH = 1

# Query parameters that commonly track referrals rather than content.
TRACKING_PARAMS = {
    "fbclid",
    "gclid",
    "ref",
    "source",
}


def _read_usage() -> dict:
    """Read the local monthly Tavily usage ledger."""

    month = datetime.now(timezone.utc).strftime("%Y-%m")

    if not USAGE_FILE.exists():
        return {"month": month, "credits_used": 0}

    try:
        data = json.loads(USAGE_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise RuntimeError(
            "Cannot read the Tavily usage ledger. "
            "Inspect .tavily_usage.json before continuing."
        ) from exc

    if data.get("month") != month:
        return {"month": month, "credits_used": 0}

    return {
        "month": month,
        "credits_used": int(data.get("credits_used", 0)),
    }


def _write_usage(usage: dict) -> None:
    """Persist usage before making an API request."""

    USAGE_FILE.write_text(
        json.dumps(usage, indent=2),
        encoding="utf-8",
    )


def normalize_url(url: str) -> str:
    """Normalize URLs to identify duplicate sources."""

    parts = urlsplit(url.strip())

    if parts.scheme.lower() not in {"http", "https"}:
        return url.strip()

    hostname = (parts.hostname or "").lower()

    if not hostname:
        return url.strip()

    # Normalize the default HTTP/HTTPS ports.
    port = parts.port

    if port and not (
        (parts.scheme.lower() == "http" and port == 80)
        or (parts.scheme.lower() == "https" and port == 443)
    ):
        hostname = f"{hostname}:{port}"

    path = re.sub(r"/+", "/", parts.path).rstrip("/") or "/"

    query_params = []

    for key, value in parse_qsl(
        parts.query,
        keep_blank_values=True,
    ):
        key_lower = key.lower()

        if key_lower.startswith("utm_"):
            continue

        if key_lower in TRACKING_PARAMS:
            continue

        query_params.append((key, value))

    normalized_query = urlencode(sorted(query_params))

    return urlunsplit(
        (
            parts.scheme.lower(),
            hostname,
            path,
            normalized_query,
            "",  # Fragments do not identify a different web page.
        )
    )


def search_budget_status() -> dict:
    """Return the remaining local monthly search budget."""

    usage = _read_usage()

    return {
        "month": usage["month"],
        "credits_used": usage["credits_used"],
        "monthly_limit": MAX_MONTHLY_CREDITS,
        "credits_remaining": max(
            0,
            MAX_MONTHLY_CREDITS - usage["credits_used"],
        ),
    }


def retrieve_sources(subqueries: list[str]) -> list[dict]:
    """
    Execute a bounded number of Tavily searches and deduplicate results.

    The monthly ledger reserves estimated credits before each request.
    Failed requests remain charged locally as a conservative safeguard.
    """

    if MAX_SEARCHES_PER_RUN < 1:
        raise RuntimeError("MAX_TAVILY_SEARCHES_PER_RUN must be at least 1.")

    if MAX_RESULTS_PER_SEARCH < 1 or MAX_UNIQUE_SOURCES < 1:
        raise RuntimeError("Search-result and unique-source limits must be positive.")

    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        raise RuntimeError("TAVILY_API_KEY is missing from the .env file.")

    # Remove empty and repeated queries before spending credits.
    # unique_queries = []
    # seen_queries = set()

    # for query in subqueries:
    #     query = query.strip()
    #     normalized_query = re.sub(r"\s+", " ", query).casefold()

    #     if normalized_query and normalized_query not in seen_queries:
    #         unique_queries.append(query)
    #         seen_queries.add(normalized_query)

    # # Hard per-run limit.
    # unique_queries = unique_queries[:MAX_SEARCHES_PER_RUN]

    # if not unique_queries:
    #     return []


    # Optimize queries before creating a Tavily client or spending credits.
    usage = _read_usage()
    credits_remaining = max(
        0,
        MAX_MONTHLY_CREDITS - usage["credits_used"],
    )

    optimization = optimize_queries(
        subqueries,
        credits_remaining=credits_remaining,
        credits_per_search=CREDITS_PER_SEARCH,
        max_searches_per_run=MAX_SEARCHES_PER_RUN,
    )

    unique_queries = optimization["selected_queries"]

    print("\nCOST OPTIMIZER")
    print("=" * 40)
    print(
        f"Unique queries: "
        f"{optimization['unique_queries_after_deduplication']}"
    )
    print(f"Selected searches: {len(unique_queries)}")
    print(
        f"Estimated credits required: "
        f"{optimization['estimated_credits_required']}"
    )
    print(f"Credits remaining: {credits_remaining}")

    for skipped in optimization["skipped_queries"]:
        print(
            f"Skipped query: {skipped['query']} "
            f"({skipped['reason']})"
        )

    if not unique_queries:
        print("Cost Optimizer selected no searches.")
        return []


    ###################################################################

    client = TavilyClient(api_key=api_key)
    unique_sources = {}

    for query in unique_queries:
        usage = _read_usage()

        if usage["credits_used"] + CREDITS_PER_SEARCH > MAX_MONTHLY_CREDITS:
            print(
                "Tavily monthly application budget reached. "
                "Stopping further searches."
            )
            break

        # Reserve estimated credits BEFORE sending the request.
        # A timeout may still consume provider credits.
        usage["credits_used"] += CREDITS_PER_SEARCH
        _write_usage(usage)

        print(f"Searching Tavily: {query}")

        response = client.search(
            query=query,
            search_depth="basic",
            max_results=MAX_RESULTS_PER_SEARCH,
            include_answer=False,
            include_raw_content=False,
        )

        for item in response.get("results", []):
            url = item.get("url", "").strip()

            if not url:
                continue

            canonical_url = normalize_url(url)

            if canonical_url in unique_sources:
                # Preserve which queries discovered this source.
                existing = unique_sources[canonical_url]
                existing["matched_queries"].add(query)
                continue

            if len(unique_sources) >= MAX_UNIQUE_SOURCES:
                continue

            unique_sources[canonical_url] = {
                "source_id": f"S{len(unique_sources) + 1}",
                "title": item.get("title", "Untitled"),
                "url": url,
                "canonical_url": canonical_url,
                "content": item.get("content", ""),
                "relevance_score": item.get("score"),
                "matched_queries": {query},
            }

    # Convert sets to JSON-friendly lists.
    sources = []

    for source in unique_sources.values():
        source["matched_queries"] = sorted(source["matched_queries"])
        sources.append(source)

    print(f"\nUnique sources retrieved: {len(sources)}")
    print(f"Search budget: {search_budget_status()}")

    return sources


if __name__ == "__main__":
    print("Current Tavily application budget:")
    print(json.dumps(search_budget_status(), indent=2))

    test_queries = [
        "Generative AI effects on software developer productivity",
        "AI-assisted coding and software quality",
        "Generative AI effects on software developer employment",
        "AI coding tools and software engineering skills",
    ]

    sources = retrieve_sources(test_queries)

    print("\nRETRIEVED SOURCES")
    print("=" * 60)

    for source in sources:
        print(f"\n[{source['source_id']}] {source['title']}")
        print(source["url"])
        print("Found by:", ", ".join(source["matched_queries"]))
        print("Relevance:", source["relevance_score"])
