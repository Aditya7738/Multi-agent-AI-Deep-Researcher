
import os
import re

DEFAULT_SEARCHES_PER_RUN = 2
DEFAULT_CREDITS_PER_SEARCH = 1

def normalize_query(query: str) -> str:
    """Normalize query text for duplicate detection."""
    return re.sub(r"\s+", " ", query.strip()).casefold()


def query_terms(query: str) -> set[str]:
    """Extract meaningful terms for measuring query overlap."""
    stop_words = {
        "a", "an", "and", "are", "as", "at", "by", "for",
        "from", "how", "in", "is", "of", "on", "or", "the",
        "to", "what", "when", "with", "impact", "effects",
    }

    return {
        word
        for word in re.findall(r"[a-zA-Z0-9]+", query.casefold())
        if len(word) > 1 and word not in stop_words
    }


def optimize_queries(
    queries: list[str],
    *,
    credits_remaining: int,
    credits_per_search: int = DEFAULT_CREDITS_PER_SEARCH,
    max_searches_per_run: int | None = None,
) -> dict:
    """
    Select distinct, useful search queries within the available budget.

    This function makes no network calls and uses no LLM tokens.
    """
    if credits_remaining < 0:
        credits_remaining = 0

    if credits_per_search < 1:
        raise ValueError("credits_per_search must be at least 1.")

    if max_searches_per_run is None:
        max_searches_per_run = int(
            os.getenv(
                "MAX_TAVILY_SEARCHES_PER_RUN",
                str(DEFAULT_SEARCHES_PER_RUN),
            )
        )

    if max_searches_per_run < 1:
        raise ValueError("max_searches_per_run must be at least 1.")

    # Deduplicate normalized queries while preserving their original order.
    unique_queries = []
    seen_queries = set()

    for query in queries:
        if not isinstance(query, str):
            continue

        query = re.sub(r"\s+", " ", query.strip())
        normalized = normalize_query(query)

        if not normalized or normalized in seen_queries:
            continue

        seen_queries.add(normalized)
        unique_queries.append(query)

    budget_limit = credits_remaining // credits_per_search
    search_limit = min(max_searches_per_run, budget_limit)

    selected = []
    skipped = []
    selected_term_sets = []

    # Prefer queries that contribute new terms instead of repeating
    # information already covered by selected queries.
    candidates = [
        {
            "query": query,
            "terms": query_terms(query),
            "original_index": index,
        }
        for index, query in enumerate(unique_queries)
    ]

    while candidates and len(selected) < search_limit:
        best_candidate = max(
            candidates,
            key=lambda item: (
                len(item["terms"] - set().union(*selected_term_sets))
                if selected_term_sets
                else len(item["terms"]),
                len(item["terms"]),
                -item["original_index"],
            ),
        )

        candidates.remove(best_candidate)
        selected.append(best_candidate["query"])
        selected_term_sets.append(best_candidate["terms"])

    selected_normalized = {
        normalize_query(query) for query in selected
    }

    for query in unique_queries:
        if normalize_query(query) not in selected_normalized:
            if budget_limit == 0:
                reason = "Monthly search budget exhausted."
            elif len(selected) >= max_searches_per_run:
                reason = "Per-run search limit reached."
            else:
                reason = "Lower priority or overlapping query."

            skipped.append(
                {
                    "query": query,
                    "reason": reason,
                }
            )

    return {
        "selected_queries": selected,
        "skipped_queries": skipped,
        "unique_queries_received": len(queries),
        "unique_queries_after_deduplication": len(unique_queries),
        "selected_searches": len(selected),
        "estimated_credits_required": (
            len(selected) * credits_per_search
        ),
        "credits_remaining_before_search": credits_remaining,
        "max_searches_per_run": max_searches_per_run,
    }


if __name__ == "__main__":
    example = optimize_queries(
        [
            "Generative AI and software engineering productivity",
            "Generative AI and software engineering productivity",
            "AI-assisted coding and code quality",
            "AI effects on developer employment",
        ],
        credits_remaining=2,
        max_searches_per_run=2,
    )

    import json

    print(json.dumps(example, indent=2))
