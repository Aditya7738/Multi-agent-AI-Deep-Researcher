
import json
import re
from pathlib import Path

import plotly.express as px
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ANALYSIS_FILE = PROJECT_ROOT / "critical_analysis.json"
INSIGHTS_FILE = PROJECT_ROOT / "insights.json"
CHART_DATA_FILE = PROJECT_ROOT / "chart_data.json"

CHART_DIR = PROJECT_ROOT / "generated_charts"
MANIFEST_FILE = PROJECT_ROOT / "visualization_manifest.json"

MAX_OBSERVATIONS = 100
MIN_OBSERVATIONS_FOR_CHART = 2


def load_json(path: Path) -> dict:
    """Load a JSON object from disk."""

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(f"{path.name} must contain a JSON object.")

    return data


def extract_source_registry(analysis_data: dict) -> dict[str, dict]:
    """Create a lookup of retrieved source IDs and their content."""

    sources = analysis_data.get("sources", [])

    if not isinstance(sources, list):
        raise ValueError("The sources field must be a list.")

    registry = {}

    for source in sources:
        if not isinstance(source, dict):
            continue

        source_id = str(source.get("source_id", "")).strip()

        if source_id:
            registry[source_id] = source

    return registry


def normalize_number(value: float) -> str:
    """Normalize numeric formatting for evidence checks."""

    return f"{value:.10f}".rstrip("0").rstrip(".")


def evidence_contains_value(value: float, evidence_text: str) -> bool:
    """
    Check whether the supplied source excerpt contains the number.

    This is a conservative text check, not proof that the number
    has been interpreted correctly.
    """

    text = evidence_text.replace(",", "")
    numeric_pattern = re.compile(
        r"(?<![\w.])[-+]?\d+(?:\.\d+)?%?(?![\w.])"
    )

    source_numbers = set()

    for match in numeric_pattern.finditer(text):
        token = match.group(0).rstrip("%")

        try:
            source_numbers.add(float(token))
        except ValueError:
            continue

    return any(
        abs(source_value - value) < 1e-9
        for source_value in source_numbers
    )


def validate_observations(
    observations: list[dict],
    source_registry: dict[str, dict],
) -> tuple[list[dict], list[str]]:
    """Validate numeric observations against retrieved source evidence."""

    accepted = []
    rejected = []

    if not isinstance(observations, list):
        return [], ["The observations field must be a list."]

    for index, observation in enumerate(
        observations[:MAX_OBSERVATIONS],
        start=1,
    ):
        if not isinstance(observation, dict):
            rejected.append(f"Observation {index}: invalid object.")
            continue

        label = str(observation.get("label", "")).strip()
        group = str(observation.get("group", "")).strip()
        source_id = str(observation.get("source_id", "")).strip()
        unit = str(observation.get("unit", "")).strip()
        evidence_text = str(
            observation.get("evidence_text", "")
        ).strip()

        try:
            value = float(observation["value"])
        except (KeyError, TypeError, ValueError):
            rejected.append(
                f"Observation {index}: missing numeric value."
            )
            continue

        if not label or not group or not unit:
            rejected.append(
                f"Observation {index}: label, group, and unit are required."
            )
            continue

        if source_id not in source_registry:
            rejected.append(
                f"Observation {index}: unknown source ID {source_id}."
            )
            continue

        if not evidence_text:
            rejected.append(
                f"Observation {index}: evidence excerpt is missing."
            )
            continue

        source_content = str(
            source_registry[source_id].get("content", "")
        )

        # Require the supplied excerpt to occur in the retrieved content.
        if evidence_text not in source_content:
            rejected.append(
                f"Observation {index}: evidence excerpt does not "
                "match the retrieved source content."
            )
            continue

        if not evidence_contains_value(value, evidence_text):
            rejected.append(
                f"Observation {index}: the numeric value is not "
                "present in the evidence excerpt."
            )
            continue

        # Percentage observations must explicitly use percentage units.
        if unit.lower() in {"percent", "%"}:
            if not re.search(r"(?<![\w.])" + re.escape(
                normalize_number(value)
            ) + r"\s*%", evidence_text.replace(",", "")):
                rejected.append(
                    f"Observation {index}: percentage value lacks "
                    "an explicit percentage marker in the excerpt."
                )
                continue

        accepted.append({
            "label": label,
            "group": group,
            "value": value,
            "unit": unit,
            "source_id": source_id,
            "source_url": source_registry[source_id].get("url", ""),
        })

    return accepted, rejected



def generate_charts(observations: list[dict]) -> list[dict]:
    """Generate separate charts for compatible units."""

    CHART_DIR.mkdir(parents=True, exist_ok=True)

    if len(observations) < MIN_OBSERVATIONS_FOR_CHART:
        return []

    dataframe = pd.DataFrame(observations)

    dataframe = dataframe.drop_duplicates(
        subset=["label", "group", "value", "unit", "source_id"]
    )

    # Normalize unit names for grouping, without mixing different units.
    dataframe["unit_key"] = dataframe["unit"].str.strip().str.casefold()

    charts = []

    for unit_key, unit_dataframe in dataframe.groupby(
        "unit_key", sort=True
    ):
        if len(unit_dataframe) < MIN_OBSERVATIONS_FOR_CHART:
            continue

        display_unit = str(unit_dataframe.iloc[0]["unit"]).strip()

        safe_unit = re.sub(r"[^a-z0-9]+", "_", unit_key).strip("_")
        chart_path = CHART_DIR / f"evidence_comparison_{safe_unit}.html"

        figure = px.bar(
            unit_dataframe,
            x="label",
            y="value",
            color="group",
            hover_data=["source_id", "source_url", "unit"],
            title=f"Source-Reported Numerical Observations ({display_unit})",
            labels={
                "label": "Observation",
                "value": f"Reported value ({display_unit})",
                "group": "Category",
            },
        )

        figure.update_layout(
            template="plotly_white",
            xaxis_title="Observation",
            yaxis_title=f"Reported value ({display_unit})",
            legend_title="Category",
            xaxis_tickangle=-25,
        )

        figure.write_html(
            str(chart_path),
            include_plotlyjs=True,
        )

        charts.append(
            {
                "chart_type": "bar",
                "title": f"Source-Reported Numerical Observations ({display_unit})",
                "path": str(chart_path.relative_to(PROJECT_ROOT)),
                "observation_count": len(unit_dataframe),
                "units": [display_unit],
                "source_ids": sorted(
                    unit_dataframe["source_id"].unique().tolist()
                ),
            }
        )

    return charts


def run_visualization() -> dict:
    """Validate chart data and generate evidence-based visualizations."""

    analysis_data = load_json(ANALYSIS_FILE)

    # Confirm the preceding analysis output exists.
    load_json(INSIGHTS_FILE)

    chart_data = load_json(CHART_DATA_FILE)

    source_registry = extract_source_registry(analysis_data)

    observations, rejected = validate_observations(
        chart_data.get("observations", []),
        source_registry,
    )

    charts = generate_charts(observations)

    result = {
        "status": "completed",
        "validated_observations": len(observations),
        "rejected_observations": len(rejected),
        "charts": charts,
        "message": (
            "Charts generated from validated numerical evidence."
            if charts
            else (
                "No chart was generated. Add sufficient, "
                "source-supported numerical observations."
            )
        ),
    }

    with MANIFEST_FILE.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)

    return result


def main() -> None:
    try:
        result = run_visualization()

        print("\nDATA VISUALIZATION RESULTS")
        print("=" * 60)
        print(json.dumps(result, indent=2, ensure_ascii=False))

    except (ValueError, FileNotFoundError, json.JSONDecodeError) as error:
        print(f"\nVisualization failed: {error}")


if __name__ == "__main__":
    main()
