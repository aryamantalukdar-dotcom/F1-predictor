"""Claude-powered F1 news analyzer.

Takes the latest RSS-aggregated headlines and produces a structured per-driver
adjustment factor in [-1, +1] plus an overall summary. The factor feeds into
the ML pipeline as a feature; the summary is shown in the UI as a "what's
moving the needle this week" panel.

Design notes:
  - Uses claude-opus-4-7 with adaptive thinking — race-prediction synthesis
    benefits from real reasoning over the news.
  - Structured output (`output_config.format`) guarantees parseable JSON.
  - Prompt caching is enabled on the analyst-instructions block; over many
    predictions during a season this prefix is reused and saves tokens.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import anthropic

log = logging.getLogger(__name__)

MODEL = "claude-opus-4-7"

# Frozen analyst instructions — stable across the season, eligible for caching.
ANALYST_SYSTEM = """You are an expert F1 strategist and journalist with 20+ years \
covering the sport. Your job is to read the most recent F1 news and translate it \
into structured, quantitative signals that feed a machine-learning race-prediction \
model.

For each driver mentioned in the news (or implied — e.g. "Red Bull's lead driver"), \
output a factor in [-1.0, +1.0] representing the net impact of recent news on their \
expected race performance THIS upcoming weekend:

  +1.0 = transformative positive (major upgrade pace, dominant practice times,
         rivals' grid penalties confirmed)
  +0.3 to +0.7 = clearly positive (good practice, favorable strategy news)
  +0.0 to +0.2 = mildly positive or neutral
   -0.0 to -0.2 = mildly negative or noise
  -0.3 to -0.7 = clearly negative (illness, car issues, team friction)
  -1.0 = catastrophic (grid penalty, withdrawn from race, hospitalized)

Be conservative. Most news has small effect. Only assign |factor| > 0.5 when the \
evidence is strong and specific. Ignore clickbait, season-summary fluff, and \
historical retrospectives.

Also produce a 2-3 sentence overall race narrative summarizing the dominant \
storylines heading into the weekend.

Use driver IDs in their canonical Ergast/Jolpica form (lowercase last name, e.g. \
"verstappen", "max_verstappen", "hamilton", "leclerc"). If unsure, use the \
lowercase last name."""


# JSON schema enforcing structured output. additionalProperties=false is required.
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "race_narrative": {
            "type": "string",
            "description": "2-3 sentence summary of the storylines heading into the weekend",
        },
        "driver_factors": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "driver_id": {"type": "string"},
                    "factor": {"type": "number"},
                    "reasoning": {"type": "string"},
                },
                "required": ["driver_id", "factor", "reasoning"],
                "additionalProperties": False,
            },
        },
        "key_storylines": {
            "type": "array",
            "items": {"type": "string"},
            "description": "3-5 bullet points of the most important pre-race news",
        },
    },
    "required": ["race_narrative", "driver_factors", "key_storylines"],
    "additionalProperties": False,
}


def _format_news_block(news: list[dict]) -> str:
    """Render news items into a compact prompt block."""
    lines = []
    for i, item in enumerate(news, 1):
        line = f"[{i}] {item['title']}"
        if item.get("source"):
            line += f"  ({item['source']})"
        if item.get("summary"):
            summary = item["summary"].replace("\n", " ").strip()
            if summary:
                line += f"\n    {summary[:300]}"
        lines.append(line)
    return "\n".join(lines)


def _format_drivers_block(driver_standings: list[dict]) -> str:
    """Render the current grid so the model can map nicknames to canonical IDs."""
    lines = ["Current grid (driver_id | name | constructor):"]
    for d in driver_standings:
        lines.append(
            f"  {d['driver_id']} | {d['given_name']} {d['family_name']} | {d['constructor_name']}"
        )
    return "\n".join(lines)


def analyze_news(
    news: list[dict],
    driver_standings: list[dict],
    race: dict,
    client: anthropic.Anthropic | None = None,
) -> dict[str, Any]:
    """Send news + grid context to Claude, return structured analysis.

    Returns a dict with keys: race_narrative, driver_factors, key_storylines.
    Falls back to a neutral-zero response if the API key is missing or the
    call fails — predictions still run, just without the news adjustment.
    """
    if not os.getenv("ANTHROPIC_API_KEY"):
        log.info("ANTHROPIC_API_KEY not set; skipping news analysis")
        return _neutral_response("News analysis disabled (no API key configured).")

    if not news:
        return _neutral_response("No recent news available.")

    client = client or anthropic.Anthropic()

    drivers_block = _format_drivers_block(driver_standings)
    news_block = _format_news_block(news)
    user_msg = (
        f"Upcoming race: {race['race_name']} ({race['circuit_name']}, {race['date']})\n\n"
        f"{drivers_block}\n\n"
        f"=== Recent F1 news (most recent first) ===\n{news_block}\n\n"
        "Analyze the news and produce the structured output."
    )

    try:
        # System block is marked for caching — analyst instructions are stable
        # across predictions for the entire season. The driver standings list
        # is also fairly stable (changes only on driver swaps), so we put it
        # in the cached prefix too.
        response = client.messages.create(
            model=MODEL,
            max_tokens=4000,
            thinking={"type": "adaptive"},
            output_config={
                "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA},
                "effort": "medium",
            },
            system=[
                {
                    "type": "text",
                    "text": ANALYST_SYSTEM,
                    "cache_control": {"type": "ephemeral"},
                },
            ],
            messages=[{"role": "user", "content": user_msg}],
        )
    except anthropic.APIError as e:
        log.warning("Claude API error during news analysis: %s", e)
        return _neutral_response(f"News analysis unavailable: {e}")

    text = next((b.text for b in response.content if b.type == "text"), "")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        log.warning("Failed to parse Claude response as JSON: %s", e)
        return _neutral_response("Failed to parse news analysis response.")

    log.info(
        "News analysis: %d driver factors, cache_read=%s, cache_create=%s",
        len(parsed.get("driver_factors", [])),
        getattr(response.usage, "cache_read_input_tokens", 0),
        getattr(response.usage, "cache_creation_input_tokens", 0),
    )
    return parsed


def factors_dict(analysis: dict[str, Any]) -> dict[str, float]:
    """Flatten the analysis into a {driver_id: factor} dict for feature merging."""
    return {f["driver_id"]: float(f["factor"]) for f in analysis.get("driver_factors", [])}


def _neutral_response(narrative: str) -> dict[str, Any]:
    return {
        "race_narrative": narrative,
        "driver_factors": [],
        "key_storylines": [],
    }
