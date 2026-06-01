"""Generate a fully self-contained static index.html with baked-in prediction data.

Usage (run from repo root):
    python -m scripts.generate_static [--out path/to/index.html]

Intended to be called by the GitHub Actions workflow. The resulting file is
committed to the gh-pages branch and served by GitHub Pages.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from datetime import datetime, timezone

# Ensure repo root is on path when invoked directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import predict  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)


def _esc(s: object) -> str:
    return (
        str(s or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#039;")
    )


def _fmt(n: float, d: int = 2) -> str:
    if n is None or not math.isfinite(float(n)):
        return "—"
    return f"{float(n):.{d}f}"


def _factor_class(f: float) -> str:
    if f > 0.05:
        return "factor-pos"
    if f < -0.05:
        return "factor-neg"
    return "factor-neutral"


def _race_banner(race: dict) -> str:
    from datetime import date as dt_date

    try:
        d = datetime.strptime(race["date"], "%Y-%m-%d")
        date_str = d.strftime("%A, %B %-d, %Y")
    except Exception:
        date_str = race.get("date", "")
    return f"""
    <div>
      <h1 class="race-name">{_esc(race.get("race_name", ""))}</h1>
      <p class="race-meta">{_esc(race.get("circuit_name",""))} &middot; {_esc(race.get("locality",""))}, {_esc(race.get("country",""))} &middot; Round {_esc(race.get("round",""))}</p>
    </div>
    <div class="race-date">{_esc(date_str)}</div>
"""


def _weather_card(w: dict) -> str:
    rain_pct = round((w.get("rain_probability") or 0) * 100)
    rain_class = "rain-warn" if rain_pct >= 50 else ""
    note = "Open-Meteo forecast" if w.get("is_forecast") else "Climatological estimate (forecast not yet available)"
    return f"""
    <h2>Race-Day Weather</h2>
    <div class="weather-grid">
      <div class="weather-stat">
        <div class="label">Rain probability</div>
        <div class="value {rain_class}">{rain_pct}%</div>
      </div>
      <div class="weather-stat">
        <div class="label">Temperature</div>
        <div class="value">{_fmt(w.get("temperature_c"), 1)}&deg;C</div>
      </div>
      <div class="weather-stat">
        <div class="label">Precipitation</div>
        <div class="value">{_fmt(w.get("precipitation_mm"), 1)} mm</div>
      </div>
      <div class="weather-stat">
        <div class="label">Wind</div>
        <div class="value">{_fmt(w.get("wind_kph"), 0)} kph</div>
      </div>
    </div>
    <p class="muted" style="margin-top:14px">{_esc(note)}</p>
"""


def _pole_card(pole: dict | None, pole_predictions: list[dict]) -> str:
    if not pole:
        return "<h2>Pole Position Prediction</h2><div class='card-loading'>No prediction available</div>"
    others = pole_predictions[1:5]
    secondary = ""
    if others:
        items = "".join(
            f"<li><span class='pos'>{d['rank']}.</span><span class='code'>{_esc(d['code'])}</span><span>{_esc(d['driver_name'])}</span></li>"
            for d in others
        )
        secondary = f"""
      <div class="pole-secondary">
        <div class="muted">Most likely front runners</div>
        <ul class="pole-secondary-list">{items}</ul>
      </div>"""
    return f"""
    <h2>Pole Position Prediction</h2>
    <div class="pole-driver">
      <div class="pole-code">{_esc(pole.get("code",""))}</div>
      <div class="pole-info">
        <div class="pole-name">{_esc(pole.get("driver_name",""))}</div>
        <div class="pole-team">{_esc(pole.get("constructor",""))}</div>
      </div>
    </div>{secondary}
"""


def _race_table(predictions: list[dict]) -> str:
    if not predictions:
        return "<div class='card-loading'>No prediction available</div>"
    max_prob = max(d.get("win_probability", 0) for d in predictions) or 1
    rows = []
    for d in predictions:
        rank = d.get("rank", 0)
        podium = {1: "podium-1", 2: "podium-2", 3: "podium-3"}.get(rank, "")
        win_pct = round(d.get("win_probability", 0) * 1000) / 10
        bar_w = (d.get("win_probability", 0) / max_prob) * 80
        f = d.get("news_factor", 0)
        fsign = "+" if f > 0 else ""
        rows.append(f"""
        <tr class="{podium}">
          <td class="rank">P{rank}</td>
          <td class="code">{_esc(d.get("code",""))}</td>
          <td class="driver-name">{_esc(d.get("driver_name",""))}</td>
          <td class="team-name">{_esc(d.get("constructor",""))}</td>
          <td class="win-prob">{win_pct}%<span class="win-prob-bar" style="width:{bar_w:.0f}px"></span></td>
          <td class="factor {_factor_class(f)}">{fsign}{_fmt(f,2)}</td>
        </tr>""")
    return f"""
    <div class="table-scroll">
    <table>
      <thead>
        <tr>
          <th>Pos</th><th>Code</th><th>Driver</th><th>Team</th>
          <th>Win prob</th><th class="col-factor">News factor</th>
        </tr>
      </thead>
      <tbody>{"".join(rows)}</tbody>
    </table>
    </div>"""


def _news_section(news: dict, meta: dict, updated: str) -> str:
    narrative = _esc(news.get("narrative") or "No narrative available.")
    storylines = news.get("storylines") or []
    sl_items = "".join(f"<li>{_esc(s)}</li>" for s in storylines) or "<li class='muted'>No major storylines detected.</li>"
    news_items = news.get("items") or []
    ni_items = []
    for n in news_items[:10]:
        when = ""
        if n.get("published"):
            try:
                when = datetime.fromisoformat(str(n["published"])).strftime("%-d %b %Y %H:%M")
            except Exception:
                when = str(n["published"])
        ni_items.append(
            f"<li><a href='{_esc(n.get('link','#'))}' target='_blank' rel='noopener'>{_esc(n.get('title',''))}</a>"
            f"<div class='news-meta'>{_esc(n.get('source',''))} {('&middot; ' + when) if when else ''}</div></li>"
        )
    ni_html = "".join(ni_items) or "<li class='muted'>No recent news fetched.</li>"
    meta_str = f"Race model: {_esc(meta.get('race_model',''))} &middot; Pole model: {_esc(meta.get('pole_model',''))} &middot; {_esc(meta.get('n_drivers',''))} drivers" if meta else ""
    return f"""
    <p class="muted" id="model-meta">{meta_str}</p>
    <p class="muted update-time">Last updated: {_esc(updated)}</p>

    <section class="card">
      <h2>What's Driving the Prediction</h2>
      <p class="narrative">{narrative}</p>
      <h3>Key storylines</h3>
      <ul id="storylines">{sl_items}</ul>
    </section>

    <section class="card">
      <h2>Latest F1 News</h2>
      <ul id="news-list">{ni_html}</ul>
    </section>"""


def build_html(data: dict, updated: str) -> str:
    race = data.get("race") or {}
    weather = data.get("weather") or {}
    pole = data.get("pole_prediction")
    pole_preds = data.get("pole_predictions") or []
    race_preds = data.get("race_predictions") or []
    news = data.get("news") or {}
    meta = data.get("meta") or {}

    css = open(os.path.join(os.path.dirname(__file__), "..", "frontend", "style.css")).read()
    extra_css = """
.update-time { text-align: right; margin-bottom: 8px; }
#model-meta { margin: 0 0 4px 0; }
"""

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>F1 Predictor &mdash; {_esc(race.get("race_name","Next Race"))}</title>
  <style>
{css}
{extra_css}
  </style>
</head>
<body>
  <header>
    <div class="brand">
      <span class="logo">F1</span>
      <span class="title">Predictor</span>
    </div>
    <div class="muted" style="font-size:0.8rem">Updated {_esc(updated)}</div>
  </header>

  <main>
    <section class="card race-banner">
      {_race_banner(race)}
    </section>

    <section class="grid-two">
      <div class="card pole-card">
        {_pole_card(pole, pole_preds)}
      </div>
      <div class="card weather-card">
        {_weather_card(weather)}
      </div>
    </section>

    <section class="card">
      <h2>Predicted Finishing Order</h2>
      {_news_section.__doc__ and "" or ""}
      {_race_table(race_preds)}
    </section>

    {_news_section(news, meta, updated)}
  </main>

  <footer>
    <p class="muted">
      Live data: Jolpica/Ergast &middot; OpenF1 &middot; Open-Meteo &middot; F1 news RSS &middot;
      Claude Sonnet 4.6 for news analysis. Predictions are model estimates, not betting advice.
    </p>
  </footer>
</body>
</html>"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate static F1 prediction page")
    parser.add_argument("--out", default="docs/index.html", help="Output HTML path")
    args = parser.parse_args()

    log.info("Running prediction pipeline...")
    try:
        data = predict.predict_next_race()
    except RuntimeError as e:
        # Transient upstream failure (typically Jolpica/Open-Meteo timeout).
        # Exit cleanly so the workflow doesn't email a failure and the
        # existing prediction page stays in place until the next run.
        log.warning("Skipping page regeneration — upstream data unavailable: %s", e)
        return 0
    except Exception as e:
        # Same treatment for any other unexpected failure: leave the existing
        # page alone rather than committing a broken one.
        log.exception("Skipping page regeneration — unexpected error: %s", e)
        return 0

    updated = datetime.now(timezone.utc).strftime("%-d %b %Y %H:%M UTC")
    html = build_html(data, updated)

    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(html)

    log.info("Written %d bytes to %s", len(html), out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
