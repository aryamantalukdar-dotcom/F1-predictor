"""Live data sources for the F1 predictor.

Wraps every external feed in one place so the rest of the pipeline can stay
synchronous and dumb. Each source has a small in-memory TTL cache so we don't
hammer free public APIs during a single prediction run.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import logging
import os
import time
from typing import Any

import feedparser
import httpx
import pandas as pd

log = logging.getLogger(__name__)

# Jolpica is the actively-maintained successor to the Ergast F1 API.
JOLPICA_BASE = "https://api.jolpi.ca/ergast/f1"
OPENF1_BASE = "https://api.openf1.org/v1"
OPEN_METEO_BASE = "https://api.open-meteo.com/v1/forecast"

F1_NEWS_FEEDS = [
    "https://www.autosport.com/rss/f1/news/",
    "https://www.motorsport.com/rss/f1/news/",
    "https://www.formula1.com/content/fom-website/en/latest/all.xml",
    "https://www.bbc.com/sport/formula1/rss.xml",
]

_CACHE: dict[str, tuple[float, Any]] = {}


def _cached(key: str, ttl: float, fn):
    now = time.time()
    hit = _CACHE.get(key)
    if hit and now - hit[0] < ttl:
        return hit[1]
    value = fn()
    _CACHE[key] = (now, value)
    return value


def _http_get(url: str, params: dict | None = None, timeout: float = 8.0) -> dict:
    """GET JSON with retry + exponential backoff.

    Jolpica/Ergast occasionally returns transient 429s and 5xx (and the API
    can go fully unreachable for short periods). Retry up to 3 times with
    backoff at 1s, 2s. Per-attempt timeout kept short (8s) so the worst case
    is ~28s of waiting before we bail and let the caller handle it.
    """
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            with httpx.Client(timeout=timeout, headers={"User-Agent": "f1-predictor/1.0"}) as c:
                r = c.get(url, params=params)
                # 404 is a real error (e.g. season not yet), fail fast.
                if r.status_code == 404:
                    r.raise_for_status()
                if r.status_code >= 500 or r.status_code == 429:
                    raise httpx.HTTPStatusError(
                        f"transient {r.status_code}", request=r.request, response=r
                    )
                r.raise_for_status()
                return r.json()
        except httpx.HTTPStatusError as e:
            # Don't retry 404; it's terminal.
            if e.response is not None and e.response.status_code == 404:
                raise
            last_exc = e
        except (httpx.TransportError, httpx.TimeoutException) as e:
            last_exc = e
        if attempt < 2:
            time.sleep(2**attempt)  # 1s, 2s
    assert last_exc is not None
    raise last_exc


# ---------------------------------------------------------------------------
# Jolpica (Ergast) — schedule, standings, results
# ---------------------------------------------------------------------------


def get_season_schedule(season: int) -> list[dict]:
    """Return the full race calendar for the season."""

    def _fetch():
        data = _http_get(f"{JOLPICA_BASE}/{season}.json")
        races = data["MRData"]["RaceTable"]["Races"]
        out = []
        for r in races:
            out.append(
                {
                    "season": int(r["season"]),
                    "round": int(r["round"]),
                    "race_name": r["raceName"],
                    "circuit_id": r["Circuit"]["circuitId"],
                    "circuit_name": r["Circuit"]["circuitName"],
                    "country": r["Circuit"]["Location"]["country"],
                    "locality": r["Circuit"]["Location"]["locality"],
                    "lat": float(r["Circuit"]["Location"]["lat"]),
                    "lon": float(r["Circuit"]["Location"]["long"]),
                    "date": r["date"],
                    "time": r.get("time"),
                }
            )
        return out

    return _cached(f"sched:{season}", ttl=3600, fn=_fetch)


def get_next_race(today: dt.date | None = None) -> dict | None:
    """Find the next upcoming race across the current/next season."""
    today = today or dt.date.today()
    for season in (today.year, today.year + 1):
        try:
            sched = get_season_schedule(season)
        except Exception as e:
            log.warning("schedule fetch failed for %s: %s", season, e)
            continue
        for race in sched:
            if dt.date.fromisoformat(race["date"]) >= today:
                return race
    return None


def get_driver_standings(season: int) -> list[dict]:
    def _fetch():
        data = _http_get(f"{JOLPICA_BASE}/{season}/driverStandings.json")
        lists = data["MRData"]["StandingsTable"]["StandingsLists"]
        if not lists:
            return []
        return [
            {
                "driver_id": s["Driver"]["driverId"],
                "code": s["Driver"].get("code", s["Driver"]["driverId"][:3].upper()),
                "given_name": s["Driver"]["givenName"],
                "family_name": s["Driver"]["familyName"],
                "constructor_id": s["Constructors"][0]["constructorId"],
                "constructor_name": s["Constructors"][0]["name"],
                "points": float(s["points"]),
                "position": int(s["position"]),
                "wins": int(s["wins"]),
            }
            for s in lists[0]["DriverStandings"]
        ]

    return _cached(f"standings:{season}", ttl=3600, fn=_fetch)


def get_constructor_standings(season: int) -> list[dict]:
    def _fetch():
        data = _http_get(f"{JOLPICA_BASE}/{season}/constructorStandings.json")
        lists = data["MRData"]["StandingsTable"]["StandingsLists"]
        if not lists:
            return []
        return [
            {
                "constructor_id": s["Constructor"]["constructorId"],
                "name": s["Constructor"]["name"],
                "points": float(s["points"]),
                "position": int(s["position"]),
                "wins": int(s["wins"]),
            }
            for s in lists[0]["ConstructorStandings"]
        ]

    return _cached(f"cstandings:{season}", ttl=3600, fn=_fetch)


def get_recent_results(season: int, last_n: int | None = None) -> pd.DataFrame:
    """All race results for the season, optionally limited to the last N rounds."""

    def _fetch():
        rows = []
        # Ergast caps page size at 100; one season fits comfortably.
        data = _http_get(f"{JOLPICA_BASE}/{season}/results.json", params={"limit": 1000})
        for race in data["MRData"]["RaceTable"]["Races"]:
            for res in race["Results"]:
                rows.append(
                    {
                        "season": int(race["season"]),
                        "round": int(race["round"]),
                        "circuit_id": race["Circuit"]["circuitId"],
                        "date": race["date"],
                        "driver_id": res["Driver"]["driverId"],
                        "constructor_id": res["Constructor"]["constructorId"],
                        "grid": int(res["grid"]),
                        "position": int(res["position"]) if res["position"].isdigit() else None,
                        "position_text": res["positionText"],
                        "points": float(res["points"]),
                        "status": res["status"],
                        "laps": int(res["laps"]),
                    }
                )
        return pd.DataFrame(rows)

    df = _cached(f"results:{season}", ttl=1800, fn=_fetch)
    if last_n is not None and not df.empty:
        rounds = sorted(df["round"].unique())[-last_n:]
        df = df[df["round"].isin(rounds)].copy()
    return df


def get_qualifying_results(season: int) -> pd.DataFrame:
    def _fetch():
        rows = []
        data = _http_get(f"{JOLPICA_BASE}/{season}/qualifying.json", params={"limit": 1000})
        for race in data["MRData"]["RaceTable"]["Races"]:
            for q in race.get("QualifyingResults", []):
                rows.append(
                    {
                        "season": int(race["season"]),
                        "round": int(race["round"]),
                        "circuit_id": race["Circuit"]["circuitId"],
                        "driver_id": q["Driver"]["driverId"],
                        "constructor_id": q["Constructor"]["constructorId"],
                        "qual_position": int(q["position"]),
                        "q1": q.get("Q1"),
                        "q2": q.get("Q2"),
                        "q3": q.get("Q3"),
                    }
                )
        return pd.DataFrame(rows)

    return _cached(f"qual:{season}", ttl=1800, fn=_fetch)


def get_sprint_results(season: int) -> pd.DataFrame:
    """Sprint qualifying + sprint race results. Returns empty frame if season has no sprints yet."""
    def _fetch():
        rows = []
        try:
            data = _http_get(f"{JOLPICA_BASE}/{season}/sprint.json", params={"limit": 500})
        except Exception:
            return pd.DataFrame()
        for race in data["MRData"]["RaceTable"]["Races"]:
            for r in race.get("SprintResults", []):
                rows.append(
                    {
                        "season": int(race["season"]),
                        "round": int(race["round"]),
                        "circuit_id": race["Circuit"]["circuitId"],
                        "driver_id": r["Driver"]["driverId"],
                        "constructor_id": r["Constructor"]["constructorId"],
                        "sprint_grid": int(r["grid"]) if r.get("grid") else None,
                        "sprint_position": int(r["position"]),
                    }
                )
        return pd.DataFrame(rows)

    return _cached(f"sprint:{season}", ttl=1800, fn=_fetch)


def get_circuit_history(circuit_id: str, seasons_back: int = 5) -> pd.DataFrame:
    """Per-driver historical results at a specific circuit across recent seasons."""
    today = dt.date.today()
    frames = []
    for season in range(today.year - seasons_back, today.year + 1):
        try:
            df = get_recent_results(season)
        except Exception:
            continue
        if not df.empty:
            frames.append(df[df["circuit_id"] == circuit_id])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------------------
# Open-Meteo — free weather forecast, no key required
# ---------------------------------------------------------------------------


def get_weather_forecast(lat: float, lon: float, target_date: dt.date) -> dict:
    """Forecast snapshot for race day. Falls back to climatology if too far ahead."""
    days_ahead = (target_date - dt.date.today()).days
    if days_ahead < 0 or days_ahead > 14:
        # Open-Meteo forecast horizon is 16 days; beyond that, return neutral defaults.
        return {
            "rain_probability": 0.2,
            "precipitation_mm": 0.0,
            "temperature_c": 22.0,
            "wind_kph": 10.0,
            "is_forecast": False,
        }

    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ",".join(
            [
                "temperature_2m_max",
                "temperature_2m_min",
                "precipitation_sum",
                "precipitation_probability_max",
                "windspeed_10m_max",
            ]
        ),
        "timezone": "auto",
        "start_date": target_date.isoformat(),
        "end_date": target_date.isoformat(),
    }
    try:
        data = _http_get(OPEN_METEO_BASE, params=params)
        d = data["daily"]
        return {
            "rain_probability": (d["precipitation_probability_max"][0] or 0) / 100.0,
            "precipitation_mm": float(d["precipitation_sum"][0] or 0),
            "temperature_c": (
                (float(d["temperature_2m_max"][0]) + float(d["temperature_2m_min"][0])) / 2
            ),
            "wind_kph": float(d["windspeed_10m_max"][0] or 0),
            "is_forecast": True,
        }
    except Exception as e:
        log.warning("weather fetch failed: %s", e)
        return {
            "rain_probability": 0.2,
            "precipitation_mm": 0.0,
            "temperature_c": 22.0,
            "wind_kph": 10.0,
            "is_forecast": False,
        }


# ---------------------------------------------------------------------------
# F1 news — RSS aggregation
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class NewsItem:
    title: str
    summary: str
    link: str
    published: str
    source: str


def get_recent_news(max_items: int = 40, hours_back: int = 96) -> list[NewsItem]:
    """Pull and dedupe recent F1 news from public RSS feeds."""

    def _fetch():
        cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=hours_back)
        seen: set[str] = set()
        items: list[NewsItem] = []
        for url in F1_NEWS_FEEDS:
            try:
                feed = feedparser.parse(url)
            except Exception as e:
                log.warning("rss fetch failed for %s: %s", url, e)
                continue
            source = feed.feed.get("title", url)
            for entry in feed.entries:
                title = entry.get("title", "").strip()
                if not title or title in seen:
                    continue
                seen.add(title)
                pub = entry.get("published_parsed") or entry.get("updated_parsed")
                if pub:
                    pub_dt = dt.datetime(*pub[:6], tzinfo=dt.timezone.utc)
                    if pub_dt < cutoff:
                        continue
                    pub_iso = pub_dt.isoformat()
                else:
                    pub_iso = ""
                items.append(
                    NewsItem(
                        title=title,
                        summary=(entry.get("summary", "") or "")[:500],
                        link=entry.get("link", ""),
                        published=pub_iso,
                        source=source,
                    )
                )
        items.sort(key=lambda x: x.published, reverse=True)
        return items[:max_items]

    return _cached(f"news:{max_items}:{hours_back}", ttl=900, fn=_fetch)


# ---------------------------------------------------------------------------
# OpenF1 — live timing (used only when a session is active)
# ---------------------------------------------------------------------------


def get_latest_session_meta() -> dict | None:
    """Most recent OpenF1 session — useful for live qualifying-aware predictions."""
    try:
        data = _http_get(f"{OPENF1_BASE}/sessions", params={"limit": 1, "order": "desc"})
        return data[0] if data else None
    except Exception as e:
        log.warning("openf1 sessions fetch failed: %s", e)
        return None


def get_practice_pace() -> pd.DataFrame:
    """Aggregate per-driver practice pace for the current/latest race meeting.

    Pulls lap times for every Practice session of the meeting via OpenF1,
    filters out in/out laps and obvious outliers (>200s), and returns one
    row per driver with their best and median clean lap.

    Returns a DataFrame with columns: driver_code, best_lap_s, median_lap_s,
    n_laps. Empty if no practice has happened yet or OpenF1 is unreachable.
    """
    def _fetch():
        meetings = _http_get(f"{OPENF1_BASE}/meetings", params={"meeting_key": "latest"})
        if not meetings:
            return pd.DataFrame()
        meeting_key = meetings[0]["meeting_key"]

        sessions = _http_get(f"{OPENF1_BASE}/sessions", params={"meeting_key": meeting_key})
        practice_sessions = [s for s in sessions if s.get("session_type") == "Practice"]
        if not practice_sessions:
            return pd.DataFrame()

        # Driver number → 3-letter code mapping (stable across the weekend)
        latest_key = max(s["session_key"] for s in practice_sessions)
        drivers = _http_get(f"{OPENF1_BASE}/drivers", params={"session_key": latest_key})
        num_to_code = {d["driver_number"]: d.get("name_acronym", "") for d in drivers}

        all_laps = []
        for sess in practice_sessions:
            try:
                laps = _http_get(f"{OPENF1_BASE}/laps", params={"session_key": sess["session_key"]})
            except Exception as e:
                log.warning("openf1 laps fetch failed for session %s: %s", sess["session_key"], e)
                continue
            for lap in laps:
                dur = lap.get("lap_duration")
                if not dur or dur > 200 or lap.get("is_pit_out_lap"):
                    continue
                all_laps.append({"driver_number": lap["driver_number"], "lap_duration": float(dur)})

        if not all_laps:
            return pd.DataFrame()

        df = pd.DataFrame(all_laps)
        agg = (
            df.groupby("driver_number")
            .agg(best_lap_s=("lap_duration", "min"),
                 median_lap_s=("lap_duration", "median"),
                 n_laps=("lap_duration", "count"))
            .reset_index()
        )
        agg["driver_code"] = agg["driver_number"].map(num_to_code)
        return agg[agg["driver_code"].astype(bool)].reset_index(drop=True)

    try:
        return _cached("practice_pace:latest", ttl=900, fn=_fetch)
    except Exception as e:
        log.warning("practice pace fetch failed: %s", e)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Convenience: assemble everything we need for one race prediction
# ---------------------------------------------------------------------------


def build_race_context(target_race: dict | None = None) -> dict:
    """One-stop call returning every live signal needed by the feature builder."""
    race = target_race or get_next_race()
    if race is None:
        raise RuntimeError("No upcoming race found")

    season = race["season"]
    race_date = dt.date.fromisoformat(race["date"])

    return {
        "race": race,
        "weather": get_weather_forecast(race["lat"], race["lon"], race_date),
        "driver_standings": get_driver_standings(season),
        "constructor_standings": get_constructor_standings(season),
        "season_results": get_recent_results(season),
        "season_qualifying": get_qualifying_results(season),
        "season_sprints": get_sprint_results(season),
        "practice_pace": get_practice_pace(),
        "circuit_history": get_circuit_history(race["circuit_id"]),
        "news": [dataclasses.asdict(n) for n in get_recent_news()],
    }
