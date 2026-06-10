"""Feature engineering for race + qualifying predictions.

Each feature row represents a (driver, race) pairing. Features blend:
  - recent form (rolling avg finish, DNF rate, momentum)
  - track-specific history at the same circuit
  - constructor strength (current championship position + recent results)
  - weather forecast for race weekend
  - track-type signals (street circuit, high-downforce, etc.)
  - news-derived adjustments produced by the Claude analyzer

The shape returned here is consumed by both the training pipeline and the
live prediction path, so they stay in lockstep.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Hand-curated track type tags — these are slow-moving and worth hardcoding
# rather than scraping. Used as categorical features.
STREET_CIRCUITS = {"monaco", "baku", "singapore", "jeddah", "miami", "vegas", "albert_park"}
HIGH_DOWNFORCE = {"monaco", "hungaroring", "singapore", "zandvoort", "marina_bay"}
POWER_CIRCUITS = {"monza", "spa", "baku", "jeddah", "vegas"}


@dataclass
class FeatureRow:
    """Schema for one driver-at-race feature vector."""

    driver_id: str
    constructor_id: str
    season: int
    round: int
    circuit_id: str

    # Current championship state
    driver_points: float
    driver_position: int
    constructor_points: float
    constructor_position: int

    # Recent form (last N races)
    avg_finish_last3: float
    avg_finish_last5: float
    dnf_rate_last5: float
    momentum: float  # negative slope of finish positions = improving form
    avg_grid_last5: float

    # Track-specific history
    track_avg_finish: float
    track_best_finish: float
    track_starts: int

    # Weather
    rain_probability: float
    temperature_c: float
    wind_kph: float

    # Track type (one-hot-ish)
    is_street: int
    is_high_downforce: int
    is_power_circuit: int

    # News signal (filled in later by news_analyzer)
    news_factor: float = 0.0

    # Optional: qualifying result, used for race model when available
    qual_position: float = np.nan

    # Optional: rank (1 = fastest) across all practice sessions for this race
    # weekend. NaN before any practice has run.
    practice_rank: float = np.nan


def _avg_finish(rows: pd.DataFrame, n: int) -> float:
    if rows.empty:
        return 12.0  # neutral midfield prior
    last = rows.sort_values("round").tail(n)
    pos = last["position"].dropna()
    if pos.empty:
        return 18.0  # all DNFs in window
    return float(pos.mean())


def _dnf_rate(rows: pd.DataFrame, n: int) -> float:
    if rows.empty:
        return 0.0
    last = rows.sort_values("round").tail(n)
    if len(last) == 0:
        return 0.0
    return float(last["position"].isna().sum()) / len(last)


def _momentum(rows: pd.DataFrame, n: int = 5) -> float:
    """Slope of position vs round, negated so positive = improving."""
    if rows.empty:
        return 0.0
    last = rows.sort_values("round").tail(n).dropna(subset=["position"])
    if len(last) < 2:
        return 0.0
    x = last["round"].to_numpy(dtype=float)
    y = last["position"].to_numpy(dtype=float)
    slope = np.polyfit(x, y, 1)[0]
    return float(-slope)


def _track_history(circuit_history: pd.DataFrame, driver_id: str) -> tuple[float, float, int]:
    if circuit_history.empty:
        return 12.0, 12.0, 0
    rows = circuit_history[circuit_history["driver_id"] == driver_id]
    pos = rows["position"].dropna()
    if pos.empty:
        return 12.0, 12.0, len(rows)
    return float(pos.mean()), float(pos.min()), len(rows)


def build_feature_frame(context: dict, news_factors: dict[str, float] | None = None) -> pd.DataFrame:
    """Build feature rows for every driver in the upcoming race.

    `context` is the dict returned by data_sources.build_race_context().
    `news_factors` maps driver_id -> [-1, +1] signed adjustment from the LLM.
    """
    race = context["race"]
    weather = context["weather"]
    standings = context["driver_standings"]
    cstandings = {c["constructor_id"]: c for c in context["constructor_standings"]}
    season_results = context["season_results"]
    season_qual = context["season_qualifying"]
    season_sprints = context.get("season_sprints", pd.DataFrame())
    practice_pace = context.get("practice_pace", pd.DataFrame())
    circuit_history = context["circuit_history"]
    news_factors = news_factors or {}

    # Build a {driver_code: practice_rank} from OpenF1 practice pace.
    # Rank 1 = fastest best-lap. Drivers without practice data stay NaN and
    # the heuristic treats that as "no signal yet" (Friday morning predictions).
    practice_rank_by_code: dict[str, float] = {}
    if not practice_pace.empty and "best_lap_s" in practice_pace.columns:
        sorted_pace = practice_pace.sort_values("best_lap_s").reset_index(drop=True)
        for i, row in sorted_pace.iterrows():
            practice_rank_by_code[str(row["driver_code"]).upper()] = float(i + 1)

    circuit_id = race["circuit_id"]
    is_street = int(circuit_id in STREET_CIRCUITS)
    is_hd = int(circuit_id in HIGH_DOWNFORCE)
    is_power = int(circuit_id in POWER_CIRCUITS)

    rows: list[FeatureRow] = []
    for d in standings:
        driver_id = d["driver_id"]
        cid = d["constructor_id"]
        cs = cstandings.get(cid, {"points": 0.0, "position": 10})

        d_results = season_results[season_results["driver_id"] == driver_id] if not season_results.empty else pd.DataFrame()
        d_qual = season_qual[season_qual["driver_id"] == driver_id] if not season_qual.empty else pd.DataFrame()

        avg3 = _avg_finish(d_results, 3)
        avg5 = _avg_finish(d_results, 5)
        dnf5 = _dnf_rate(d_results, 5)
        mom = _momentum(d_results, 5)
        avg_grid = float(d_results.tail(5)["grid"].mean()) if not d_results.empty else 12.0

        track_avg, track_best, track_starts = _track_history(circuit_history, driver_id)

        # Latest qualifying we have for this driver, if the qual session has run
        latest_qual = (
            d_qual[d_qual["round"] == race["round"]]["qual_position"].iloc[0]
            if not d_qual.empty and (d_qual["round"] == race["round"]).any()
            else np.nan
        )

        # Sprint result is a strong proxy for race pace when main quali hasn't
        # happened yet (Friday predictions on a sprint weekend). Use sprint
        # qualifying position if available, else sprint race finish position.
        if pd.isna(latest_qual) and not season_sprints.empty:
            sp = season_sprints[
                (season_sprints["round"] == race["round"])
                & (season_sprints["driver_id"] == driver_id)
            ]
            if not sp.empty:
                grid = sp["sprint_grid"].iloc[0]
                pos = sp["sprint_position"].iloc[0]
                latest_qual = float(grid) if pd.notna(grid) and grid else float(pos)

        rows.append(
            FeatureRow(
                driver_id=driver_id,
                constructor_id=cid,
                season=race["season"],
                round=race["round"],
                circuit_id=circuit_id,
                driver_points=d["points"],
                driver_position=d["position"],
                constructor_points=cs["points"],
                constructor_position=cs["position"],
                avg_finish_last3=avg3,
                avg_finish_last5=avg5,
                dnf_rate_last5=dnf5,
                momentum=mom,
                avg_grid_last5=avg_grid,
                track_avg_finish=track_avg,
                track_best_finish=track_best,
                track_starts=track_starts,
                rain_probability=weather["rain_probability"],
                temperature_c=weather["temperature_c"],
                wind_kph=weather["wind_kph"],
                is_street=is_street,
                is_high_downforce=is_hd,
                is_power_circuit=is_power,
                news_factor=float(news_factors.get(driver_id, 0.0)),
                qual_position=float(latest_qual) if pd.notna(latest_qual) else np.nan,
                practice_rank=practice_rank_by_code.get(d.get("code", "").upper(), np.nan),
            )
        )

    df = pd.DataFrame([r.__dict__ for r in rows])
    return df


# ---------------------------------------------------------------------------
# Training-side feature builder: assembles many historical (driver, race) rows
# ---------------------------------------------------------------------------


def build_training_frame(seasons: list[int], data_sources_module) -> pd.DataFrame:
    """Walk historical seasons round-by-round, building features as they were
    visible at race time. Used only by train.py.

    Note: this is a lightweight reconstruction — real strict point-in-time
    rebuilding would require capturing standings *before* each race. We
    approximate by computing rolling features from prior rounds only.
    """
    frames = []
    for season in seasons:
        try:
            schedule = data_sources_module.get_season_schedule(season)
            results = data_sources_module.get_recent_results(season)
            qual = data_sources_module.get_qualifying_results(season)
        except Exception as e:
            log.warning("training data unavailable for %s: %s", season, e)
            continue
        if results.empty:
            log.warning("no results returned for season %s; skipping", season)
            continue
        log.info("season %s: %d result rows across %d races", season, len(results), results["round"].nunique())

        for race in schedule:
            rnd = race["round"]
            circuit_id = race["circuit_id"]
            race_results = results[results["round"] == rnd]
            if race_results.empty:
                continue
            prior = results[results["round"] < rnd]

            # Point-in-time championship standings reconstructed from prior
            # rounds, so driver/constructor strength are live training signals
            # instead of constant zeros the model learns to ignore.
            if not prior.empty:
                d_points = prior.groupby("driver_id")["points"].sum()
                d_pos = d_points.rank(ascending=False, method="min")
                c_points = prior.groupby("constructor_id")["points"].sum()
                c_pos = c_points.rank(ascending=False, method="min")
            else:
                d_points = d_pos = c_points = c_pos = pd.Series(dtype=float)

            for _, r in race_results.iterrows():
                did = r["driver_id"]
                cid = r["constructor_id"]
                d_prior = prior[prior["driver_id"] == did]
                avg3 = _avg_finish(d_prior, 3)
                avg5 = _avg_finish(d_prior, 5)
                dnf5 = _dnf_rate(d_prior, 5)
                mom = _momentum(d_prior, 5)
                avg_grid = float(d_prior.tail(5)["grid"].mean()) if not d_prior.empty else 12.0

                track_rows = prior[(prior["driver_id"] == did) & (prior["circuit_id"] == circuit_id)]
                track_avg = float(track_rows["position"].dropna().mean()) if not track_rows.empty else 12.0
                track_best = float(track_rows["position"].dropna().min()) if not track_rows.empty else 12.0

                q = qual[(qual["round"] == rnd) & (qual["driver_id"] == did)]
                qpos = float(q.iloc[0]["qual_position"]) if not q.empty else np.nan

                frames.append(
                    {
                        "driver_id": did,
                        "constructor_id": cid,
                        "season": season,
                        "round": rnd,
                        "circuit_id": circuit_id,
                        "driver_points": float(d_points.get(did, 0.0)),
                        "driver_position": float(d_pos.get(did, 12.0)),
                        "constructor_points": float(c_points.get(cid, 0.0)),
                        "constructor_position": float(c_pos.get(cid, 6.0)),
                        "avg_finish_last3": avg3,
                        "avg_finish_last5": avg5,
                        "dnf_rate_last5": dnf5,
                        "momentum": mom,
                        "avg_grid_last5": avg_grid,
                        "track_avg_finish": track_avg,
                        "track_best_finish": track_best,
                        "track_starts": len(track_rows),
                        "is_street": int(circuit_id in STREET_CIRCUITS),
                        "is_high_downforce": int(circuit_id in HIGH_DOWNFORCE),
                        "is_power_circuit": int(circuit_id in POWER_CIRCUITS),
                        "qual_position": qpos,
                        # Targets
                        "race_position": r["position"],
                        "grid_position": r["grid"],
                    }
                )

    return pd.DataFrame(frames)


def feature_columns(include_qual: bool = True) -> list[str]:
    """Model feature list, shared by training and inference.

    Weather and news_factor are intentionally NOT here: they can't be
    reconstructed historically, so as trained features they'd be constant
    zeros that LightGBM ignores. They act as post-hoc adjustments in
    models.rank_predictions / the heuristic instead.
    """
    cols = [
        "avg_finish_last3",
        "avg_finish_last5",
        "dnf_rate_last5",
        "momentum",
        "avg_grid_last5",
        "track_avg_finish",
        "track_best_finish",
        "track_starts",
        "is_street",
        "is_high_downforce",
        "is_power_circuit",
        "driver_points",
        "driver_position",
        "constructor_points",
        "constructor_position",
    ]
    if include_qual:
        cols.append("qual_position")
    return cols
