"""End-to-end prediction pipeline.

Pulls live data, runs news analysis, builds features, runs both models, and
returns a structured payload ready for the API or CLI to render.

Designed to be called for any specific race (defaults to the next upcoming).
"""

from __future__ import annotations

import logging
import os
from typing import Any

import pandas as pd

from . import data_sources, features, models, news_analyzer

log = logging.getLogger(__name__)


def _driver_meta(standings: list[dict]) -> dict[str, dict]:
    return {
        d["driver_id"]: {
            "name": f"{d['given_name']} {d['family_name']}",
            "code": d["code"],
            "constructor": d["constructor_name"],
        }
        for d in standings
    }


def _heuristic_predictor(feature_df: pd.DataFrame, driver_meta: dict[str, dict]) -> list[dict]:
    """Fallback predictor used when no trained model is available.

    Combines a few hand-weighted signals into an expected position. This is
    intentionally simple but produces a sane ranking on the very first run
    before anyone has trained the real models.
    """
    scores = (
        0.45 * feature_df["avg_finish_last5"].fillna(12)
        + 0.20 * feature_df["track_avg_finish"].fillna(12)
        + 0.10 * feature_df["constructor_position"].fillna(10)
        + 0.05 * feature_df["driver_position"].fillna(10)
        + 0.20 * feature_df.get("qual_position", pd.Series([12.0] * len(feature_df))).fillna(12)
        - 1.5 * feature_df["news_factor"].fillna(0)
        - 0.6 * feature_df["momentum"].fillna(0)
        + 3.0 * feature_df["dnf_rate_last5"].fillna(0)
    )

    order = scores.argsort()
    ranks = order.argsort().to_numpy(dtype=float)
    import numpy as np

    win_logits = -ranks * 1.2
    win_probs = np.exp(win_logits) / np.exp(win_logits).sum()

    out = []
    for i, idx in enumerate(order):
        row = feature_df.iloc[idx]
        did = row["driver_id"]
        meta = driver_meta.get(did, {})
        out.append(
            {
                "rank": int(i + 1),
                "driver_id": did,
                "driver_name": meta.get("name", did),
                "constructor": meta.get("constructor", row["constructor_id"]),
                "code": meta.get("code", did[:3].upper()),
                "predicted_position": float(scores.iloc[idx]),
                "raw_model_position": float(scores.iloc[idx]),
                "news_factor": float(row["news_factor"]),
                "win_probability": float(win_probs[idx]),
            }
        )
    return out


def _heuristic_pole(feature_df: pd.DataFrame, driver_meta: dict[str, dict]) -> list[dict]:
    """Pole-specific heuristic — leans more on raw pace signals."""
    scores = (
        0.40 * feature_df["avg_grid_last5"].fillna(12)
        + 0.25 * feature_df["track_avg_finish"].fillna(12)
        + 0.20 * feature_df["constructor_position"].fillna(10)
        + 0.10 * feature_df["driver_position"].fillna(10)
        - 0.8 * feature_df["news_factor"].fillna(0)
    )
    df_with_score = feature_df.assign(__score__=scores)
    df_sorted = df_with_score.sort_values("__score__").reset_index(drop=True)

    out = []
    for i, row in df_sorted.iterrows():
        did = row["driver_id"]
        meta = driver_meta.get(did, {})
        out.append(
            {
                "rank": int(i + 1),
                "driver_id": did,
                "driver_name": meta.get("name", did),
                "constructor": meta.get("constructor", row["constructor_id"]),
                "code": meta.get("code", did[:3].upper()),
                "predicted_position": float(row["__score__"]),
                "raw_model_position": float(row["__score__"]),
                "news_factor": float(row["news_factor"]),
            }
        )
    return out


def predict_next_race(use_models: bool = True) -> dict[str, Any]:
    """Build the full prediction payload for the next upcoming race."""
    log.info("Fetching live race context")
    context = data_sources.build_race_context()

    log.info("Running news analysis via Claude")
    analysis = news_analyzer.analyze_news(
        news=context["news"],
        driver_standings=context["driver_standings"],
        race=context["race"],
    )
    factors = news_analyzer.factors_dict(analysis)

    log.info("Building feature frame")
    feature_df = features.build_feature_frame(context, news_factors=factors)
    driver_meta = _driver_meta(context["driver_standings"])

    # ---- Race finish prediction ----
    race_predictions: list[dict]
    if use_models and os.path.exists(models.race_model_path()):
        log.info("Loading trained race model")
        race_model = models.RacePredictor()
        race_model.load(models.race_model_path())
        race_predictions = models.rank_predictions(feature_df, race_model, driver_meta)
        race_model_used = "lightgbm"
    else:
        log.info("No trained race model found; using heuristic fallback")
        race_predictions = _heuristic_predictor(feature_df, driver_meta)
        race_model_used = "heuristic"

    # ---- Pole prediction ----
    pole_predictions: list[dict]
    if use_models and os.path.exists(models.pole_model_path()):
        log.info("Loading trained pole model")
        pole_model = models.PolePredictor()
        pole_model.load(models.pole_model_path())
        pole_predictions = models.rank_predictions(feature_df, pole_model, driver_meta)
        pole_model_used = "lightgbm"
    else:
        log.info("No trained pole model found; using heuristic fallback")
        pole_predictions = _heuristic_pole(feature_df, driver_meta)
        pole_model_used = "heuristic"

    pole = pole_predictions[0] if pole_predictions else None

    return {
        "race": context["race"],
        "weather": context["weather"],
        "race_predictions": race_predictions,
        "pole_prediction": pole,
        "pole_predictions": pole_predictions[:5],
        "news": {
            "narrative": analysis.get("race_narrative", ""),
            "storylines": analysis.get("key_storylines", []),
            "items": context["news"][:10],
            "driver_factors": analysis.get("driver_factors", []),
        },
        "meta": {
            "race_model": race_model_used,
            "pole_model": pole_model_used,
            "n_drivers": len(feature_df),
        },
    }
