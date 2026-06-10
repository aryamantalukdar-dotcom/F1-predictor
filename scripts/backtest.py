"""Walk-forward backtest of the Thursday (pre-weekend) prediction snapshot.

For every race in the evaluation window this script:
  1. Trains race + pole models ONLY on races that finished before it
     (honest out-of-sample, with the same recency weighting as production).
  2. Predicts qualifying with no weekend data (Thursday snapshot).
  3. Feeds the predicted grid into the race model (two-stage, same as prod).
  4. Scores against the actual result.

Metrics (chosen to match the stated goal — top-10 order + pole):
  - pole_hit:        predicted pole == actual pole-sitter
  - winner_hit:      predicted P1 == actual race winner
  - podium_overlap:  |predicted top3 ∩ actual top3| / 3
  - top10_spearman:  rank correlation over the drivers who actually
                     finished in the top 10
  - top10_mae:       mean |predicted rank - actual position| for actual
                     top-10 finishers

A naive baseline (order by avg finish over the last 5 races) is scored on
the same races so we can tell whether the model earns its complexity.

Usage:
    python -m scripts.backtest --train-seasons 2022 2023 2024 2025 2026 --eval-seasons 2025 2026
"""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from backend import data_sources, features, models

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger("backtest")

AGE_WEIGHTS = {0: 4.0, 1: 2.0, 2: 1.0, 3: 0.6, 4: 0.4}


def _weights(df: pd.DataFrame) -> pd.Series:
    max_season = int(df["season"].max())
    return (max_season - df["season"]).map(lambda a: AGE_WEIGHTS.get(int(a), 0.3))


def _score_race(pred_rank: dict[str, int], race_rows: pd.DataFrame) -> dict | None:
    actual = race_rows.dropna(subset=["race_position"]).set_index("driver_id")["race_position"]
    if actual.empty:
        return None
    top10 = actual[actual <= 10]
    common = [d for d in top10.index if d in pred_rank]
    if len(common) < 5:
        return None

    pred = np.array([pred_rank[d] for d in common], dtype=float)
    act = np.array([top10[d] for d in common], dtype=float)
    rho = spearmanr(pred, act).statistic if len(common) >= 3 else np.nan

    actual_winner = actual.idxmin()
    pred_winner = min(pred_rank, key=pred_rank.get)
    actual_podium = set(actual.nsmallest(3).index)
    pred_podium = {d for d, r in sorted(pred_rank.items(), key=lambda kv: kv[1])[:3]}

    return {
        "winner_hit": float(pred_winner == actual_winner),
        "podium_overlap": len(actual_podium & pred_podium) / 3.0,
        "top10_spearman": float(rho),
        "top10_mae": float(np.abs(pred - act).mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-seasons", type=int, nargs="+", default=[2022, 2023, 2024, 2025, 2026])
    parser.add_argument("--eval-seasons", type=int, nargs="+", default=[2025, 2026])
    args = parser.parse_args()

    print(f"Building frame for seasons {args.train_seasons} ...")
    frame = features.build_training_frame(args.train_seasons, data_sources)
    if frame.empty:
        print("FATAL: no data")
        return 1
    frame["news_factor"] = 0.0  # Thursday snapshot: no news/practice/odds
    frame = frame.sort_values(["season", "round"]).reset_index(drop=True)

    races = (
        frame[frame["season"].isin(args.eval_seasons)][["season", "round"]]
        .drop_duplicates()
        .itertuples(index=False)
    )

    model_scores, base_scores, pole_hits = [], [], []
    n_eval = 0

    for season, rnd in races:
        train_df = frame[(frame["season"] < season) | ((frame["season"] == season) & (frame["round"] < rnd))].copy()
        race_rows = frame[(frame["season"] == season) & (frame["round"] == rnd)].copy()
        if len(train_df) < 300 or race_rows.empty:
            continue
        train_df["_weight"] = _weights(train_df)

        try:
            pole_model = models.PolePredictor()
            pole_model.fit(train_df)
            race_model = models.RacePredictor()
            race_model.fit(train_df)
        except Exception as e:
            log.warning("train failed for %s R%s: %s", season, rnd, e)
            continue

        # --- Thursday snapshot: hide actual quali, predict it ---
        snapshot = race_rows.copy()
        actual_quali = snapshot.set_index("driver_id")["qual_position"]
        snapshot["qual_position"] = np.nan

        pole_preds = models.rank_predictions(snapshot, pole_model, {}, dnf_aware=False)
        pole_rank = {p["driver_id"]: p["rank"] for p in pole_preds}

        # Pole hit
        actual_pole = actual_quali.dropna()
        if not actual_pole.empty:
            actual_pole_driver = actual_pole.idxmin()
            pred_pole_driver = min(pole_rank, key=pole_rank.get)
            pole_hits.append(float(pred_pole_driver == actual_pole_driver))

        # Two-stage race prediction
        snapshot["qual_position"] = snapshot["driver_id"].map({d: float(r) for d, r in pole_rank.items()})
        race_preds = models.rank_predictions(snapshot, race_model, {})
        pred_rank = {p["driver_id"]: p["rank"] for p in race_preds}

        s = _score_race(pred_rank, race_rows)
        if s:
            model_scores.append(s)
            n_eval += 1

        # Baseline: order by recent form only
        base_order = race_rows.sort_values("avg_finish_last5")["driver_id"].tolist()
        base_rank = {d: i + 1 for i, d in enumerate(base_order)}
        b = _score_race(base_rank, race_rows)
        if b:
            base_scores.append(b)

    if not model_scores:
        print("No races evaluated — not enough data.")
        return 1

    def _agg(scores: list[dict]) -> dict:
        keys = scores[0].keys()
        return {k: float(np.nanmean([s[k] for s in scores])) for k in keys}

    m, b = _agg(model_scores), _agg(base_scores)
    print(f"\n=== Backtest: {n_eval} races, Thursday snapshot ===")
    print(f"{'metric':<18} {'model':>8} {'baseline':>9}")
    print("-" * 38)
    for k in m:
        print(f"{k:<18} {m[k]:>8.3f} {b.get(k, float('nan')):>9.3f}")
    if pole_hits:
        print(f"{'pole_hit':<18} {np.mean(pole_hits):>8.3f} {'—':>9}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
