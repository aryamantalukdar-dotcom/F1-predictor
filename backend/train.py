"""Train the race + pole LightGBM models on historical seasons.

Usage:
    python -m backend.train --seasons 2021 2022 2023 2024 2025

Pulls historical results via Jolpica/Ergast, builds rolling features, trains
both predictors, and dumps them to models_cache/.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from . import data_sources, features, models

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)


def main() -> int:
    import datetime as dt

    current_year = dt.date.today().year
    default_seasons = list(range(current_year - 4, current_year + 1))

    parser = argparse.ArgumentParser(description="Train F1 prediction models")
    parser.add_argument(
        "--seasons", type=int, nargs="+", default=default_seasons,
        help="Seasons to use for training (default: last 5 incl. current)",
    )
    args = parser.parse_args()

    log.info("Building training frame from seasons: %s", args.seasons)
    df = features.build_training_frame(args.seasons, data_sources)
    if df.empty:
        log.error("No training data assembled — check network/API access")
        return 1

    # Recency weighting: the current season dominates. A regulation reset
    # (like 2026) reshuffles the competitive order, so pre-reset history is
    # context, not gospel.
    max_season = int(df["season"].max())
    age_weights = {0: 4.0, 1: 2.0, 2: 1.0, 3: 0.6, 4: 0.4}
    df["_weight"] = (max_season - df["season"]).map(lambda a: age_weights.get(int(a), 0.3))
    log.info(
        "Training frame: %d rows across %d races (season weights: %s)",
        len(df),
        df.groupby(["season", "round"]).ngroups,
        {int(s): age_weights.get(max_season - int(s), 0.3) for s in sorted(df["season"].unique())},
    )

    os.makedirs(models.models_dir(), exist_ok=True)

    log.info("Training race model...")
    race = models.RacePredictor()
    race_result = race.fit(df)
    race.save(models.race_model_path())
    log.info("Race model saved → %s (val MAE=%.3f)", models.race_model_path(), race_result.mae)

    log.info("Training pole model...")
    pole = models.PolePredictor()
    pole_result = pole.fit(df)
    pole.save(models.pole_model_path())
    log.info("Pole model saved → %s (val MAE=%.3f)", models.pole_model_path(), pole_result.mae)

    return 0


if __name__ == "__main__":
    sys.exit(main())
