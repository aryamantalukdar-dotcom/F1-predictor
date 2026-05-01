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
    parser = argparse.ArgumentParser(description="Train F1 prediction models")
    parser.add_argument(
        "--seasons", type=int, nargs="+", default=[2021, 2022, 2023, 2024, 2025],
        help="Seasons to use for training",
    )
    args = parser.parse_args()

    log.info("Building training frame from seasons: %s", args.seasons)
    df = features.build_training_frame(args.seasons, data_sources)
    if df.empty:
        log.error("No training data assembled — check network/API access")
        return 1
    log.info("Training frame: %d rows across %d races", len(df), df.groupby(["season", "round"]).ngroups)

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
