"""One-shot diagnostic: fetch the most recent F1 race that has published
results and print the finishing order to stdout.

Used to backtest predictions against reality from a CI environment that has
outbound network access. Falls back across seasons if the current season's
latest race hasn't been published to Jolpica yet (typically 24-48h delay).
"""

from __future__ import annotations

import datetime as dt
import sys

from backend import data_sources


def _last_race_with_results(season: int):
    """Return (race_meta, results_df) for the most recent round with results."""
    results = data_sources.get_recent_results(season)
    if results.empty:
        return None
    schedule = {r["round"]: r for r in data_sources.get_season_schedule(season)}
    rounds_with_results = sorted(results["round"].unique(), reverse=True)
    for rnd in rounds_with_results:
        race_results = results[results["round"] == rnd]
        if not race_results.empty and race_results["position"].notna().any():
            return schedule.get(rnd, {"round": rnd, "race_name": f"Round {rnd}"}), race_results
    return None


def main() -> int:
    today = dt.date.today()
    season = today.year

    found = None
    for try_season in (season, season - 1):
        try:
            found = _last_race_with_results(try_season)
        except Exception as e:
            print(f"WARN: results fetch failed for {try_season}: {e}")
            continue
        if found:
            break

    if not found:
        print("FATAL: no completed races with results found")
        return 1

    race_meta, race_results = found
    race = race_results.sort_values("position", na_position="last")

    print(
        f"=== {race_meta.get('race_name')} "
        f"(Round {race_meta.get('round')}, {race_meta.get('date', '?')}) ==="
    )
    print(f"\nActual finishing order ({len(race)} entries):")
    print(f"{'Pos':<4}  {'Driver':<20}  {'Constructor':<18}  {'Grid':>4}  Status")
    print("-" * 80)
    for _, r in race.iterrows():
        pos = f"P{int(r['position'])}" if r["position"] else "DNF"
        print(
            f"{pos:<4}  {r['driver_id'][:20]:<20}  "
            f"{r['constructor_id'][:18]:<18}  {int(r['grid']):>4}  {r['status']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
