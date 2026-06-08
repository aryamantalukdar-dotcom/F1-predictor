"""One-shot diagnostic: fetch the most recent completed F1 race result and
print it to stdout. Used to backtest predictions against reality from a CI
environment that has outbound network access.

Usage:
    python -m scripts.fetch_last_result
"""

from __future__ import annotations

import datetime as dt
import sys

from backend import data_sources


def main() -> int:
    today = dt.date.today()
    season = today.year

    # Find the most recently completed race
    try:
        schedule = data_sources.get_season_schedule(season)
    except Exception as e:
        print(f"FATAL: schedule fetch failed: {e}")
        return 1

    past = [r for r in schedule if dt.date.fromisoformat(r["date"]) < today]
    if not past:
        # Fall back to previous season's last race
        try:
            schedule = data_sources.get_season_schedule(season - 1)
            past = list(schedule)
        except Exception as e:
            print(f"FATAL: no completed races found: {e}")
            return 1

    last = past[-1]
    print(f"=== Last completed race: {last['race_name']} (Round {last['round']}, {last['date']}) ===")

    results = data_sources.get_recent_results(last["season"])
    race = results[results["round"] == last["round"]].sort_values("position", na_position="last")

    print(f"\nActual finishing order ({len(race)} entries):")
    print(f"{'Pos':<4}  {'Code':<5}  {'Driver':<24}  {'Team':<22}  {'Grid':>4}  Status")
    print("-" * 90)
    for _, r in race.iterrows():
        pos = f"P{int(r['position'])}" if r["position"] else "DNF"
        print(
            f"{pos:<4}  {r['driver_id'][:5]:<5}  {r['driver_id'][:24]:<24}  "
            f"{r['constructor_id'][:22]:<22}  {int(r['grid']):>4}  {r['status']}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
