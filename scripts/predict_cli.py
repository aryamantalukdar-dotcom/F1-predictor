"""CLI: print the next-race prediction as a readable table.

Usage:
    python -m scripts.predict_cli
"""

from __future__ import annotations

import logging
import sys

from backend import predict


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = predict.predict_next_race()

    race = p["race"]
    weather = p["weather"]
    print(f"\n=== {race['race_name']} — {race['date']} ===")
    print(f"  Circuit: {race['circuit_name']} ({race['locality']}, {race['country']})")
    print(
        f"  Weather: {weather['temperature_c']:.1f}°C, "
        f"{int(weather['rain_probability']*100)}% rain, {weather['wind_kph']:.0f} kph wind"
    )
    print(f"  Models: race={p['meta']['race_model']}, pole={p['meta']['pole_model']}")

    pole = p["pole_prediction"]
    if pole:
        print(f"\n  POLE PREDICTION → P1: {pole['driver_name']} ({pole['constructor']})")

    print("\n  PREDICTED FINISHING ORDER:")
    print("  Pos  Code  Driver                    Team                      Win%   News")
    print("  " + "-" * 78)
    for d in p["race_predictions"]:
        print(
            f"  P{d['rank']:<3} {d['code']:<5} {d['driver_name'][:24]:<24}  "
            f"{d['constructor'][:24]:<24}  {d['win_probability']*100:>5.1f}%  "
            f"{d['news_factor']:+.2f}"
        )

    if p["news"].get("narrative"):
        print(f"\n  NARRATIVE: {p['news']['narrative']}")
    if p["news"].get("storylines"):
        print("\n  STORYLINES:")
        for s in p["news"]["storylines"]:
            print(f"    - {s}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
