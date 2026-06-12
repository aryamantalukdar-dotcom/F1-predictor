"""Betting-market odds as a prediction signal.

Bookmaker odds are the strongest publicly available predictor of race
outcomes — the market aggregates testing pace, paddock information, and
expert opinion that no free dataset captures. We fetch race-winner outright
odds from The Odds API (free tier: 500 requests/month), average implied
probabilities across bookmakers, remove the vig, and convert to a per-driver
market rank that rank_predictions blends into the model output.

Requires the ODDS_API_KEY env var (free key from https://the-odds-api.com).
Degrades gracefully to "no signal" when the key is missing, the API is
unreachable, or no F1 market is currently listed.
"""

from __future__ import annotations

import logging
import os

from . import data_sources

log = logging.getLogger(__name__)

ODDS_API_BASE = "https://api.the-odds-api.com/v4"


def _discover_f1_sport_key(api_key: str) -> str | None:
    """Find the F1 sport key dynamically — keys can change between seasons.

    The Odds API marks a sport ``active: false`` when it has no live event
    feed at the *moment of the sports-list call*, even if outright markets
    exist. We list candidate F1 keys (active flag ignored) and let the
    events endpoint be the real gate.
    """
    sports = data_sources._http_get(
        f"{ODDS_API_BASE}/sports/", params={"apiKey": api_key, "all": "true"}
    )
    log.info("Odds API: %d total sports returned for this key", len(sports))
    motor_keys = [
        s.get("key", "?")
        for s in sports
        if any(t in (s.get("key", "") + " " + s.get("title", "") + " " + s.get("group", "")).lower()
               for t in ("motor", "racing", "f1", "formula", "nascar", "indycar"))
    ]
    log.info("Odds API motor-related keys (%d): %s", len(motor_keys), motor_keys or "<none>")

    candidates: list[str] = []
    for s in sports:
        text = f"{s.get('key','')} {s.get('title','')} {s.get('group','')}".lower()
        if (
            "formula" in text
            or s.get("key", "").startswith("motorsport_f1")
            or "f1" in s.get("key", "").lower().split("_")
        ):
            candidates.append(s["key"])
    candidates.sort(key=lambda k: (0 if k == "motorsport_f1" else 1, k))
    log.info("Odds API F1 candidate keys: %s", candidates or "none")

    # Direct probe: even if F1 isn't in the sports list (some accounts hide
    # sports they don't currently subscribe to), the canonical key may still
    # be reachable on the events endpoint. Try it before giving up.
    if not candidates:
        for fallback in ("motorsport_f1", "motorsport_formula_one"):
            try:
                evt = data_sources._http_get(
                    f"{ODDS_API_BASE}/sports/{fallback}/odds",
                    params={
                        "apiKey": api_key,
                        "regions": "eu,uk,us",
                        "markets": "outrights",
                        "oddsFormat": "decimal",
                    },
                )
                if isinstance(evt, list):
                    log.info("Odds API direct probe %s → %d events", fallback, len(evt))
                    return fallback
            except Exception as e:
                log.info("Odds API direct probe %s failed: %s", fallback, e)
                continue
        return None
    return candidates[0]


def get_market_probs(driver_standings: list[dict]) -> dict[str, float]:
    """Return {driver_id: implied_win_probability} from bookmaker outrights.

    Empty dict when no key / no market / any failure.
    """
    api_key = os.getenv("ODDS_API_KEY")
    if not api_key:
        log.info("ODDS_API_KEY not set; skipping betting-market signal")
        return {}

    def _fetch():
        sport_key = _discover_f1_sport_key(api_key)
        if not sport_key:
            log.info("No F1 sport key found on The Odds API")
            return {}

        try:
            events = data_sources._http_get(
                f"{ODDS_API_BASE}/sports/{sport_key}/odds",
                params={
                    "apiKey": api_key,
                    "regions": "eu,uk,us",
                    "markets": "outrights",
                    "oddsFormat": "decimal",
                },
            )
        except Exception as e:
            log.warning("Odds events fetch failed (sport=%s): %s", sport_key, e)
            return {}
        if not events:
            log.info("Odds API returned 0 events for %s — no markets posted yet", sport_key)
            return {}
        log.info("Odds API: %d event(s) returned for %s", len(events), sport_key)

        # Aggregate implied probs per outcome name across all bookmakers of
        # the nearest event (the next race's winner market).
        event = events[0]
        sums: dict[str, float] = {}
        counts: dict[str, int] = {}
        for bookmaker in event.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                for outcome in market.get("outcomes", []):
                    price = float(outcome.get("price") or 0)
                    if price <= 1.0:
                        continue
                    name = str(outcome.get("name", "")).strip().lower()
                    sums[name] = sums.get(name, 0.0) + 1.0 / price
                    counts[name] = counts.get(name, 0) + 1
        if not sums:
            return {}
        avg = {name: sums[name] / counts[name] for name in sums}

        # De-vig: normalise so probabilities sum to 1
        total = sum(avg.values())
        probs_by_name = {name: p / total for name, p in avg.items()}

        # Map bookmaker outcome names ("Max Verstappen") to Ergast driver ids
        out: dict[str, float] = {}
        for d in driver_standings:
            family = d["family_name"].lower()
            given = d["given_name"].lower()
            for name, p in probs_by_name.items():
                if family in name and (given[:3] in name or len(family) > 4):
                    out[d["driver_id"]] = p
                    break
        log.info("Betting market: matched %d/%d drivers", len(out), len(driver_standings))
        return out

    try:
        return data_sources._cached("odds:next_race", ttl=1800, fn=_fetch)
    except Exception as e:
        log.warning("Betting odds fetch failed: %s", e)
        return {}


def market_ranks(market_probs: dict[str, float]) -> dict[str, float]:
    """Convert implied probabilities to a 1..N rank (1 = market favourite)."""
    ordered = sorted(market_probs.items(), key=lambda kv: kv[1], reverse=True)
    return {driver_id: float(i + 1) for i, (driver_id, _) in enumerate(ordered)}
