"""LightGBM-based predictors for race finishing order and pole position.

Two separate models:
  - RacePredictor:    predicts expected race finishing position (regression)
  - PolePredictor:    predicts expected qualifying position (regression)

Both use LightGBM regression on integer position with monotone constraints
where they make physical sense (e.g. higher constructor points → better
expected finish). Predictions are then ranked to produce the ordering.

Also exposes a softmax-over-rank to surface "win probability" estimates,
which read better in the UI than raw position predictions.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from . import features as feat

log = logging.getLogger(__name__)


def _lgb_params(seed: int = 42) -> dict:
    return {
        "objective": "regression",
        "metric": "mae",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 10,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 5,
        "lambda_l2": 0.1,
        "verbose": -1,
        "seed": seed,
    }


@dataclass
class TrainResult:
    mae: float
    n_train: int
    n_val: int


class _BaseLGBM:
    target_col: str = ""
    include_qual: bool = True

    def __init__(self):
        self.model: lgb.Booster | None = None
        self.feature_cols = feat.feature_columns(include_qual=self.include_qual)

    def _xy(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        # Target may be missing for DNFs; for training purposes treat DNF as
        # a worst-case finish so the model learns to penalize unreliable cars.
        df = df.copy()
        df[self.target_col] = df[self.target_col].fillna(20)
        # Defensive: any feature column missing from the frame becomes 0
        for c in self.feature_cols:
            if c not in df.columns:
                df[c] = 0.0
        X = df[self.feature_cols]
        y = df[self.target_col].astype(float)
        return X, y

    def fit(self, df: pd.DataFrame, val_frac: float = 0.2) -> TrainResult:
        if df.empty:
            raise ValueError("Empty training frame")
        df = df.dropna(subset=[self.target_col]).copy()
        # Sort by season + round so the validation split is the most recent races
        df = df.sort_values(["season", "round"]).reset_index(drop=True)
        cut = int(len(df) * (1 - val_frac))
        train_df, val_df = df.iloc[:cut], df.iloc[cut:]

        X_tr, y_tr = self._xy(train_df)
        X_vl, y_vl = self._xy(val_df)

        # Optional recency weighting: train.py attaches a `_weight` column so
        # current-season races dominate over pre-regulation-change history.
        w_tr = train_df["_weight"].to_numpy() if "_weight" in train_df.columns else None
        w_vl = val_df["_weight"].to_numpy() if "_weight" in val_df.columns else None

        dtrain = lgb.Dataset(X_tr, y_tr, weight=w_tr)
        dval = lgb.Dataset(X_vl, y_vl, reference=dtrain, weight=w_vl)

        self.model = lgb.train(
            _lgb_params(),
            dtrain,
            num_boost_round=2000,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )

        preds = self.model.predict(X_vl)
        mae = float(np.mean(np.abs(preds - y_vl.to_numpy())))
        log.info("%s trained: n=%d, val_mae=%.3f", self.__class__.__name__, len(train_df), mae)
        return TrainResult(mae=mae, n_train=len(train_df), n_val=len(val_df))

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not trained")
        X, _ = self._xy(df.assign(**{self.target_col: 0.0}))
        return self.model.predict(X)

    def save(self, path: str) -> None:
        if self.model is None:
            raise RuntimeError("Nothing to save")
        joblib.dump(
            {"booster_text": self.model.model_to_string(), "feature_cols": self.feature_cols},
            path,
        )

    def load(self, path: str) -> None:
        blob = joblib.load(path)
        self.model = lgb.Booster(model_str=blob["booster_text"])
        self.feature_cols = blob["feature_cols"]


class RacePredictor(_BaseLGBM):
    target_col = "race_position"
    include_qual = True


class PolePredictor(_BaseLGBM):
    """Predicts qualifying position. We don't use qual_position itself as input.

    Target is the qualifying classification, NOT race grid: grid includes
    penalties and pit-lane starts (grid=0 in Ergast data, which a regression
    would learn as "better than pole").
    """

    target_col = "qual_position"
    include_qual = False


# ---------------------------------------------------------------------------
# Inference helpers — convert raw predictions into ranked output + probabilities
# ---------------------------------------------------------------------------


# Weight pulling the race prediction toward a plain recent-form ordering.
# Walk-forward backtesting showed avg_finish_last5 alone outperforming the
# model on the Thursday snapshot (Spearman 0.70 vs 0.67), so the ensemble
# anchors on form and lets the model adjust at the margins.
FORM_ANCHOR = 0.5


def rank_predictions(
    feature_df: pd.DataFrame,
    predictor: _BaseLGBM,
    driver_meta: dict[str, dict],
    dnf_aware: bool = True,
    form_anchor: float = FORM_ANCHOR,
) -> list[dict]:
    """Run the predictor and return a ranked list of {driver_id, predicted_position, win_prob, ...}.

    Three post-hoc adjustments on top of the raw LightGBM output:
      1. News factor: shift each driver's predicted position by -1.5 * factor
         (Claude-derived signed [-1, 1] sentiment).
      2. Practice pace: when OpenF1 has lap times from the current weekend's
         practice sessions, blend the practice rank in at weight 0.45. This
         is the strongest real-world short-term pace signal.
      3. Rookie/sparse-driver shrinkage: drivers with very few starts at
         this circuit (track_starts < 2) get pulled 40% toward their team's
         median predicted position. Counteracts rookie blowup like a
         four-race-old driver suddenly leading at 70%.

    Softmax temperature lowered from 1.2 to 0.55 — the old value produced
    a 70% top-driver probability which is empirically too peaked.
    """
    raw = predictor.predict(feature_df)
    nudged = raw - 1.5 * feature_df["news_factor"].to_numpy()

    # 2) Practice-pace nudge
    if "practice_rank" in feature_df.columns:
        pr = feature_df["practice_rank"].to_numpy(dtype=float)
        mask = ~np.isnan(pr)
        if mask.any():
            nudged[mask] = 0.55 * nudged[mask] + 0.45 * pr[mask]

    # 3) Bayesian shrinkage toward team median for sparse-history drivers
    if "track_starts" in feature_df.columns and "constructor_id" in feature_df.columns:
        df = feature_df.copy()
        df["_pred"] = nudged
        team_medians = df.groupby("constructor_id")["_pred"].transform("median").to_numpy()
        starts = feature_df["track_starts"].fillna(0).to_numpy(dtype=float)
        shrink = np.where(starts < 2, 0.40, 0.0)
        nudged = (1 - shrink) * nudged + shrink * team_medians

    # 4) Championship skill floor.
    # Canada GP backtest: model predicted Verstappen P10 with 0% win prob;
    # actual result was P3. A top-5 championship driver shouldn't be
    # predicted to finish far down based on rolling form alone. Pull the
    # prediction toward their championship position by 30%, with stronger
    # pull at the top of the table. DNF-prone drivers (>40% recent DNF
    # rate) skip this — the form penalty there is legitimate.
    if "driver_position" in feature_df.columns:
        champ_pos = feature_df["driver_position"].fillna(15).to_numpy(dtype=float)
        dnf_rate = feature_df.get("dnf_rate_last5", pd.Series([0.0] * len(feature_df))).fillna(0).to_numpy()
        # Blend weight: 0.35 for championship P1-P3, fading to 0 by P10.
        blend = np.clip(0.40 - 0.04 * champ_pos, 0.0, 0.35)
        # Skip the floor entirely for drivers whose reliability is genuinely bad
        blend = np.where(dnf_rate > 0.4, 0.0, blend)
        nudged = (1 - blend) * nudged + blend * champ_pos

    # 5) Betting-market blend. When odds are available, pull toward the
    # market's implied ranking — bookmakers aggregate information (testing
    # pace, paddock chatter, insider sentiment) no public dataset has.
    if "odds_rank" in feature_df.columns:
        orank = feature_df["odds_rank"].to_numpy(dtype=float)
        omask = ~np.isnan(orank)
        if omask.any():
            nudged[omask] = 0.65 * nudged[omask] + 0.35 * orank[omask]

    # 6) Recent-form anchor (race path only — quali has its own dynamics).
    # See FORM_ANCHOR note above: the baseline ordering is too strong to
    # override wholesale.
    if dnf_aware and form_anchor > 0 and "avg_finish_last5" in feature_df.columns:
        form_rank = (
            feature_df["avg_finish_last5"].fillna(12.0).rank(method="first").to_numpy(dtype=float)
        )
        nudged = (1.0 - form_anchor) * nudged + form_anchor * form_rank

    rain_p = (
        float(feature_df["rain_probability"].iloc[0])
        if "rain_probability" in feature_df.columns and len(feature_df)
        else 0.0
    )

    exp_pos, win_probs = _simulate_outcomes(
        nudged, feature_df, rain_p=rain_p, dnf_aware=dnf_aware
    )

    order = np.argsort(exp_pos)
    out = []
    for i, idx in enumerate(order):
        row = feature_df.iloc[idx]
        did = row["driver_id"]
        meta = driver_meta.get(did, {})
        out.append(
            {
                "rank": i + 1,
                "driver_id": did,
                "driver_name": meta.get("name", did),
                "constructor": meta.get("constructor", row["constructor_id"]),
                "code": meta.get("code", did[:3].upper()),
                "predicted_position": float(exp_pos[idx]),
                "raw_model_position": float(raw[idx]),
                "news_factor": float(row["news_factor"]),
                "win_probability": float(win_probs[idx]),
            }
        )
    return out


def _simulate_outcomes(
    pace: np.ndarray, feature_df: pd.DataFrame, rain_p: float = 0.0, dnf_aware: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Monte Carlo over race outcomes → (expected_position, win_probability).

    Replaces the old softmax-over-rank, which only looked at rank and
    ignored both score gaps and reliability. Each simulation draws
    per-driver pace noise (wider in rain) and DNF events; DNF'd cars drop
    to the back. Win probability is then literally "share of simulations
    won", which is calibrated by construction given the inputs.
    """
    n = len(pace)
    if n == 0:
        return np.array([]), np.array([])

    rng = np.random.default_rng(7)
    n_sims = 4000
    rain_chaos = min(max(rain_p, 0.0), 0.6)
    noise_scale = 2.3 + 2.5 * rain_chaos

    sims = pace[None, :] + rng.normal(0.0, noise_scale, size=(n_sims, n))

    if dnf_aware:
        # P(DNF) per driver: recent driver rate shrunk toward the team rate
        # and a global base rate (small samples otherwise dominate).
        driver_dnf = (
            feature_df.get("dnf_rate_last5", pd.Series([0.1] * n)).fillna(0.1).to_numpy(dtype=float)
        )
        if "constructor_id" in feature_df.columns:
            team_dnf = (
                feature_df.assign(_d=driver_dnf)
                .groupby("constructor_id")["_d"]
                .transform("mean")
                .to_numpy()
            )
        else:
            team_dnf = driver_dnf
        p_dnf = np.clip(0.5 * driver_dnf + 0.3 * team_dnf + 0.2 * 0.10, 0.02, 0.40)
        # Rain raises everyone's chance of an incident
        p_dnf = np.clip(p_dnf * (1.0 + 0.8 * rain_chaos), 0.02, 0.55)
        dnf_draws = rng.random((n_sims, n)) < p_dnf[None, :]
        sims = np.where(dnf_draws, sims + 100.0, sims)

    sim_ranks = sims.argsort(axis=1).argsort(axis=1)  # 0 = wins that sim
    win_probs = (sim_ranks == 0).mean(axis=0)
    exp_pos = sim_ranks.mean(axis=0) + 1.0
    return exp_pos, win_probs


# ---------------------------------------------------------------------------
# Persistence convenience
# ---------------------------------------------------------------------------


def models_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "models_cache"))


def race_model_path() -> str:
    return os.path.join(models_dir(), "race_predictor.pkl")


def pole_model_path() -> str:
    return os.path.join(models_dir(), "pole_predictor.pkl")
