"""FastAPI server exposing predictions + serving the static frontend."""

from __future__ import annotations

import logging
import os
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import data_sources, predict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(title="F1 Predictor", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache the latest prediction in-process. Predictions are expensive (LLM call
# + several HTTP fetches), and the underlying signals don't change minute-to-
# minute. We refresh every 15 min unless the client passes ?fresh=1.
_PREDICTION_CACHE: dict = {}
_PREDICTION_TTL = 900


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/next-race")
def next_race():
    race = data_sources.get_next_race()
    if not race:
        raise HTTPException(status_code=404, detail="No upcoming race found")
    return race


@app.get("/api/predict")
def get_prediction(fresh: int = 0):
    now = time.time()
    cached = _PREDICTION_CACHE.get("payload")
    cached_at = _PREDICTION_CACHE.get("ts", 0)

    if cached and not fresh and now - cached_at < _PREDICTION_TTL:
        return {**cached, "cached": True, "cached_at": cached_at}

    try:
        payload = predict.predict_next_race()
    except Exception as e:
        log.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}") from e

    _PREDICTION_CACHE["payload"] = payload
    _PREDICTION_CACHE["ts"] = now
    return {**payload, "cached": False, "cached_at": now}


@app.get("/api/news")
def get_news():
    items = data_sources.get_recent_news()
    return {"items": [item.__dict__ for item in items]}


# ---------------------------------------------------------------------------
# Static frontend
# ---------------------------------------------------------------------------

_FRONTEND_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "frontend"))

if os.path.isdir(_FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=_FRONTEND_DIR), name="static")

    @app.get("/")
    def index():
        return FileResponse(os.path.join(_FRONTEND_DIR, "index.html"))
