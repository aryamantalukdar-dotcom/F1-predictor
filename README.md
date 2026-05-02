# F1 Predictor

A live F1 race-prediction app that combines multiple data feeds, an ensemble
ML model, and Claude-powered news analysis to predict:

1. **The order of driver finishes** for the upcoming Grand Prix.
2. **The pole-sitter** for qualifying.

The system is designed to use the most up-to-date information available right
up to race weekend.

## What goes into the prediction

| Signal | Source |
|---|---|
| Race calendar / drivers / standings / past results | [Jolpica F1 API](https://github.com/jolpica/jolpica-f1) (Ergast successor) |
| Per-driver season qualifying | Jolpica `/qualifying` |
| Track-specific historical performance | 5 seasons rolled up from Jolpica |
| Live timing / latest session metadata | [OpenF1](https://openf1.org) |
| Race-day weather forecast | [Open-Meteo](https://open-meteo.com) (no API key required) |
| Latest F1 news | RSS aggregation: Autosport, Motorsport.com, Formula1.com, BBC F1 |
| News-derived per-driver impact factors | Claude Opus 4.7 (adaptive thinking + structured outputs) |
| Recent form, momentum, DNF rate | Computed rolling features |

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Jolpica /  │     │ Open-Meteo  │     │  News RSS   │
│   OpenF1    │     │             │     │   feeds     │
└──────┬──────┘     └──────┬──────┘     └──────┬──────┘
       │                   │                    │
       ▼                   ▼                    ▼
┌─────────────────────────────────────────────────────┐
│              backend/data_sources.py                │
└──────────────────────┬──────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
┌──────────────┐  ┌──────────┐  ┌──────────────────┐
│  features.py │  │ news_    │  │ Claude Opus 4.7  │
│  (rolling +  │  │ analyzer │──▶ (adaptive think, │
│  per-track)  │  │   .py    │  │  structured out) │
└──────┬───────┘  └────┬─────┘  └──────────────────┘
       │               │
       │       per-driver factors
       │               │
       ▼               ▼
┌─────────────────────────────────────────────────────┐
│   models.py — LightGBM race + pole predictors       │
│   (with heuristic fallback before training)         │
└──────────────────────┬──────────────────────────────┘
                       ▼
                ┌──────────────┐
                │   app.py     │  FastAPI + static frontend
                └──────────────┘
```

## Quick start

### 1. Install dependencies

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt
```

### 2. (Optional but recommended) Set your Anthropic API key

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

Without this, the news-analysis step is skipped and predictions still run on
ML signals alone.

### 3. Run predictions immediately

A heuristic baseline ships with the repo, so you can predict on day one:

```bash
python -m scripts.predict_cli
```

Or start the web app:

```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000
# open http://localhost:8000
```

### 4. Train the real models (recommended, takes a few minutes)

```bash
python -m backend.train --seasons 2021 2022 2023 2024 2025
```

This pulls historical results from Jolpica, builds a rolling feature matrix,
trains LightGBM regressors for race finish + qualifying position, and saves
them to `models_cache/`. The next prediction will use them automatically.

## API endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | Health check |
| `GET` | `/api/next-race` | Metadata for the next upcoming race |
| `GET` | `/api/predict?fresh=1` | Full prediction payload (cached 15 min unless `fresh=1`) |
| `GET` | `/api/news` | Latest aggregated F1 news |

## How predictions are produced

1. **Identify the next race** from the Jolpica season schedule.
2. **Fetch live signals**: standings, recent results, qualifying, circuit
   history (5 seasons back), Open-Meteo race-day forecast, latest news RSS.
3. **News analysis**: Claude reads the news + grid context and emits a
   structured `{driver_id: factor_in_[-1, +1]}` map. The system prompt is
   marked for prompt caching, so repeated calls during a race weekend are
   cheap.
4. **Feature engineering**: rolling avg-finish (last 3, last 5), DNF rate,
   form momentum (negative slope of recent positions), per-track history,
   constructor strength, championship state, weather, news factor, and a
   handful of track-type categoricals.
5. **Race + pole models**: two LightGBM regressors trained on historical
   seasons predict expected race position and qualifying position
   respectively. Predictions are ranked and converted to softmax win
   probabilities.
6. **News nudge**: each driver's predicted position is shifted by
   `-1.5 × news_factor`, so a +0.5 factor is worth roughly three-quarters of
   a place.

## Project layout

```
F1-predictor/
├── backend/
│   ├── data_sources.py     # All live API integrations
│   ├── features.py         # Feature engineering (training + inference)
│   ├── news_analyzer.py    # Claude API news → structured factors
│   ├── models.py           # LightGBM race + pole predictors
│   ├── predict.py          # End-to-end pipeline orchestrator
│   ├── train.py            # Training entry point
│   ├── app.py              # FastAPI server + static mounts
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── app.js              # Vanilla JS, no build step
│   └── style.css
├── scripts/
│   └── predict_cli.py      # CLI predictor
├── models_cache/           # Trained LightGBM models (gitignored)
├── data_cache/             # Optional FastF1 cache (gitignored)
└── README.md
```

## Deploy live (always-on URL)

The repo ships with a `Dockerfile` plus config for the major free-tier hosts.
Pick whichever you prefer — each gives you a permanent `https://...` URL.

### Option A — Render (easiest, one click)

1. Push this repo to your GitHub (the `claude/f1-prediction-app-1o7ca` branch
   already has all deploy configs).
2. Go to https://render.com → **New** → **Blueprint** → connect the repo.
   Render reads `render.yaml` and provisions a free Docker web service.
3. In the service's **Environment** tab, add `ANTHROPIC_API_KEY` (optional —
   without it the app still runs, just without news-derived factors).
4. First deploy takes ~5 min. You'll get a URL like
   `https://f1-predictor-xxxx.onrender.com`. The health check at
   `/api/health` keeps the service warm.

> Render's free tier sleeps after 15 min of inactivity and wakes on the next
> request (~30 s cold start). For true always-on, use the Starter plan or
> Fly.io below.

### Option B — Fly.io (always-on, generous free tier)

```bash
# one-time
curl -L https://fly.io/install.sh | sh
fly auth signup   # or: fly auth login

# from the repo root
fly launch --no-deploy --copy-config --name f1-predictor
fly secrets set ANTHROPIC_API_KEY=sk-ant-...   # optional
fly deploy
```

`fly.toml` is pre-configured with a 512 MB shared-CPU VM, HTTPS, and a health
check on `/api/health`. Set `min_machines_running = 1` in `fly.toml` if you
want zero cold starts.

### Option C — Railway

Click **New Project → Deploy from GitHub** and pick the repo. Railway reads
`railway.toml` automatically. Add `ANTHROPIC_API_KEY` in the Variables tab.

### Option D — Any Docker host

```bash
docker build -t f1-predictor .
docker run -p 8000:8000 -e ANTHROPIC_API_KEY=sk-ant-... f1-predictor
```

Works on Google Cloud Run, AWS App Runner, Azure Container Apps, etc.

### Training models in production

The container ships with the heuristic fallback so it works on day one. To
upgrade to the real LightGBM models on a live host, exec into the container
and run training once — the resulting `models_cache/*.pkl` files are picked
up automatically:

```bash
# Render: use the Shell tab
# Fly.io: fly ssh console
python -m backend.train --seasons 2021 2022 2023 2024 2025
```

## Disclaimer

These are model estimates from public data. Don't bet your house on them.
