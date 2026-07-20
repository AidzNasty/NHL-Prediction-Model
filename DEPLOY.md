# Deploying the NHL Predictions website

The dashboard is a **read-only viewer** of data in MotherDuck. It does not train
models — [`train_and_predict.py`](train_and_predict.py) does that separately and
writes results to the database. So hosting the site only needs one secret:
your **`MOTHERDUCK_TOKEN`** (it's in your local `.streamlit/secrets.toml` / `.env`,
both gitignored — never commit it).

There are **two frontends** in this repo:

| App | File | Look | Host it on |
|-----|------|------|-----------|
| **Flask + SPA** (recommended) | `web/server.py` + `web/static/` | Custom animated dark UI | Render / Railway |
| Streamlit | `app.py` | Generic Streamlit look | Streamlit Community Cloud |

Both read the same MotherDuck database. Pick one — you don't need both.

---

## Option A — Flask app on Render (recommended, free)

This deploys the polished, animated site. The repo already has a
[`render.yaml`](render.yaml) blueprint for it.

1. Go to <https://render.com> and sign in with GitHub.
2. **New → Blueprint**.
3. Connect the repo **`AidzNasty/NHL-Prediction-Model`**, branch `main`
   (or `web-dashboard`). Render reads `render.yaml` automatically and configures
   the `gunicorn … web/server.py` service.
4. When prompted, set the secret **`MOTHERDUCK_TOKEN`** (marked `sync: false`
   in the blueprint so Render asks for it). `MOTHERDUCK_DB` is already `my_db`.
5. **Apply / Deploy.** You'll get a URL like `https://nhl-predictions.onrender.com`.

**Free-tier note:** Render free services **sleep after ~15 min idle**, so the
first visit after a lull takes ~30–60s to wake. Options:
- Leave it — fine for personal use.
- Keep it awake with a free pinger (e.g. UptimeRobot) hitting `/api/status`
  every few minutes.
- Render's cheapest always-on paid tier (~$7/mo).

**Railway** works the same way via [`railway.json`](railway.json) if you prefer
it over Render.

---

## Option B — Streamlit app on Streamlit Community Cloud (free, always-on)

Deploys `app.py` (generic Streamlit styling, no custom animations).

1. Go to <https://share.streamlit.io> → sign in with GitHub → **Create app**.
2. Repository `AidzNasty/NHL-Prediction-Model`, branch `main`, main file `app.py`.
3. **Advanced settings → Secrets**, paste:
   ```toml
   MOTHERDUCK_TOKEN = "your-token-here"
   MOTHERDUCK_DB = "my_db"
   ```
4. **Deploy** → URL like `https://<app>.streamlit.app`. Doesn't sleep.

---

## Keeping the site's data fresh

The live site only shows what's in MotherDuck. During the season, run the daily
pipeline to generate new predictions (the `pipeline_*.bat` files / Task Scheduler
already do this):

```
python train_and_predict.py          # retrain + predict today's games
```

The website reflects new rows automatically (within its short query cache).

### Backfilling gaps

If the pipeline misses some days, fill any completed games that lack predictions
(team + player, walk-forward / leak-free):

```
python fill_missing_predictions.py --dry-run                 # list what's missing
python fill_missing_predictions.py --game-ids 2790,2791,2792 # fill specific games
python fill_missing_predictions.py --all-missing             # fill every gap
```

To re-score existing prediction rows with the current model:

```
python backfill_predictions.py --dry-run
python backfill_predictions.py
```

---

## Local preview

```
python web/server.py        # Flask app  -> http://localhost:8000
streamlit run app.py        # Streamlit  -> http://localhost:8501
```

Requires `MOTHERDUCK_TOKEN` in the environment / `.env` / `.streamlit/secrets.toml`.
