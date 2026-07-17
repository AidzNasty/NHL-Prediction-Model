# NHL Predictions — Web App

A live-data website that emulates the Streamlit dashboard (`app.py`), rebuilt as a
Flask API + polished single-page front end. Same MotherDuck data, same 7 pages,
faster and more responsive.

## Run

```bash
pip install -r ../requirements.txt      # needs flask (added) + duckdb, pandas, python-dotenv
python server.py                         # from the web/ directory
# → http://127.0.0.1:8000
```

Reads `MOTHERDUCK_TOKEN` / `MOTHERDUCK_DB` from the project `.env` (same as `app.py`).

## Architecture

- **`server.py`** — Flask backend. Reuses the exact SQL from `app.py`, exposed as
  JSON endpoints. Single shared MotherDuck connection guarded by a lock, with a
  5-minute TTL cache so repeated page loads don't re-hit the DB.
- **`static/index.html`** — app shell (sidebar nav, model-status panel).
- **`static/styles.css`** — dark hockey theme (cyan accent, Bebas Neue + Inter),
  "data-dense dashboard" layout.
- **`static/app.js`** — client router, fetches the API, renders every page,
  sortable/filterable tables, and Chart.js charts.

### Endpoints
| Route | Page |
|---|---|
| `GET /api/status` | Sidebar model status |
| `GET /api/today` | Today's Games (+ player projections) |
| `GET /api/playoffs` | Playoff series tracker & predictions |
| `GET /api/player-props` | Player Props |
| `GET /api/accuracy` | Model Accuracy |
| `GET /api/team-stats?season=` | Team Stats |
| `GET /api/streaks/teams` · `GET /api/streaks/players?pos=&team=` | Hot & Cold Streaks |

## Enhancements over the Streamlit app

- **Off-season fallback** — when there are no predictions for the literal calendar
  today, Today's Games / Player Props automatically show the most recent slate
  (with a banner), instead of an empty "no predictions" screen.
- **Click-to-sort tables** everywhere, with sticky headers.
- **Instant client-side filtering** (no full-page reruns like Streamlit).
- **Responsive** — sidebar collapses to a drawer on mobile.
- Loading spinners, hover states, colour-coded confidence pills and form chips.

- **Refresh data button** (sidebar) — clears the server-side query cache and
  re-renders the active page, so you can pull fresh predictions without
  restarting. Shows a "Data as of HH:MM:SS" timestamp. Backed by `POST /api/refresh`.

## Charts

Chart.js is loaded from a CDN (`cdn.jsdelivr.net`). If you need a fully offline
build, download `chart.umd.min.js` into `static/` and point the `<script>` in
`index.html` at the local copy.

## Production deployment

The app ships with a `Procfile` (repo root) and uses `gunicorn` (in
`requirements.txt`). Works on Heroku, Railway, Render, Fly.io, etc.

```
web: gunicorn --chdir web server:app --workers 1 --threads 8 --timeout 120 --bind 0.0.0.0:$PORT
```

- **1 worker / 8 threads (gthread):** the app holds one MotherDuck connection
  guarded by a lock, so a single threaded worker keeps exactly one DB connection
  while still handling concurrent requests. The 5-min cache absorbs most load. To
  scale out, raise `--workers` (each worker opens its own MotherDuck connection).
- **`$PORT`** is provided by the platform; `server.py` also honours `PORT` when
  run directly (`python server.py`).

### Required environment variables (set these in the platform dashboard)

| Var | Value |
|---|---|
| `MOTHERDUCK_TOKEN` | your MotherDuck token (do **not** commit it) |
| `MOTHERDUCK_DB` | `my_db` |

Locally these come from the project `.env`; in production set them as config vars.

### Run the production server locally

```bash
pip install -r ../requirements.txt          # installs gunicorn
cd ..                                        # repo root (Procfile location)
PORT=8000 gunicorn --chdir web server:app --workers 1 --threads 8 --bind 0.0.0.0:8000
```

### One-click / auto-deploy

Config files live in the repo root — use the one matching your host:

- **Render** — [`render.yaml`](../render.yaml) Blueprint. In Render: *New →
  Blueprint → connect this repo*. It sets the start command and health check; you
  add `MOTHERDUCK_TOKEN` as a secret in the dashboard (it's marked `sync: false`).
- **Railway** — [`railway.json`](../railway.json). Create a project from the repo;
  Railway auto-detects Python, uses the start command from the file, and deploys on
  every push. Set `MOTHERDUCK_TOKEN` / `MOTHERDUCK_DB` as service variables.
- **GitHub Actions** — [`.github/workflows/deploy.yml`](../.github/workflows/deploy.yml)
  runs on push to `main`: it installs deps and import-checks `server:app`, then
  triggers a Render deploy. One-time setup: copy your Render **Deploy Hook** URL
  into a repo secret named `RENDER_DEPLOY_HOOK_URL`. Without that secret the verify
  step still runs and the deploy step is skipped with a warning. (Railway doesn't
  need this workflow — it deploys from GitHub natively.)

All paths keep `MOTHERDUCK_TOKEN` in platform config/secrets — never in the repo
(`.env` is gitignored).
