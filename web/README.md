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

## Deployment

Not deployed. The live site is the Streamlit app (`app.py`) on Streamlit
Community Cloud, which redeploys from GitHub `main`. This Flask version runs
locally only (see **Run** above).
