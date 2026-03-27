# NHL Prediction Model — Working Notes
**Project**: NHL daily ML prediction model (MotherDuck / DuckDB)
**Updated**: 2026-03-26

---

## Database
- **Engine**: MotherDuck (DuckDB cloud)
- **DB name**: set via `MOTHERDUCK_DB` in `.env` — all scripts resolve to `my_db`
- **Teams table abbreviations**: Uses NHL standard (NJD, SJS, TBL, LAK, VGK) — NOT Hockey Reference style (N.J, S.J, T.B, L.A, VEG)

## Active Scripts (core pipeline)
| Script | Purpose | Run when |
|---|---|---|
| `train_and_predict.py` | Retrain models + predict today's games | Daily |
| `scrape_gamestats.py` | Pull team box score stats from Hockey Reference | After games |
| `scrape_nhl_lineups.py` | Pull starting goalies + active players from NHL API | Morning + evening |
| `scrape_dailyfaceoff.py` | Lineup status + starting goalies from Daily Faceoff | Morning + evening |
| `update_player_accuracy.py` | Update player prediction actuals from NHL API | Daily AM |
| `player_model.py` | Player-level ML model | (imported) |
| `team_model.py` | Team-level ML model | (imported) |
| `db_features.py` | Feature engineering for models | (imported) |
| `db_connection.py` | Shared MotherDuck connection | (imported) |

## One-Time Fix Scripts
| Script | Purpose |
|---|---|
| `add_missing_players.py` | Insert players on NHL rosters missing from Players table |
| `fix_mangled_names.py` | Fix truncated/broken player names from import encoding issues |
| `fix_bad_jersey_matches.py` | Null out jersey numbers matched to wrong players |
| `fix_missing_scores.py` | Backfill NULL scores from Hockey Reference |
| `fix_null_gamestats.py` | Delete incomplete GameStats rows for re-scraping |
| `fix_pending_predictions.py` | Resolve old Predictions rows in legacy integer-FK format |

## Database Tables (DDL: nhl_motherduck_ddl.sql)
- `Teams` — 32 teams, static reference
- `Games` — one row per game, FK → Teams
- `GameStats` — two rows per game (one per team), FK → Games + Teams
- `Players` — FK → Teams
- `PlayerStats` — FK → Players + Games
- `Predictions` — FK → Games + Teams
- `PlayerPredictions` — FK → Players + Games
- `PlayerGameStatus` — FK → Players + Games (columns: StatusID, PlayerID, GameID, Status, StatusDate, ConfirmedLineup)
- `StatCategories`, `TeamStandings`, `PlayerContracts`, `GoalieStats`

---

## Known Issues & Fixes

### [FIXED 2026-03-26] Windows charmap crash (all core scripts)
- **Cause**: Unicode symbols (`✓`, `✗`, `→`, `≥`, `⭐`) in print statements fail on Windows cp1252
- **Fix**: Replaced with ASCII equivalents in all core pipeline scripts

### [FIXED 2026-03-26] Team abbreviation mismatch in NHL API scripts
- **Cause**: Scripts used Hockey Reference abbreviation mappings (NJD→N.J, VGK→VEG etc.) but the DB Teams table uses NHL standard abbreviations (same as API)
- **Fixed in**: `scrape_nhl_lineups.py`, `update_player_accuracy.py`, `fix_mangled_names.py`
- **Rule**: No abbreviation mapping needed when querying Teams table from NHL API data

### [FIXED 2026-03-26] PlayerGameStatus INSERT used wrong column name
- **Error**: `Table PlayerGameStatus does not have a column with name "Notes"` (column is `GameID`)
- **Fix**: Updated INSERT in `scrape_nhl_lineups.py` to use `(StatusID, PlayerID, Status, StatusDate, ConfirmedLineup)`

### [FIXED 2026-03-26] Starting goalie not captured for LIVE/FINAL games
- **Cause**: NHL API `starter=True` only set during PRE game state; is NULL once game starts
- **Fix**: Added TOI-based fallback in `scrape_nhl_lineups.py` — goalie with highest TOI > 0 = starter
- **Result**: 25 starting goalies captured today across 13 games, all 32 teams resolved

### [FIXED 2026-03-26] Daily Faceoff goalie start → wrong status
- **Cause**: `classify_status()` returned `"Active"` for goalie start news
- **Fix**: Returns `"Starting Goalie"` for goalie start category

### [FIXED 2026-03-26] Player accuracy recording 0 players updated
- **Cause 1**: API abbreviation mapping converting NJD→N.J etc. — home team not found → nhl_game_id = None → skip
- **Cause 2**: Player name matching only matched exact full names; NHL API returns abbreviated names ("M. Tkachuk")
- **Cause 3**: Players missing from boxscore (scratches) were skipped, leaving ActualGoals=NULL forever
- **Fix**: Removed abbreviation mapping; added 3-tier name matching (exact → abbreviated → last name); scratches recorded as 0

### [FIXED 2026-03-26] Players missing from DB
- **Cause**: `Players` table was populated from a static Excel file; new signings/trades since then were absent
- **Fix**: `add_missing_players.py` — inserted 61 missing players from NHL API rosters (including Barkov, Cernak, Kopitar etc.)
- **Also**: `fix_mangled_names.py` — fixed 801 bio records; `fix_bad_jersey_matches.py` — nulled 18 bad jersey matches

### [FIXED 2026-03-26] Jacob Markstrom name mangled
- **Stored as**: `Markstrm` (missing 'o') — didn't match NHL API "J. Markstrom"
- **Fix**: Direct UPDATE to `Markstrom`

### [FIXED 2026-03-26] Anze Kopitar first name mangled
- **Stored as**: `Ane Kopitar` — didn't match NHL API "A. Kopitar"
- **Fix**: Direct UPDATE to `Anze`

### [FIXED 2026-03-26] Model accuracy only showing 15 predictions
- **Cause**: Old predictions had `Season = NULL`; query filtered `WHERE Season = '2025-26'`
- **Fix**: Updated all 504 old Predictions rows to set Season from linked Games table

### [FIXED 2026-03-26] WinnerTeamID mismatch on GameID 2009
- BOS 10-2 NYR on 2026-01-10 had WinnerTeamID pointing to NYR — corrected to BOS

---

## FK Constraint Notes
Foreign keys that block new data:
- `GameStats.GameID` → game row must exist in `Games` before stats can be inserted
- `GameStats.TeamID` → team must exist in `Teams`
- `PlayerGameStatus.PlayerID` → player must exist in `Players`
- `PlayerGameStatus.GameID` → game must exist in `Games`
- `Predictions.GameID` → game must be in `Games`

If scraping fails with FK violation, check:
1. Is the game already in `Games`?
2. Is the team abbreviation correct? (DB uses NHL standard: NJD, SJS, TBL, LAK, VGK)

---

## Model Performance (as of 2026-03-26)
- CV accuracy (team model): 60.7% ±2.0%
- Season predictions: 519 total, 506 resolved
- Correct: 242 / Wrong: 264 → 47.8% actual accuracy
- Gap vs CV expected: old predictions used earlier model version; new predictions should trend toward 60%+
- Player accuracy (2-day sample): goal 28.6% [>=20% threshold], point 47.8% [>=40% threshold]

---

## Player Accuracy Notes
- `PlayerPredictions` only has data from 2026-03-25 onwards (model newly generating these)
- Scratched players (not in boxscore) are recorded as 0 goals/assists/points
- `update_player_accuracy.py` now covers full season (2025-10-01 onwards, LIMIT 300)
- Daily Faceoff is primary source for lineup status and goalie starts
- NHL API boxscore is secondary source (via `scrape_nhl_lineups.py`) for confirmed dressed players
