"""
scrape_playergamelog.py
------------------------
Backfills the PlayerGameLog table (per-player, per-game stat lines)
from the NHL gamecenter boxscore API.

The Games table stores an internal GameID, not the NHL API game ID, so
each DB game is matched to its NHL game via the schedule API (by date +
home-team abbreviation), then the boxscore is fetched and parsed.

Fields populated from the boxscore player entries:
    Goals, Assists, Points, PlusMinus, PIM, Shots (sog), Hits,
    Blocks (blockedShots), Giveaways, Takeaways, TOI (sec), PPGoals
Fields the boxscore endpoint does not expose are stored as 0
    (TOI_PP, TOI_SH, PPAssists, SHGoals, SHAssists, FaceoffWins, FaceoffLosses),
matching the existing rows in the table.

Players are matched to the DB Players table by name, restricted to the
game's two teams to avoid collisions. Boxscore players not found in the
DB are skipped and reported (run add_missing_players.py to add them).

USAGE:
------
python scrape_playergamelog.py --season 2025-26          # all games missing logs
python scrape_playergamelog.py --season 2025-26 --limit 5   # test
python scrape_playergamelog.py --date 2026-04-06         # one date
python scrape_playergamelog.py --game 2463               # one DB GameID
"""

import os
import sys
import time
import argparse
import unicodedata
import requests
import duckdb
from datetime import date
from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()
TOKEN = os.getenv("MOTHERDUCK_TOKEN")
DB    = os.getenv("MOTHERDUCK_DB", "my_db")

if not TOKEN:
    raise ValueError("MOTHERDUCK_TOKEN not found in .env file")

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}

# ============================================================
# Args
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--season", default="2025-26", help="Season e.g. 2025-26")
parser.add_argument("--date",   default=None, help="Only games on this date (YYYY-MM-DD)")
parser.add_argument("--game",   type=int, default=None, help="Single DB GameID")
parser.add_argument("--limit",  type=int, default=None, help="Max games (testing)")
parser.add_argument("--force",  action="store_true",
                    help="Reprocess games even if they already have PlayerGameLog rows")
parser.add_argument("--auto-add", dest="auto_add", action="store_true",
                    help="Insert players missing from the Players table (from NHL API) "
                         "instead of skipping them")
args = parser.parse_args()

# ============================================================
# Connect
# ============================================================
print(f"Connecting to MotherDuck: {DB}...")
con = duckdb.connect(f"md:{DB}?motherduck_token={TOKEN}")
print("Connected!\n")

# ============================================================
# Helpers
# ============================================================
def strip_accents(s):
    return "".join(
        c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)
    )

def norm(s):
    return strip_accents((s or "").strip().lower())

def toi_to_seconds(toi):
    """'17:00' -> 1020. Returns 0 on bad input."""
    try:
        m, s = str(toi).split(":")
        return int(m) * 60 + int(s)
    except Exception:
        return 0

_team_lookup_cache = {}

def build_team_lookup(team_id):
    """Name -> DB PlayerID for one team, with full-name / 'f. last' / last-name keys.
    Cached per team_id (rosters don't change during a run)."""
    if team_id in _team_lookup_cache:
        return _team_lookup_cache[team_id]
    rows = con.execute(
        f"SELECT PlayerID, FirstName, LastName FROM Players WHERE TeamID = {team_id}"
    ).fetchall()
    lk = {}
    for pid, fn, ln in rows:
        fn_n, ln_n = norm(fn), norm(ln)
        if fn_n and ln_n:
            lk[f"{fn_n} {ln_n}"] = pid
            lk[f"{fn_n[0]}. {ln_n}"] = pid
        if ln_n:
            lk.setdefault(f"__last__{ln_n}", pid)
    _team_lookup_cache[team_id] = lk
    return lk

def build_global_lookups():
    """League-wide unambiguous name -> PlayerID maps, used as a fallback for
    players who changed teams since the game (esp. cross-season backfills).
    Only names that resolve to exactly one PlayerID are kept."""
    rows = con.execute("SELECT PlayerID, FirstName, LastName FROM Players").fetchall()
    full, flast, last = {}, {}, {}
    for pid, fn, ln in rows:
        fn_n, ln_n = norm(fn), norm(ln)
        if fn_n and ln_n:
            full.setdefault(f"{fn_n} {ln_n}", set()).add(pid)
            flast.setdefault(f"{fn_n[0]}. {ln_n}", set()).add(pid)
        if ln_n:
            last.setdefault(ln_n, set()).add(pid)
    uniq = lambda d: {k: next(iter(v)) for k, v in d.items() if len(v) == 1}
    return {"full": uniq(full), "flast": uniq(flast), "last": uniq(last)}

def match_player(lk, glob, raw_name):
    nm = norm(raw_name)
    parts = nm.split()
    flast = f"{parts[0][0]}. {parts[-1]}" if len(parts) >= 2 else None
    last  = parts[-1] if parts else None

    # 1. Team-restricted (most reliable)
    if nm in lk:
        return lk[nm]
    if flast and flast in lk:
        return lk[flast]
    if last and f"__last__{last}" in lk:
        return lk[f"__last__{last}"]

    # 2. League-wide, only when unambiguous (handles traded/moved players)
    if nm in glob["full"]:
        return glob["full"][nm]
    if flast and flast in glob["flast"]:
        return glob["flast"][flast]
    if last and last in glob["last"]:
        return glob["last"][last]
    return None

# -- Auto-add players missing from the DB (from NHL API) -------
_nhl_player_cache = {}   # nhl_playerId -> DB PlayerID (added this run)
_next_player_id   = [con.execute("SELECT COALESCE(MAX(PlayerID), 0) + 1 FROM Players").fetchone()[0]]

def title_ascii(s):
    return strip_accents((s or "").strip()).title()

def auto_add_player(nhl_pid, team_id):
    """Insert a Players row for a player not found in the DB, using the NHL
    player landing endpoint. Cached by NHL playerId so each missing player is
    inserted only once per run. Returns the new DB PlayerID (or None on failure)."""
    if nhl_pid in _nhl_player_cache:
        return _nhl_player_cache[nhl_pid]
    try:
        land = requests.get(
            f"https://api-web.nhle.com/v1/player/{nhl_pid}/landing",
            headers=HEADERS, timeout=15,
        ).json()
    except Exception as e:
        print(f"    [auto-add error] playerId {nhl_pid}: {e}")
        return None

    first = title_ascii((land.get("firstName", {}) or {}).get("default", ""))
    last  = title_ascii((land.get("lastName", {}) or {}).get("default", ""))
    if not first and not last:
        return None

    def si(v):
        try:
            return int(v) if v is not None else None
        except Exception:
            return None

    pid = _next_player_id[0]
    con.execute("""
        INSERT INTO Players
        (PlayerID, TeamID, FirstName, LastName, Position,
         JerseyNumber, IsActive, HeightInches, WeightLbs,
         DateOfBirth, BirthCity, BirthCountry)
        VALUES (?, ?, ?, ?, ?, ?, FALSE, ?, ?, ?, ?, ?)
    """, [
        pid, team_id, first, last, land.get("position"),
        si(land.get("sweaterNumber")),
        si(land.get("heightInInches")), si(land.get("weightInPounds")),
        (land.get("birthDate") or None),
        title_ascii((land.get("birthCity", {}) or {}).get("default", "")) or None,
        land.get("birthCountry"),
    ])
    _next_player_id[0] += 1
    _nhl_player_cache[nhl_pid] = pid
    return pid

# schedule cache: date_str -> {home_abbrev: nhl_game_id}
_schedule_cache = {}

def nhl_game_id_for(game_date, home_abbrev):
    date_str = str(game_date)[:10]
    if date_str not in _schedule_cache:
        mapping = {}
        try:
            resp = requests.get(
                f"https://api-web.nhle.com/v1/schedule/{date_str}",
                headers=HEADERS, timeout=15,
            )
            if resp.status_code == 200:
                for week in resp.json().get("gameWeek", []):
                    if str(week.get("date", ""))[:10] == date_str:
                        for g in week.get("games", []):
                            ha = g.get("homeTeam", {}).get("abbrev", "")
                            if ha:
                                mapping[ha] = g.get("id")
        except Exception as e:
            print(f"  [schedule error] {date_str}: {e}")
        _schedule_cache[date_str] = mapping
        time.sleep(0.4)
    return _schedule_cache[date_str].get(home_abbrev)

# ============================================================
# Build target game list
# ============================================================
where = [
    "g.HomeScore IS NOT NULL",
    "g.AwayScore IS NOT NULL",
    f"g.Season = '{args.season}'",
]
if not args.force:
    where.append("pgl.GameID IS NULL")   # only games with no rows yet
if args.date:
    where.append(f"g.GameDate = '{args.date}'")
if args.game:
    where = [f"g.GameID = {args.game}"]  # explicit single game (ignore other filters)

query = f"""
    SELECT g.GameID, g.GameDate, g.Season, g.GameType,
           g.HomeTeamID, g.AwayTeamID,
           ht.Abbreviation AS HomeAbbrev, awt.Abbreviation AS AwayAbbrev
    FROM Games g
    JOIN Teams ht  ON g.HomeTeamID = ht.TeamID
    JOIN Teams awt ON g.AwayTeamID = awt.TeamID
    LEFT JOIN (SELECT DISTINCT GameID FROM PlayerGameLog) pgl ON g.GameID = pgl.GameID
    WHERE {" AND ".join(where)}
    ORDER BY g.GameDate ASC
"""
games = con.execute(query).fetchall()
if args.limit:
    games = games[:args.limit]

print(f"Games to backfill: {len(games)}\n")
if not games:
    print("Nothing to do — all matching games already have PlayerGameLog rows.")
    con.close()
    sys.exit(0)

# ============================================================
# Main loop
# ============================================================
next_id = con.execute("SELECT COALESCE(MAX(PlayerGameLogID), 0) FROM PlayerGameLog").fetchone()[0] + 1
glob = build_global_lookups()

INSERT_SQL = """
INSERT INTO PlayerGameLog (
    PlayerGameLogID, PlayerID, GameID, TeamID, GameDate, Season, IsHome,
    Goals, Assists, Points, PlusMinus, PIM, Shots, Hits, Blocks,
    Giveaways, Takeaways, TOI, TOI_PP, TOI_SH,
    PPGoals, PPAssists, SHGoals, SHAssists, FaceoffWins, FaceoffLosses, GameType
) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
"""

games_done = rows_inserted = players_skipped = games_failed = 0

for gi, (game_id, game_date, season, game_type,
         home_id, away_id, home_abbrev, away_abbrev) in enumerate(games):

    print(f"[{gi+1}/{len(games)}] {str(game_date)[:10]} | {away_abbrev} @ {home_abbrev} (GameID {game_id})")

    nhl_id = nhl_game_id_for(game_date, home_abbrev)
    if not nhl_id:
        print(f"  [SKIP] Could not resolve NHL game ID")
        games_failed += 1
        continue

    try:
        resp = requests.get(
            f"https://api-web.nhle.com/v1/gamecenter/{nhl_id}/boxscore",
            headers=HEADERS, timeout=15,
        )
        if resp.status_code != 200:
            print(f"  [SKIP] boxscore HTTP {resp.status_code}")
            games_failed += 1
            continue
        pbg = resp.json().get("playerByGameStats", {})
        if not pbg:
            print(f"  [SKIP] no playerByGameStats")
            games_failed += 1
            continue

        # Idempotent: clear any existing rows for this game so re-runs don't duplicate
        con.execute(f"DELETE FROM PlayerGameLog WHERE GameID = {game_id}")

        game_rows = 0
        game_skips = []
        for side, team_id, is_home in [("homeTeam", home_id, True),
                                       ("awayTeam", away_id, False)]:
            lk = build_team_lookup(team_id)
            side_data = pbg.get(side, {})
            for group in ("forwards", "defense", "goalies"):
                for p in side_data.get(group, []):
                    raw = p.get("name", {}).get("default", "")
                    pid = match_player(lk, glob, raw)
                    if not pid and args.auto_add:
                        nhl_pid = p.get("playerId")
                        if nhl_pid:
                            pid = auto_add_player(nhl_pid, team_id)
                    if not pid:
                        game_skips.append(f"{side[:4]}:{raw}")
                        players_skipped += 1
                        continue

                    goals   = int(p.get("goals", 0) or 0)
                    assists = int(p.get("assists", 0) or 0)
                    con.execute(INSERT_SQL, [
                        next_id, pid, game_id, team_id, game_date, season, is_home,
                        goals, assists, goals + assists,
                        int(p.get("plusMinus", 0) or 0),
                        int(p.get("pim", 0) or 0),
                        int(p.get("sog", 0) or 0),
                        int(p.get("hits", 0) or 0),
                        int(p.get("blockedShots", 0) or 0),
                        int(p.get("giveaways", 0) or 0),
                        int(p.get("takeaways", 0) or 0),
                        toi_to_seconds(p.get("toi", "0:00")),
                        0, 0,                                   # TOI_PP, TOI_SH (not in endpoint)
                        int(p.get("powerPlayGoals", 0) or 0),
                        0, 0, 0,                                # PPAssists, SHGoals, SHAssists
                        0, 0,                                   # FaceoffWins, FaceoffLosses (only pct given)
                        game_type,
                    ])
                    next_id += 1
                    game_rows += 1

        if game_rows == 0:
            print(f"  [SKIP] no players matched")
            games_failed += 1
        else:
            rows_inserted += game_rows
            games_done += 1
            note = f"  ({len(game_skips)} unmatched: {game_skips})" if game_skips else ""
            print(f"  OK {game_rows} player rows{note}")

    except Exception as e:
        print(f"  [ERROR] {e}")
        games_failed += 1

    time.sleep(0.5)

# ============================================================
# Summary
# ============================================================
print(f"\n=== PlayerGameLog Backfill Complete ===")
print(f"  Games done:       {games_done}")
print(f"  Games failed:     {games_failed}")
print(f"  Rows inserted:    {rows_inserted}")
print(f"  Players auto-added: {len(_nhl_player_cache)} (inserted into Players from NHL API)")
print(f"  Players skipped:  {players_skipped} (no NHL playerId / lookup failed)")
total = con.execute("SELECT COUNT(*) FROM PlayerGameLog").fetchone()[0]
print(f"  Total in DB:      {total} PlayerGameLog rows")
con.close()
print("\nDone!")
