"""
fix_predictions_schema.py
--------------------------
Adds missing columns to the existing Predictions table
to match what train_and_predict.py expects.

Run ONCE before running train_and_predict.py.

Usage: python fix_predictions_schema.py
"""

import os
import duckdb
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("MOTHERDUCK_TOKEN")
DB    = os.getenv("MOTHERDUCK_DB", "my_db")

print(f"Connecting to MotherDuck: {DB}...")
con = duckdb.connect(f"md:{DB}?motherduck_token={TOKEN}")
print("Connected!\n")

existing = [r[0].lower() for r in con.execute("""
    SELECT column_name FROM information_schema.columns
    WHERE table_name = 'Predictions'
""").fetchall()]

new_cols = [
    ("HomeTeamID",         "INTEGER"),
    ("AwayTeamID",         "INTEGER"),
    ("PredictedWinner",    "VARCHAR(100)"),
    ("PredictedHomeScore", "INTEGER"),
    ("PredictedAwayScore", "INTEGER"),
    ("ActualWinner",       "VARCHAR(100)"),
    ("ActualHomeScore",    "INTEGER"),
    ("ActualAwayScore",    "INTEGER"),
]

for col, dtype in new_cols:
    if col.lower() not in existing:
        con.execute(f"ALTER TABLE Predictions ADD COLUMN {col} {dtype}")
        print(f"  Added: {col}")
    else:
        print(f"  Already exists: {col}")

# Show final schema
print("\nFinal Predictions schema:")
cols = con.execute("""
    SELECT column_name, data_type
    FROM information_schema.columns
    WHERE table_name = 'Predictions'
    ORDER BY ordinal_position
""").fetchall()
for c in cols:
    print(f"  {c[0]:35} {c[1]}")

con.close()
print("\nDone! Now run: python train_and_predict.py")
