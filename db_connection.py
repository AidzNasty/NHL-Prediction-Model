"""
db_connection.py
----------------
Shared MotherDuck connection module for the NHL Prediction Model.
Import this in any script that needs database access.

Usage:
    from db_connection import get_connection

    con = get_connection()
    df = con.execute("SELECT * FROM Teams").df()
"""

import os
import duckdb
from dotenv import load_dotenv

# Load environment variables from .env file in the same directory
load_dotenv()

MOTHERDUCK_TOKEN = os.getenv("MOTHERDUCK_TOKEN")
MOTHERDUCK_DB    = os.getenv("MOTHERDUCK_DB", "nhl_prediction_db")


def get_connection() -> duckdb.DuckDBPyConnection:
    """
    Returns an active MotherDuck connection.
    The database is created automatically if it doesn't exist yet.
    """
    if not MOTHERDUCK_TOKEN:
        raise ValueError(
            "MOTHERDUCK_TOKEN not found. "
            "Make sure your .env file exists and contains MOTHERDUCK_TOKEN."
        )

    connection_string = f"md:{MOTHERDUCK_DB}?motherduck_token={MOTHERDUCK_TOKEN}"

    try:
        con = duckdb.connect(connection_string)
        print(f"[db_connection] Connected to MotherDuck: {MOTHERDUCK_DB}")
        return con
    except Exception as e:
        raise ConnectionError(f"[db_connection] Failed to connect to MotherDuck: {e}")


def run_ddl(ddl_path: str) -> None:
    """
    Utility function to run a .sql DDL file against MotherDuck.
    Use once to initialize the schema.

    Args:
        ddl_path: Path to the .sql file (e.g. 'nhl_motherduck_ddl.sql')
    """
    con = get_connection()
    with open(ddl_path, "r") as f:
        sql = f.read()

    # Split on semicolons and execute each statement
    statements = [s.strip() for s in sql.split(";") if s.strip()]
    for stmt in statements:
        # Skip comment-only blocks
        if stmt.startswith("--"):
            continue
        try:
            result = con.execute(stmt)
            # Print output of SHOW TABLES if present
            if "SHOW" in stmt.upper():
                print("\n[db_connection] Tables in database:")
                print(result.df().to_string(index=False))
        except Exception as e:
            print(f"[db_connection] Error executing statement:\n{stmt}\nError: {e}")

    con.close()
    print("\n[db_connection] DDL execution complete.")


if __name__ == "__main__":
    # Run this file directly to initialize the schema
    # python db_connection.py
    import os
    ddl_file = os.path.join(os.path.dirname(__file__), "nhl_motherduck_ddl.sql")
    print(f"[db_connection] Running DDL from: {ddl_file}")
    run_ddl(ddl_file)
