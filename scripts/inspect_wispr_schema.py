"""Print only Wispr Flow's SQLite schema and row counts; never transcript content."""
from __future__ import annotations

import argparse
import os
import sqlite3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("database")
    args = parser.parse_args()
    path = os.path.abspath(args.database)
    uri = "file:" + path.replace("\\", "/") + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        tables = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
        ).fetchall()
        print(f"DB_BYTES={os.path.getsize(path)}")
        for (table,) in tables:
            escaped = table.replace("]", "]]" )
            columns = ",".join(
                row[1] for row in connection.execute(f"PRAGMA table_info([{escaped}])")
            )
            row_count = connection.execute(f"SELECT COUNT(*) FROM [{escaped}]").fetchone()[0]
            print(f"TABLE={table} COLUMNS={columns} ROWS={row_count}")


if __name__ == "__main__":
    main()
