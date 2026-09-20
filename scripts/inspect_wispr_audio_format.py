"""Identify container metadata for History audio rows without printing transcripts."""
from __future__ import annotations

import argparse
import os
import sqlite3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("database")
    args = parser.parse_args()
    uri = "file:" + os.path.abspath(args.database).replace("\\", "/") + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        rows = connection.execute(
            """
            SELECT rowid, duration, length(audio), hex(substr(audio, 1, 16)),
                   length(opusChunks), substr(opusChunks, 1, 80)
            FROM History
            WHERE audio IS NOT NULL
            ORDER BY timestamp
            """
        ).fetchall()
    for rowid, duration, audio_size, magic, opus_size, opus_prefix in rows:
        print(f"row={rowid} duration_s={duration} audio_bytes={audio_size} magic={magic} "
              f"opus_bytes={opus_size} opus_prefix={opus_prefix!r}")


if __name__ == "__main__":
    main()
