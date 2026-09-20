"""Print non-sensitive availability statistics for Wispr Flow History rows."""
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
        row = connection.execute(
            """
            SELECT
              COUNT(*),
              SUM(audio IS NOT NULL), SUM(opusChunks IS NOT NULL),
              SUM(asrText IS NOT NULL AND length(trim(asrText)) > 0),
              SUM(formattedText IS NOT NULL AND length(trim(formattedText)) > 0),
              SUM(editedText IS NOT NULL AND length(trim(editedText)) > 0),
              MIN(duration), MAX(duration), AVG(duration),
              MIN(length(audio)), MAX(length(audio)), AVG(length(audio)),
              MIN(length(opusChunks)), MAX(length(opusChunks)), AVG(length(opusChunks))
            FROM History
            """
        ).fetchone()
        names = [
            "history_rows", "audio_rows", "opus_chunk_rows", "asr_text_rows",
            "formatted_text_rows", "edited_text_rows", "duration_min", "duration_max",
            "duration_avg", "audio_bytes_min", "audio_bytes_max", "audio_bytes_avg",
            "opus_bytes_min", "opus_bytes_max", "opus_bytes_avg",
        ]
        for name, value in zip(names, row):
            print(f"{name}={value}")
        for column in ("audio", "opusChunks"):
            print(f"{column}_storage_types=")
            for item_type, count in connection.execute(
                f"SELECT typeof({column}), COUNT(*) FROM History GROUP BY typeof({column})"
            ):
                print(f"  {item_type}:{count}")


if __name__ == "__main__":
    main()
