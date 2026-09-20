"""Export only directly paired Wispr History WAV/transcript rows.

The SQLite source is always opened read-only. Each emitted manifest row comes
from the same History row as its WAV blob, never from timestamp heuristics.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("database")
    parser.add_argument("--output", default="voice/data/wispr")
    args = parser.parse_args()
    database = os.path.abspath(args.database)
    output = Path(args.output)
    audio_dir = output / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    uri = "file:" + database.replace("\\", "/") + "?mode=ro"

    with sqlite3.connect(uri, uri=True) as connection:
        rows = connection.execute(
            """
            SELECT rowid, timestamp, duration, asrText, formattedText, editedText, audio
            FROM History
            WHERE audio IS NOT NULL
              AND COALESCE(NULLIF(trim(asrText), ''), NULLIF(trim(formattedText), ''),
                           NULLIF(trim(editedText), '')) IS NOT NULL
            ORDER BY timestamp
            """
        ).fetchall()

    manifest_path = output / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for index, (rowid, timestamp, duration, asr, formatted, edited, audio) in enumerate(rows, 1):
            filename = f"wispr_{index:04d}.wav"
            (audio_dir / filename).write_bytes(audio)
            record = {
                "audio_path": str(Path("audio") / filename),
                "source_row": rowid,
                "timestamp": timestamp,
                "duration_seconds": duration,
                "asr_text": asr,
                "formatted_text": formatted,
                "edited_text": edited,
            }
            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Exported {len(rows)} direct WAV/transcript pairs to {output}")


if __name__ == "__main__":
    main()
