"""Build explicit train/dev/test manifests from Common Voice Scripted Speech.

The archive's official TSV split is retained.  Only the ``path`` column is
used to locate audio under ``clips/``; missing files or sentences are skipped
and reported instead of being inferred.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


SPLITS = ("train", "dev", "test")


def import_split(source: Path, split: str) -> tuple[list[dict], list[dict]]:
    tsv = source / f"{split}.tsv"
    if not tsv.is_file():
        raise SystemExit(f"Missing official split file: {tsv}")
    items, skipped = [], []
    with tsv.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"path", "sentence"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise SystemExit(f"{tsv} must contain: {', '.join(sorted(required))}")
        for row_number, row in enumerate(reader, 2):
            filename = (row.get("path") or "").strip()
            transcript = (row.get("sentence") or "").strip()
            audio = source / "clips" / filename
            if not transcript or not filename or not audio.is_file():
                skipped.append({
                    "split": split, "row": row_number,
                    "reason": "missing_sentence" if not transcript else "unresolved_audio",
                })
                continue
            items.append({
                "audio_path": str(audio.resolve()),
                "transcript": transcript,
                "source": "mozilla_common_voice_scripted_27",
                "dataset_split": split,
                "speaker_id": (row.get("client_id") or "").strip() or None,
                "metadata_file": tsv.name,
                "metadata_row": row_number,
            })
    return items, skipped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help=".../cv-corpus-*/tr directory")
    parser.add_argument("--output", type=Path, default=Path("voice/data/common_voice_scripted"))
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if not (source / "clips").is_dir():
        raise SystemExit(f"Expected Common Voice clips directory under: {source}")
    output.mkdir(parents=True, exist_ok=True)
    all_skipped = []
    for split in SPLITS:
        items, skipped = import_split(source, split)
        with (output / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for item in items:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        all_skipped.extend(skipped)
        print(f"{split}: {len(items)} explicit audio/transcript pairs; skipped {len(skipped)}.")
    with (output / "skipped.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "row", "reason"])
        writer.writeheader()
        writer.writerows(all_skipped)


if __name__ == "__main__":
    main()
