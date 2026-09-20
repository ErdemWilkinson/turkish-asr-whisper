"""Create a manifest from an extracted Mozilla Common Voice Spontaneous archive.

Only explicit audio filename columns are used for matching. Files with missing
or ambiguous names are reported and excluded rather than guessed.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


AUDIO_COLUMNS = ("audio_file", "audio", "path", "filename", "file", "clip")


def audio_index(source: Path) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {}
    for path in source.rglob("*"):
        if path.suffix.lower() in {".mp3", ".wav", ".ogg", ".opus", ".flac"}:
            result.setdefault(path.name, []).append(path)
    return result


def resolve_audio(value: str, source: Path, by_name: dict[str, list[Path]]) -> Path | None:
    candidate = source / value
    if candidate.is_file():
        return candidate
    for directory in (source / "audios", source / "audio", source / "clips"):
        candidate = directory / value
        if candidate.is_file():
            return candidate
    matches = by_name.get(Path(value).name, [])
    return matches[0] if len(matches) == 1 else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="Extracted Mozilla archive directory")
    parser.add_argument("--output", type=Path, default=Path("voice/data/common_voice_spontaneous"))
    args = parser.parse_args()
    source = args.source.resolve()
    if not source.is_dir():
        raise SystemExit(f"Not a directory: {source}")
    by_name = audio_index(source)
    if not by_name:
        raise SystemExit("No audio files found under the source directory.")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest, skipped = [], []
    seen_audio: set[Path] = set()
    for tsv in source.rglob("*.tsv"):
        with tsv.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if not reader.fieldnames or "transcription" not in reader.fieldnames:
                continue
            audio_column = next((key for key in AUDIO_COLUMNS if key in reader.fieldnames), None)
            if audio_column is None:
                skipped.append({"tsv": str(tsv.relative_to(source)), "row": "*", "reason": "no_audio_column"})
                continue
            for row_number, row in enumerate(reader, 2):
                text = (row.get("transcription") or "").strip()
                value = (row.get(audio_column) or "").strip()
                audio = resolve_audio(value, source, by_name) if value else None
                if not text or audio is None:
                    skipped.append({"tsv": str(tsv.relative_to(source)), "row": row_number,
                                    "reason": "missing_transcription" if not text else "unresolved_audio"})
                    continue
                audio = audio.resolve()
                if audio in seen_audio:
                    continue
                seen_audio.add(audio)
                manifest.append({
                    "audio_path": str(audio), "transcript": text,
                    "source": "mozilla_common_voice_spontaneous",
                    "metadata_file": str(tsv.relative_to(source)), "metadata_row": row_number,
                    "speaker_id": (row.get("client_id") or "").strip() or None,
                    "dataset_split": (row.get("split") or "").strip() or None,
                })
    with (args.output / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for item in manifest:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    with (args.output / "skipped.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["tsv", "row", "reason"])
        writer.writeheader()
        writer.writerows(skipped)
    print(f"Imported {len(manifest)} explicit audio/transcript pairs; skipped {len(skipped)} rows.")


if __name__ == "__main__":
    main()
