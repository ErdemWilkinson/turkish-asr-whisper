"""Create a conservative, metadata-validated Common Voice subset for ASR.

This filters only observable problems (bad text, implausible duration/rate,
duplicates and speaker overrepresentation). It does not claim to judge audio
noise; microphone recordings still need the separate recording protocol.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ALLOWED = re.compile(r"^[a-zçğıöşü' ]+$")


def normalise(text: str) -> str:
    text = text.lower().replace("i̇", "i")
    text = re.sub(r"[^a-zçğıöşü' ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def durations(path: Path) -> dict[str, int]:
    result = {}
    with path.open(encoding="utf-8-sig") as handle:
        next(handle)
        for line in handle:
            name, milliseconds = line.rstrip("\n").split("\t", 1)
            result[name] = int(milliseconds)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "common_voice_scripted")
    parser.add_argument("--durations", type=Path,
                        default=ROOT / "data" / "source" / "cv-corpus-27.0-2026-09-11" / "tr" / "clip_durations.tsv")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "common_voice_quality_v1")
    parser.add_argument("--max-per-speaker", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    clip_ms = durations(args.durations)
    rng = random.Random(args.seed)
    seen_text: set[str] = set()
    report = {"criteria": {"duration_seconds": [1.0, 10.0], "characters": [5, 180],
                            "characters_per_second": [4.0, 26.0], "max_per_speaker": args.max_per_speaker},
              "splits": {}}
    # Test first prevents exact prompts from being reused in train/dev.
    for split in ("test", "dev", "train"):
        source = args.data / f"{split}.jsonl"
        rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
        rng.shuffle(rows)
        accepted, reasons, per_speaker = [], Counter(), Counter()
        for row in rows:
            raw = row["transcript"]
            text = normalise(raw)
            name = Path(row["audio_path"]).name
            duration = clip_ms.get(name, 0) / 1000
            rate = len(text) / duration if duration else 0
            speaker = row.get("speaker_id") or "unknown"
            if "\t" in raw or "\n" in raw or "\r" in raw:
                reasons["malformed_text"] += 1
            elif not ALLOWED.fullmatch(text):
                reasons["unsupported_text"] += 1
            elif len(text) < 5 or len(text) > 180:
                reasons["text_length"] += 1
            elif duration < 1.0 or duration > 10.0:
                reasons["duration"] += 1
            elif rate < 4.0 or rate > 26.0:
                reasons["speaking_rate"] += 1
            elif text in seen_text:
                reasons["duplicate_transcript"] += 1
            elif per_speaker[speaker] >= args.max_per_speaker:
                reasons["speaker_cap"] += 1
            else:
                row["transcript"] = text
                row["quality_tier"] = "metadata_validated_v1"
                row["duration_ms"] = clip_ms[name]
                accepted.append(row)
                seen_text.add(text)
                per_speaker[speaker] += 1
        with (args.output / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for row in accepted:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        report["splits"][split] = {
            "input": len(rows), "accepted": len(accepted), "rejected": len(rows) - len(accepted),
            "speakers": len(per_speaker), "duration_hours": round(sum(x["duration_ms"] for x in accepted) / 3_600_000, 2),
            "rejection_reasons": dict(reasons),
        }
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
