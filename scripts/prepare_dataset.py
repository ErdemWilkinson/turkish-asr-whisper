"""Validate Turkish command recordings and write a reproducible manifest."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
SPEC = json.loads((ROOT / "commands.v1.json").read_text(encoding="utf-8"))
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "manifests" / "samples.csv"


def main() -> None:
    if not RAW.exists():
        raise SystemExit(f"No recordings found: {RAW}")
    labels, styles = set(SPEC["labels"]), set(SPEC["styles"])
    rows: list[dict[str, str | int]] = []
    for wav in sorted(RAW.glob("*/*/*/*.wav")):
        speaker, style, label = wav.parts[-4:-1]
        if style not in styles or label not in labels:
            print(f"SKIP invalid path label/style: {wav}")
            continue
        info = sf.info(wav)
        if info.channels != 1 or info.samplerate != SPEC["sample_rate_hz"]:
            print(f"SKIP expected mono {SPEC['sample_rate_hz']} Hz: {wav}")
            continue
        duration_ms = round(info.frames * 1000 / info.samplerate)
        if not 600 <= duration_ms <= 1400:
            print(f"SKIP expected 0.6–1.4 sec: {wav} ({duration_ms} ms)")
            continue
        rows.append({"path": str(wav.relative_to(ROOT)), "speaker": speaker,
                     "style": style, "label": label, "duration_ms": duration_ms})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "speaker", "style", "label", "duration_ms"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} valid recordings to {OUT}")


if __name__ == "__main__":
    main()
