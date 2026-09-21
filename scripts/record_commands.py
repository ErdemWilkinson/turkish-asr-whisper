"""Interactive recorder for the command-classifier dataset (commands.v1.json).

Walks you through recording every (label, style) combination this project's
deployment-target model needs, at the exact format prepare_dataset.py
expects: mono 16kHz WAV files under

    data/raw/<speaker_id>/<style>/<label>/<take>.wav

Usage:
    python scripts/record_commands.py --speaker erdem
    python scripts/record_commands.py --speaker erdem --resume   # skip already-recorded takes

For each label you'll be shown the Turkish phrase to say (from
commands.v1.json's "phrases"), prompted per style (normal/quiet/whisper),
and asked to record `minimum_samples_per_label_and_style` (currently 80)
takes. After each take you can keep it, redo it, or skip ahead. `unknown`
and `silence` have no fixed phrase -- see the prompts printed for those.

This is a single-speaker session driver. Repeat with a different
--speaker for each person (the acceptance gate in voice/README.md needs
speaker-disjoint train/dev/test splits, so more distinct speakers matters
more than more takes from one person).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
SPEC = json.loads((ROOT / "commands.v1.json").read_text(encoding="utf-8"))
SAMPLE_RATE = SPEC["sample_rate_hz"]
CLIP_MS = SPEC["clip_duration_ms"]
MIN_TAKES = SPEC["minimum_samples_per_label_and_style"]

# unknown/silence have no fixed phrase in commands.v1.json's "phrases" map --
# unknown needs varied non-command speech (so the classifier learns to
# reject it), silence needs literal silence/room tone.
PROMPT_OVERRIDES = {
    "unknown": "(komut olmayan rastgele bir Türkçe cümle söyleyin, her seferinde farklı)",
    "silence": "(konuşmayın -- sadece oda sesi/sessizlik kaydedin)",
}


def record_take(duration_s: float) -> np.ndarray:
    print(f"  Recording in", end="", flush=True)
    for n in (3, 2, 1):
        print(f" {n}...", end="", flush=True)
        sd.sleep(400)
    print(" GO", flush=True)
    audio = sd.rec(int(duration_s * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                    channels=1, dtype="int16")
    sd.wait()
    return audio


def next_take_index(label_dir: Path) -> int:
    """Return a new numeric filename, rejecting ambiguous existing files."""
    used = set()
    for path in label_dir.glob("*.wav"):
        if not path.stem.isdigit():
            raise SystemExit(f"Unexpected recording name {path}; rename it before --resume.")
        used.add(int(path.stem))
    return max(used, default=-1) + 1


def record_label_style(speaker: str, label: str, style: str, resume: bool, replace: bool) -> None:
    phrase = SPEC["phrases"].get(label)
    prompt = PROMPT_OVERRIDES.get(label, f'"{phrase[0]}"' if phrase else "(bilinmeyen)")
    out_dir = ROOT / "data" / "raw" / speaker / style / label
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = list(out_dir.glob("*.wav"))
    if existing and not resume and not replace:
        raise SystemExit(f"{out_dir} already contains recordings; use --resume or explicit --replace.")
    if replace:
        for path in existing:
            path.unlink()
    start = next_take_index(out_dir) if resume else 0
    completed = len(list(out_dir.glob("*.wav")))
    if completed >= MIN_TAKES:
        print(f"[{label}/{style}] already has {completed} takes (>= {MIN_TAKES}), skipping")
        return

    print(f"\n=== {label} / {style} ({completed}/{MIN_TAKES} done) ===")
    print(f"Say: {prompt}")
    if style == "whisper":
        print("  (WHISPER volume -- barely audible, as if not to wake someone)")
    elif style == "quiet":
        print("  (quiet indoor voice, not full volume)")

    i = start
    while completed < MIN_TAKES:
        input(f"  Take {completed + 1}/{MIN_TAKES} -- press Enter to start recording "
              f"(Ctrl+C to stop this session): ")
        audio = record_take(CLIP_MS / 1000.0)
        peak = int(np.abs(audio).max())
        if peak < 50 and label != "silence":
            print(f"  WARNING: recording is nearly silent (peak={peak}) -- likely a bad take")
        choice = input("  [k]eep / [r]edo / [s]kip this take? [k] ").strip().lower() or "k"
        if choice == "r":
            continue
        if choice == "s":
            break
        path = out_dir / f"{i:03d}.wav"
        sf.write(path, audio, SAMPLE_RATE, subtype="PCM_16")
        print(f"  saved {path.relative_to(ROOT)} (peak={peak})")
        i += 1
        completed += 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speaker", required=True, help="speaker id, e.g. 'erdem' -- must be a valid folder name")
    parser.add_argument("--resume", action="store_true",
                         help="skip takes already recorded for a label/style instead of starting over")
    parser.add_argument("--replace", action="store_true",
                        help="delete existing WAV takes for selected label/styles before recording")
    parser.add_argument("--labels", nargs="*", default=None,
                         help="only record these labels (default: all in commands.v1.json)")
    parser.add_argument("--styles", nargs="*", default=None,
                         help="only record these styles (default: all in commands.v1.json)")
    args = parser.parse_args()

    devices = sd.query_devices()
    default_in = sd.default.device[0]
    print(f"Using input device: {devices[default_in]['name'] if default_in is not None else '(system default)'}")
    print(f"Format: mono, {SAMPLE_RATE} Hz, {CLIP_MS} ms per clip, {MIN_TAKES} takes per label/style\n")

    labels = args.labels or SPEC["labels"]
    styles = args.styles or SPEC["styles"]

    try:
        for label in labels:
            if label not in SPEC["labels"]:
                print(f"Unknown label {label!r}, skipping", file=sys.stderr)
                continue
            for style in styles:
                if style not in SPEC["styles"]:
                    print(f"Unknown style {style!r}, skipping", file=sys.stderr)
                    continue
                record_label_style(args.speaker, label, style, args.resume, args.replace)
    except KeyboardInterrupt:
        print("\nSession stopped early -- already-saved takes are kept. Re-run with --resume to continue.")
        return

    print("\nAll requested label/style combinations done for this speaker.")
    print("Run scripts/prepare_dataset.py next to validate and build the manifest.")


if __name__ == "__main__":
    main()
