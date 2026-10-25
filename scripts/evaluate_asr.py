"""Measure a CTC ASR inference model with WER and CER on held-out data."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf

# CtcLoss is a custom tf.keras.layers.Layer subclass baked into any .keras
# file saved as a *training* model (train_asr_common_voice.py's own
# --warm-start path already knows this, passing the same custom_objects
# dict below). Without it, tf.keras.models.load_model() on a training
# checkpoint (e.g. a best_training.keras saved mid-run) fails with
# "TypeError: Cannot deserialize object of type 'CtcLoss'" -- see
# KNOWN_ISSUES.md's Round 18 entry. An already-exported erdem_asr.keras
# doesn't contain a CtcLoss layer, so this import/argument is a no-op for
# that case and only matters for evaluating an interrupted run's checkpoint.
from asr_shared import ALPHABET, extract_features, pad_features
from text_norm import normalise
from train_asr_common_voice import CtcLoss

ROOT = Path(__file__).resolve().parents[1]


def distance(left: list[str], right: list[str]) -> int:
    previous = list(range(len(right) + 1))
    for i, token in enumerate(left, 1):
        current = [i]
        for j, other in enumerate(right, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (token != other)))
        previous = current
    return previous[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path,
                        default=ROOT / "artifacts" / "asr_common_voice_stage0" / "erdem_asr.keras")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "common_voice_quality_v1" / "test.jsonl")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts" / "asr_common_voice_stage0" / "evaluation.json")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.data.read_text(encoding="utf-8").splitlines()]
    rows = [row for row in rows if normalise(row["transcript"]) and Path(row["audio_path"]).is_file()]
    random.Random(args.seed).shuffle(rows)
    rows = rows[:args.limit]
    if not rows:
        raise SystemExit("No usable test rows.")
    model = tf.keras.models.load_model(args.model, compile=False, custom_objects={"CtcLoss": CtcLoss})
    char_errors = char_total = word_errors = word_total = 0
    examples = []
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start:start + args.batch_size]
        features = [extract_features(row["audio_path"]) for row in batch]
        inputs = pad_features(features)
        lengths = np.array([(item.shape[0] + 1) // 2 for item in features], dtype=np.int32)
        probabilities = model.predict(inputs, verbose=0)
        decoded, _ = tf.keras.backend.ctc_decode(probabilities, lengths, greedy=True)
        for row, tokens in zip(batch, decoded[0].numpy()):
            predicted = "".join(ALPHABET[token] for token in tokens if token >= 0).strip()
            reference = normalise(row["transcript"])
            char_errors += distance(list(reference), list(predicted))
            char_total += len(reference)
            word_errors += distance(reference.split(), predicted.split())
            word_total += len(reference.split())
            if len(examples) < 10:
                examples.append({"reference": reference, "prediction": predicted})
        print(f"Evaluated {min(start + len(batch), len(rows))}/{len(rows)}")
    report = {
        "model": str(args.model), "test_data": str(args.data), "samples": len(rows),
        "cer": char_errors / char_total if char_total else None,
        "wer": word_errors / word_total if word_total else None,
        "character_accuracy": 1 - (char_errors / char_total) if char_total else None,
        "word_accuracy": 1 - (word_errors / word_total) if word_total else None,
        "examples": examples,
        "interpretation": "CTC baseline evaluation; this is not the ESP32 command-model acceptance metric.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("samples", "cer", "wer", "character_accuracy", "word_accuracy")}, indent=2))


if __name__ == "__main__":
    main()
