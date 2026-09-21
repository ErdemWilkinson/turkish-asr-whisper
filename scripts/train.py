"""Train a compact MFCC/CNN Turkish command classifier.

The validation split is speaker-disjoint, so a score is not inflated by the
same voice appearing in both training and validation recordings.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import librosa
import numpy as np
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifests" / "samples.csv"
ARTIFACTS = ROOT / "artifacts"
SAMPLE_RATE, SAMPLES, N_MFCC, FRAMES = 16_000, 16_000, 13, 49
WHISPER_GATE, NORMAL_GATE, NEGATIVE_FALSE_ACCEPT_GATE = 0.85, 0.92, 0.02


def features(relative_path: str) -> np.ndarray:
    audio, _ = librosa.load(ROOT / relative_path, sr=SAMPLE_RATE, mono=True)
    audio = np.pad(audio[:SAMPLES], (0, max(0, SAMPLES - len(audio))))
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC,
                                n_fft=640, hop_length=320, n_mels=40)
    mfcc = np.pad(mfcc[:, :FRAMES], ((0, 0), (0, max(0, FRAMES - mfcc.shape[1]))))
    return mfcc.T.astype(np.float32)[..., np.newaxis]


def accuracy_for_mask(pred: np.ndarray, expected: np.ndarray, mask: np.ndarray) -> float | None:
    return float((pred[mask] == expected[mask]).mean()) if mask.any() else None


def report_metrics(pred: np.ndarray, expected: np.ndarray, rows: list[dict], indices: np.ndarray,
                   labels: list[str]) -> dict:
    styles = np.array([rows[i]["style"] for i in indices])
    actual_labels = np.array([rows[i]["label"] for i in indices])
    metrics = {
        "accuracy": float((pred == expected).mean()),
        "samples": int(len(indices)),
        "style_accuracy": {style: accuracy_for_mask(pred, expected, styles == style)
                           for style in ("normal", "quiet", "whisper")},
        "per_label_recall": {},
    }
    for label_id, label in enumerate(labels):
        metrics["per_label_recall"][label] = accuracy_for_mask(pred, expected, actual_labels == label)
    negative = np.isin(actual_labels, ["unknown", "silence"])
    metrics["negative_samples"] = int(negative.sum())
    metrics["negative_false_accept_rate"] = (
        float((~np.isin(pred[negative], [labels.index("unknown"), labels.index("silence")])).mean())
        if negative.any() else None
    )
    return metrics


def main() -> None:
    if not MANIFEST.exists():
        raise SystemExit("Run prepare_dataset.py before training.")
    with MANIFEST.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len({row["speaker"] for row in rows}) < 3:
        raise SystemExit("At least three speakers are needed for speaker-disjoint train/dev/test splits.")

    labels = sorted({row["label"] for row in rows})
    label_to_id = {label: index for index, label in enumerate(labels)}
    x = np.stack([features(row["path"]) for row in rows])
    y = np.array([label_to_id[row["label"]] for row in rows], dtype=np.int32)
    groups = np.array([row["speaker"] for row in rows])
    train_dev_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
                                    .split(x, y, groups))
    train_idx_relative, dev_idx_relative = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=43)
                                                 .split(x[train_dev_idx], y[train_dev_idx], groups[train_dev_idx]))
    train_idx, dev_idx = train_dev_idx[train_idx_relative], train_dev_idx[dev_idx_relative]
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(FRAMES, N_MFCC, 1)),
        tf.keras.layers.Conv2D(12, 3, activation="relu", padding="same"),
        tf.keras.layers.MaxPool2D(2),
        tf.keras.layers.Conv2D(20, 3, activation="relu", padding="same"),
        tf.keras.layers.MaxPool2D(2),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(len(labels), activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8,
                                                    restore_best_weights=True)]
    model.fit(x[train_idx], y[train_idx], validation_data=(x[dev_idx], y[dev_idx]),
              epochs=60, batch_size=32, callbacks=callbacks, verbose=2)

    dev_metrics = report_metrics(model.predict(x[dev_idx], verbose=0).argmax(axis=1), y[dev_idx], rows, dev_idx, labels)
    test_metrics = report_metrics(model.predict(x[test_idx], verbose=0).argmax(axis=1), y[test_idx], rows, test_idx, labels)
    gates = {
        "whisper_accuracy_min": WHISPER_GATE,
        "normal_accuracy_min": NORMAL_GATE,
        "negative_false_accept_rate_max": NEGATIVE_FALSE_ACCEPT_GATE,
    }
    acceptance_passed = (
        test_metrics["style_accuracy"]["whisper"] is not None and test_metrics["style_accuracy"]["whisper"] >= WHISPER_GATE
        and test_metrics["style_accuracy"]["normal"] is not None and test_metrics["style_accuracy"]["normal"] >= NORMAL_GATE
        and test_metrics["negative_false_accept_rate"] is not None and test_metrics["negative_false_accept_rate"] <= NEGATIVE_FALSE_ACCEPT_GATE
    )
    ARTIFACTS.mkdir(exist_ok=True)
    model.save(ARTIFACTS / "command_model.keras")
    np.save(ARTIFACTS / "representative_features.npy", x[train_idx][:200])
    (ARTIFACTS / "labels.json").write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
    report = {"split": {"train_samples": int(len(train_idx)), "dev_samples": int(len(dev_idx)), "test_samples": int(len(test_idx)),
                        "train_speakers": sorted(set(groups[train_idx])), "dev_speakers": sorted(set(groups[dev_idx])),
                        "test_speakers": sorted(set(groups[test_idx]))},
              "dev": dev_metrics, "held_out_test": test_metrics, "acceptance_gates": gates,
              "acceptance_passed": acceptance_passed}
    (ARTIFACTS / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
