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


def features(relative_path: str) -> np.ndarray:
    audio, _ = librosa.load(ROOT / relative_path, sr=SAMPLE_RATE, mono=True)
    audio = np.pad(audio[:SAMPLES], (0, max(0, SAMPLES - len(audio))))
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC,
                                n_fft=640, hop_length=320, n_mels=40)
    mfcc = np.pad(mfcc[:, :FRAMES], ((0, 0), (0, max(0, FRAMES - mfcc.shape[1]))))
    return mfcc.T.astype(np.float32)[..., np.newaxis]


def main() -> None:
    if not MANIFEST.exists():
        raise SystemExit("Run prepare_dataset.py before training.")
    with MANIFEST.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len({row["speaker"] for row in rows}) < 2:
        raise SystemExit("At least two speakers are needed for a speaker-disjoint validation split.")

    labels = sorted({row["label"] for row in rows})
    label_to_id = {label: index for index, label in enumerate(labels)}
    x = np.stack([features(row["path"]) for row in rows])
    y = np.array([label_to_id[row["label"]] for row in rows], dtype=np.int32)
    groups = np.array([row["speaker"] for row in rows])
    train_idx, val_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
                              .split(x, y, groups))
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
    model.fit(x[train_idx], y[train_idx], validation_data=(x[val_idx], y[val_idx]),
              epochs=60, batch_size=32, callbacks=callbacks, verbose=2)

    pred = model.predict(x[val_idx], verbose=0).argmax(axis=1)
    whisper = np.array([rows[i]["style"] == "whisper" for i in val_idx])
    val_accuracy = float((pred == y[val_idx]).mean())
    whisper_accuracy = float((pred[whisper] == y[val_idx][whisper]).mean()) if whisper.any() else None
    ARTIFACTS.mkdir(exist_ok=True)
    model.save(ARTIFACTS / "command_model.keras")
    np.save(ARTIFACTS / "representative_features.npy", x[train_idx][:200])
    (ARTIFACTS / "labels.json").write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
    report = {"validation_accuracy": val_accuracy, "whisper_validation_accuracy": whisper_accuracy,
              "validation_samples": int(len(val_idx)), "whisper_validation_samples": int(whisper.sum())}
    (ARTIFACTS / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
