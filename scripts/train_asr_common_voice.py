"""Train a streaming Turkish character-level CTC baseline from Common Voice.

This is an offline, CPU-friendly training path: audio is loaded per batch, so
the full corpus is never padded into RAM.  It deliberately preserves Mozilla's
official train/dev split.  The resulting model is a research baseline, not an
ESP32-P4 deployment artifact.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import librosa
import numpy as np
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]
RATE, N_MELS, HOP = 16_000, 40, 320
ALPHABET = " abcçdefgğhıijklmnoöpqrsştuüvwxyz'"
CHAR_TO_ID = {char: index for index, char in enumerate(ALPHABET)}


def normalise(text: str) -> str:
    text = text.lower().replace("i̇", "i")
    text = re.sub(r"[^a-zçğıöşü' ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_rows(path: Path, limit: int | None, seed: int) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        raw_text = row["transcript"]
        # A handful of malformed TSV records contain subsequent columns in the
        # sentence field. They are not trustworthy speech/transcript pairs.
        if "\t" in raw_text or "\n" in raw_text or "\r" in raw_text:
            continue
        text = normalise(raw_text)
        if text and len(text) <= 300 and Path(row["audio_path"]).is_file():
            row["transcript"] = text
            rows.append(row)
    random.Random(seed).shuffle(rows)
    return rows if limit is None else rows[:limit]


def extract(path: str) -> np.ndarray:
    audio, _ = librosa.load(path, sr=RATE, mono=True)
    mel = librosa.feature.melspectrogram(
        y=audio, sr=RATE, n_fft=640, hop_length=HOP, n_mels=N_MELS, power=2.0
    )
    return librosa.power_to_db(mel, ref=np.max).T.astype(np.float32)


class BatchSequence(tf.keras.utils.Sequence):
    def __init__(self, rows: list[dict], batch_size: int, shuffle: bool):
        self.rows, self.batch_size, self.shuffle = rows, batch_size, shuffle
        self.indices = np.arange(len(rows))
        self.on_epoch_end()

    def __len__(self) -> int:
        return len(self.rows) // self.batch_size

    def on_epoch_end(self) -> None:
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, index: int):
        selected = [self.rows[i] for i in self.indices[index * self.batch_size:(index + 1) * self.batch_size]]
        features = [extract(row["audio_path"]) for row in selected]
        labels = [[CHAR_TO_ID[c] for c in row["transcript"]] for row in selected]
        max_frames, max_labels = max(x.shape[0] for x in features), max(len(y) for y in labels)
        x = np.zeros((len(selected), max_frames, N_MELS), dtype=np.float32)
        y = np.zeros((len(selected), max_labels), dtype=np.int32)
        input_len = np.zeros((len(selected), 1), dtype=np.int32)
        label_len = np.zeros((len(selected), 1), dtype=np.int32)
        for i, (feature, label) in enumerate(zip(features, labels)):
            x[i, :feature.shape[0]] = feature
            y[i, :len(label)] = label
            input_len[i, 0] = (feature.shape[0] + 1) // 2
            label_len[i, 0] = len(label)
        return {"features": x, "labels": y, "input_len": input_len, "label_len": label_len}, None


class CtcLoss(tf.keras.layers.Layer):
    def call(self, inputs):
        labels, logits, input_len, label_len = inputs
        self.add_loss(tf.keras.backend.ctc_batch_cost(labels, logits, input_len, label_len))
        return logits


def build_models() -> tuple[tf.keras.Model, tf.keras.Model]:
    features = tf.keras.Input(shape=(None, N_MELS), name="features")
    labels = tf.keras.Input(shape=(None,), dtype="int32", name="labels")
    input_len = tf.keras.Input(shape=(1,), dtype="int32", name="input_len")
    label_len = tf.keras.Input(shape=(1,), dtype="int32", name="label_len")
    x = tf.keras.layers.Conv1D(48, 5, strides=2, padding="same", activation="relu")(features)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, return_sequences=True))(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    logits = tf.keras.layers.Dense(len(ALPHABET) + 1, activation="softmax", name="characters")(x)
    training = tf.keras.Model([features, labels, input_len, label_len], CtcLoss()([labels, logits, input_len, label_len]))
    inference = tf.keras.Model(features, logits)
    training.compile(optimizer=tf.keras.optimizers.Adam(1e-3))
    return training, inference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "common_voice_scripted")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "asr_common_voice_stage1")
    parser.add_argument("--train-limit", type=int, default=6000)
    parser.add_argument("--dev-limit", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warm-start", type=Path, help="Existing inference.keras to continue from")
    args = parser.parse_args()
    tf.keras.utils.set_random_seed(args.seed)
    train_rows = load_rows(args.data / "train.jsonl", args.train_limit, args.seed)
    dev_rows = load_rows(args.data / "dev.jsonl", args.dev_limit, args.seed + 1)
    if len(train_rows) < args.batch_size or len(dev_rows) < args.batch_size:
        raise SystemExit("Not enough valid rows for the requested batch size.")
    args.output.mkdir(parents=True, exist_ok=True)
    training, inference = build_models()
    if args.warm_start:
        warm = tf.keras.models.load_model(args.warm_start, compile=False)
        inference.set_weights(warm.get_weights())
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        tf.keras.callbacks.CSVLogger(str(args.output / "history.csv")),
        tf.keras.callbacks.ModelCheckpoint(str(args.output / "best_training.keras"), monitor="val_loss", save_best_only=True),
    ]
    history = training.fit(
        BatchSequence(train_rows, args.batch_size, shuffle=True),
        validation_data=BatchSequence(dev_rows, args.batch_size, shuffle=False),
        epochs=args.epochs, callbacks=callbacks, verbose=2,
    )
    inference.save(args.output / "inference.keras")
    (args.output / "metrics.json").write_text(json.dumps({
        "train_rows": len(train_rows), "dev_rows": len(dev_rows),
        "epochs_requested": args.epochs, "epochs_completed": len(history.history["loss"]),
        "loss": [float(x) for x in history.history["loss"]],
        "val_loss": [float(x) for x in history.history["val_loss"]],
        "alphabet": ALPHABET,
        "status": "baseline_trained_not_device_ready",
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
