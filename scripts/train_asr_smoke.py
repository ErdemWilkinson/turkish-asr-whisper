"""Train a deliberately small CTC ASR baseline from paired Turkish audio/text.

This validates the local ASR path. It is not a deployable ESP32-P4 model and
must not be mistaken for the labelled command recognizer.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import librosa
import numpy as np
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "common_voice_spontaneous" / "manifest.jsonl"
ARTIFACTS = ROOT / "artifacts" / "asr_smoke"
RATE, N_MELS, HOP = 16_000, 40, 320
# Q/W/X loanwords and proper names occur in the community transcripts.
ALPHABET = " abcçdefgğhıijklmnoöpqrsştuüvwxyz'"
# CTC reserves its final class for blank. Padding zeros are ignored through
# label_len, so real characters use 0..N-1 and blank is N.
CHAR_TO_ID = {char: index for index, char in enumerate(ALPHABET)}


def normalise(text: str) -> str:
    text = text.lower().replace("i̇", "i")
    text = re.sub(r"[^a-zçğıöşü' ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def feature(path: str) -> np.ndarray:
    audio, _ = librosa.load(path, sr=RATE, mono=True)
    mel = librosa.feature.melspectrogram(y=audio, sr=RATE, n_fft=640,
                                         hop_length=HOP, n_mels=N_MELS, power=2.0)
    return librosa.power_to_db(mel, ref=np.max).T.astype(np.float32)


def pad_features(items: list[np.ndarray]) -> np.ndarray:
    longest = max(item.shape[0] for item in items)
    output = np.zeros((len(items), longest, N_MELS), dtype=np.float32)
    for index, item in enumerate(items):
        output[index, : item.shape[0]] = item
    return output


def pad_labels(items: list[list[int]]) -> np.ndarray:
    longest = max(len(item) for item in items)
    output = np.zeros((len(items), longest), dtype=np.int32)
    for index, item in enumerate(items):
        output[index, : len(item)] = item
    return output


class CtcLoss(tf.keras.layers.Layer):
    def call(self, inputs: list[tf.Tensor]) -> tf.Tensor:
        labels, logits, input_len, label_len = inputs
        loss = tf.keras.backend.ctc_batch_cost(labels, logits, input_len, label_len)
        self.add_loss(loss)
        return logits


def main() -> None:
    rows = [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines()]
    rows = [row for row in rows if normalise(row["transcript"]) and row.get("speaker_id")]
    if len(rows) < 10:
        raise SystemExit("Need at least ten paired samples with speaker identifiers.")
    x_items = [feature(row["audio_path"]) for row in rows]
    y_items = [[CHAR_TO_ID[char] for char in normalise(row["transcript"])] for row in rows]
    x, y = pad_features(x_items), pad_labels(y_items)
    input_len = np.array([[item.shape[0] // 2] for item in x_items], dtype=np.int32)
    label_len = np.array([[len(item)] for item in y_items], dtype=np.int32)
    speakers = np.array([row["speaker_id"] for row in rows])
    train_idx, val_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
                              .split(x, groups=speakers))

    features_in = tf.keras.Input(shape=(None, N_MELS), name="features")
    labels_in = tf.keras.Input(shape=(None,), dtype="int32", name="labels")
    input_len_in = tf.keras.Input(shape=(1,), dtype="int32", name="input_len")
    label_len_in = tf.keras.Input(shape=(1,), dtype="int32", name="label_len")
    hidden = tf.keras.layers.Conv1D(24, 5, strides=2, padding="same", activation="relu")(features_in)
    hidden = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32, return_sequences=True))(hidden)
    logits = tf.keras.layers.Dense(len(CHAR_TO_ID) + 1, activation="softmax", name="characters")(hidden)
    output = CtcLoss()([labels_in, logits, input_len_in, label_len_in])
    model = tf.keras.Model([features_in, labels_in, input_len_in, label_len_in], output)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3))
    callbacks = [tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=4, restore_best_weights=True
    )]
    history = model.fit(
        [x[train_idx], y[train_idx], input_len[train_idx], label_len[train_idx]],
        epochs=20, batch_size=4, verbose=2, callbacks=callbacks,
        validation_data=([x[val_idx], y[val_idx], input_len[val_idx], label_len[val_idx]], None),
    )
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    model.save(ARTIFACTS / "ctc_smoke.keras")
    (ARTIFACTS / "metrics.json").write_text(json.dumps({
        "samples": len(rows), "train_samples": len(train_idx), "validation_samples": len(val_idx),
        "train_speakers": len(set(speakers[train_idx])), "validation_speakers": len(set(speakers[val_idx])),
        "loss": [float(value) for value in history.history["loss"]],
        "val_loss": [float(value) for value in history.history["val_loss"]],
        "status": "smoke_baseline_not_deployable",
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
