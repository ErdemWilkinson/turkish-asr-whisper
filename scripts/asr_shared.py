"""Front-end constants and helpers shared by the ASR scripts.

Training, quality filtering and evaluation must agree on the feature pipeline
and on how many CTC output steps a clip yields, so both live here instead of
being copied into every script.
"""
from __future__ import annotations

import numpy as np

RATE, N_MELS, HOP = 16_000, 40, 320
ALPHABET = " abcçdefgğhıijklmnoöpqrsştuüvwxyz'"
# The model's first layer is a stride-2 Conv1D: one output step per two frames.
CONV_STRIDE = 2
# power_to_db(ref=max) floors at -80 dB, i.e. "silence". Padding with 0 would
# mean "as loud as the loudest frame in the clip" and the backward GRU would
# read that before the real audio.
PAD_DB = -80.0


def extract_features(path: str) -> np.ndarray:
    """Log-mel spectrogram, shape (frames, N_MELS)."""
    import librosa  # imported lazily so the lightweight filter scripts stay cheap

    audio, _ = librosa.load(path, sr=RATE, mono=True)
    mel = librosa.feature.melspectrogram(y=audio, sr=RATE, n_fft=640, hop_length=HOP,
                                         n_mels=N_MELS, power=2.0)
    return librosa.power_to_db(mel, ref=np.max).T.astype(np.float32)


def pad_features(items: list[np.ndarray]) -> np.ndarray:
    """Stack variable-length (frames, N_MELS) arrays, padding with silence."""
    output = np.full((len(items), max(item.shape[0] for item in items), N_MELS), PAD_DB, dtype=np.float32)
    for index, item in enumerate(items):
        output[index, : item.shape[0]] = item
    return output


def clip_seconds(row: dict) -> float | None:
    """Duration of a manifest row's audio, from metadata when it is there."""
    if row.get("duration_ms"):
        return row["duration_ms"] / 1000
    try:
        import librosa

        return float(librosa.get_duration(path=row["audio_path"]))
    except Exception:
        return None


def ctc_feasible(text: str, seconds: float, margin: int = 2) -> bool:
    """Can CTC align `text` to a clip of this length?

    CTC needs one output step per character plus a blank between each pair of
    identical neighbours. Keras' ctc_batch_cost cannot score a clip that has
    fewer steps than that. `margin` absorbs resampling rounding.
    """
    frames = 1 + int(seconds * RATE) // HOP  # librosa pads both ends (center=True)
    steps = (frames + CONV_STRIDE - 1) // CONV_STRIDE
    needed = len(text) + sum(a == b for a, b in zip(text, text[1:]))
    return needed + margin <= steps
