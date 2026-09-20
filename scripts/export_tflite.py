"""Export the trained classifier as a full-int8 TensorFlow Lite model."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"


def representative_dataset():
    for row in np.load(ARTIFACTS / "representative_features.npy"):
        yield [row[np.newaxis, ...].astype(np.float32)]


def main() -> None:
    model_path = ARTIFACTS / "command_model.keras"
    if not model_path.exists():
        raise SystemExit("Train a model before exporting it.")
    converter = tf.lite.TFLiteConverter.from_keras_model(tf.keras.models.load_model(model_path))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    output = converter.convert()
    target = ARTIFACTS / "turkish_whisper_commands_int8.tflite"
    target.write_bytes(output)
    print(f"Wrote {target} ({len(output):,} bytes)")


if __name__ == "__main__":
    main()
