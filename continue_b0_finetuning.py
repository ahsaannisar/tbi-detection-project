"""Conservatively continue the existing production B0 using validation macro-F1.

No test files are read. The initial production weights are themselves a
candidate, preventing continuation from replacing them with a worse epoch.
"""

import json
from pathlib import Path

import tensorflow as tf

from classification_experiments import (
    ImageSequence,
    ValidationMacroF1,
    calculate_metrics,
    read_items,
)


def main():
    output = Path("experiments/exp09_continue_existing_b0")
    output.mkdir(parents=True, exist_ok=True)
    train_items = read_items(Path("tbi-split"), "train", originals_only=False)
    validation_items = read_items(Path("tbi-split"), "validation", originals_only=True)
    validation_labels = [label for _, label in validation_items]
    train_sequence = ImageSequence(train_items, 224, 16, "raw", True, True, "nearest")
    validation_sequence = ImageSequence(validation_items, 224, 16, "raw", False, False, "nearest")

    model = tf.keras.models.load_model("tbi_efficientnetb0_best.keras", compile=False)
    base = model.get_layer("efficientnetb0")
    base.trainable = True
    for layer in base.layers[:-50]:
        layer.trainable = False

    initial_probabilities = model.predict(validation_sequence, verbose=2)
    initial_metrics = calculate_metrics(validation_labels, initial_probabilities)
    model.save_weights(output / "best.weights.h5")
    with (output / "best_validation_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump({"epoch": 0, **initial_metrics}, handle, indent=2)

    callback = ValidationMacroF1(validation_sequence, validation_labels, output, patience=2)
    callback.best = initial_metrics["macro_f1"]
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-6, clipnorm=1.0),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    history = model.fit(train_sequence, epochs=8, callbacks=[callback], verbose=2)
    model.load_weights(output / "best.weights.h5")
    model.save(output / "best_model.keras")
    probabilities = model.predict(validation_sequence, verbose=0)
    metrics = calculate_metrics(validation_labels, probabilities)
    with (output / "final_validation_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    with (output / "history.json").open("w", encoding="utf-8") as handle:
        json.dump(history.history, handle, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
