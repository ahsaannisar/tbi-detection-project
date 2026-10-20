"""Select a U-Net threshold on validation, then evaluate it once on test."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image


THRESHOLDS = np.arange(0.05, 0.56, 0.025)


def paths_for(split: str):
    image_dir = Path("segmentation-slices") / split / "images"
    mask_dir = Path("segmentation-slices") / split / "masks"
    images = sorted(image_dir.glob("*.png"))
    masks = [mask_dir / path.name for path in images]
    if not all(path.exists() for path in masks):
        raise RuntimeError(f"Image/mask mismatch in {split}")
    return images, masks


def evaluate(model, split: str, thresholds, batch_size: int = 12):
    images, masks = paths_for(split)
    totals = {float(t): {"tp": 0, "fp": 0, "fn": 0, "positive_slice_dice": []} for t in thresholds}
    for start in range(0, len(images), batch_size):
        image_batch = []
        mask_batch = []
        for image_path, mask_path in zip(images[start:start + batch_size], masks[start:start + batch_size]):
            with Image.open(image_path) as image:
                image_batch.append(np.asarray(image.convert("L").resize((256, 256), Image.Resampling.NEAREST), dtype=np.float32) / 255.0)
            with Image.open(mask_path) as mask:
                mask_batch.append(np.asarray(mask.convert("L").resize((256, 256), Image.Resampling.NEAREST)) > 0)
        x = np.asarray(image_batch)[..., None]
        y = np.asarray(mask_batch, dtype=bool)
        probability = model.predict(x, verbose=0)[..., 0]
        for threshold in thresholds:
            key = float(threshold)
            prediction = probability >= threshold
            tp_by_slice = np.sum(prediction & y, axis=(1, 2))
            fp_by_slice = np.sum(prediction & ~y, axis=(1, 2))
            fn_by_slice = np.sum(~prediction & y, axis=(1, 2))
            totals[key]["tp"] += int(tp_by_slice.sum())
            totals[key]["fp"] += int(fp_by_slice.sum())
            totals[key]["fn"] += int(fn_by_slice.sum())
            positive = np.sum(y, axis=(1, 2)) > 0
            denominators = 2 * tp_by_slice + fp_by_slice + fn_by_slice
            dice = np.divide(2 * tp_by_slice, denominators, out=np.zeros_like(denominators, dtype=float), where=denominators > 0)
            totals[key]["positive_slice_dice"].extend(dice[positive].tolist())
        if start % (batch_size * 20) == 0:
            print(split, start, "/", len(images))
    rows = []
    for threshold in thresholds:
        value = totals[float(threshold)]
        tp, fp, fn = value["tp"], value["fp"], value["fn"]
        rows.append({
            "threshold": float(threshold),
            "dice": float((2 * tp) / max(2 * tp + fp + fn, 1)),
            "iou": float(tp / max(tp + fp + fn, 1)),
            "precision": float(tp / max(tp + fp, 1)),
            "recall": float(tp / max(tp + fn, 1)),
            "mean_positive_slice_dice": float(np.mean(value["positive_slice_dice"])) if value["positive_slice_dice"] else 0.0,
        })
    return rows


def main():
    output = Path("experiments/unet_threshold_optimization")
    output.mkdir(parents=True, exist_ok=True)
    model = tf.keras.models.load_model("unet_best_v2.keras", compile=False)
    validation = evaluate(model, "validation", THRESHOLDS)
    selected = max(validation, key=lambda row: row["dice"])
    test = evaluate(model, "test", [selected["threshold"]])[0]
    with (output / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump({"selection_split": "validation", "validation_sweep": validation, "selected_threshold": selected["threshold"], "selected_validation_metrics": selected, "frozen_test_metrics": test}, handle, indent=2)
    print(json.dumps({"selected_validation_metrics": selected, "frozen_test_metrics": test}, indent=2))


if __name__ == "__main__":
    main()
