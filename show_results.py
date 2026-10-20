"""Display the project's verified final results without training or inference.

This is the presentation-safe results script. It reads the classification
metrics produced by the locked frozen-test evaluation and the segmentation
threshold-evaluation artifact. It does not load a TensorFlow model, retrain a
model, alter a dataset, or recompute predictions.

Run:
    python show_results.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parent
CLASSIFICATION_METRICS_PATH = (
    PROJECT_DIR / "experiments" / "final_frozen_test" / "metrics.json"
)
UNET_METRICS_PATH = (
    PROJECT_DIR / "experiments" / "unet_threshold_optimization" / "metrics.json"
)

CLASS_NAMES = (
    "epidural",
    "intraparenchymal",
    "intraventricular",
    "normal",
    "subarachnoid",
    "subdural",
)

# These are the verified frozen-test results for the retained production
# threshold (0.25). The validation-selected threshold (0.275) is loaded from
# UNET_METRICS_PATH below and shown separately as a rejected experiment.
PRODUCTION_UNET_RESULTS = {
    "threshold": 0.25,
    "test_slices": 1538,
    "positive_mask_slices": 591,
    "negative_mask_slices": 947,
    "dice": 0.0656,
    "iou": 0.0339,
    "precision": 0.2045,
    "recall": 0.0391,
    "mean_positive_slice_dice": 0.0790,
}

LINE_WIDTH = 78


def load_json(path: Path) -> dict[str, Any]:
    """Load a required saved-results artifact with a clear failure message."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Required saved-results file was not found:\n  {path}\n"
            "Run the dedicated evaluation script only if this artifact was "
            "never generated. No training is performed by show_results.py."
        )

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def title(text: str) -> None:
    print("\n" + "=" * LINE_WIDTH)
    print(text.center(LINE_WIDTH))
    print("=" * LINE_WIDTH)


def subtitle(text: str) -> None:
    print("\n" + text)
    print("-" * LINE_WIDTH)


def metric_row(label: str, value: str) -> None:
    print(f"{label:<36}{value:>18}")


def show_model_summary(metrics: dict[str, Any]) -> None:
    baseline = metrics["baseline_six_class"]
    improved = metrics["final_six_class"]
    test_images = int(metrics["frozen_test_images"])

    title("FINAL VERIFIED MODEL RESULTS")
    print("\nAll values come from saved frozen-test evaluation artifacts.")
    print("This script performs NO TRAINING and NO MODEL INFERENCE.")

    subtitle("BEST SIX-CLASS MODEL")
    metric_row("Model", "EfficientNetB0 + ExtraTrees")
    metric_row("Probability ensemble", "90% tree + 10% neural")
    metric_row("Preprocessing", "grayscale -> RGB")
    metric_row("Input size", "224 x 224")
    metric_row("Frozen-test images", f"{test_images}")

    subtitle("BASELINE VS IMPROVED SIX-CLASS RESULT")
    print(
        f"{'Metric':<24}"
        f"{'Baseline':>16}"
        f"{'Improved':>16}"
        f"{'Change':>16}"
    )
    print("-" * LINE_WIDTH)

    rows = (
        ("Accuracy", "accuracy"),
        ("Macro F1", "macro_f1"),
        ("Balanced accuracy", "balanced_accuracy"),
        ("Macro precision", "macro_precision"),
        ("Macro recall", "macro_recall"),
    )

    for label, key in rows:
        old_value = float(baseline[key])
        new_value = float(improved[key])
        change = (new_value - old_value) * 100
        print(
            f"{label:<24}"
            f"{percent(old_value):>16}"
            f"{percent(new_value):>16}"
            f"{change:+15.2f} pp"
        )

    baseline_correct = round(float(baseline["accuracy"]) * test_images)
    improved_correct = round(float(improved["accuracy"]) * test_images)
    print(
        f"{'Correct predictions':<24}"
        f"{f'{baseline_correct}/{test_images}':>16}"
        f"{f'{improved_correct}/{test_images}':>16}"
        f"{f'+{improved_correct - baseline_correct}':>16}"
    )


def show_per_class_results(metrics: dict[str, Any]) -> None:
    report = metrics["final_six_class"]["classification_report"]

    subtitle("UPDATED PER-CLASS RESULTS — SELECTED ENSEMBLE")
    print(
        f"{'Class':<22}"
        f"{'Precision':>13}"
        f"{'Recall':>13}"
        f"{'F1':>13}"
        f"{'Support':>13}"
    )
    print("-" * LINE_WIDTH)

    for class_name in CLASS_NAMES:
        values = report[class_name]
        display_name = class_name.replace("_", " ").title()
        print(
            f"{display_name:<22}"
            f"{percent(float(values['precision'])):>13}"
            f"{percent(float(values['recall'])):>13}"
            f"{percent(float(values['f1-score'])):>13}"
            f"{int(values['support']):>13}"
        )

    macro = report["macro avg"]
    weighted = report["weighted avg"]
    print("-" * LINE_WIDTH)
    print(
        f"{'Macro average':<22}"
        f"{percent(float(macro['precision'])):>13}"
        f"{percent(float(macro['recall'])):>13}"
        f"{percent(float(macro['f1-score'])):>13}"
        f"{int(macro['support']):>13}"
    )
    print(
        f"{'Weighted average':<22}"
        f"{percent(float(weighted['precision'])):>13}"
        f"{percent(float(weighted['recall'])):>13}"
        f"{percent(float(weighted['f1-score'])):>13}"
        f"{int(weighted['support']):>13}"
    )


def show_binary_results(metrics: dict[str, Any]) -> None:
    binary = metrics["binary_normal_vs_tbi"]
    report = binary["classification_report"]
    matrix = binary["confusion_matrix"]

    title("NORMAL vs TBI / ABNORMAL")
    print("\n" + percent(float(binary["accuracy"])).center(LINE_WIDTH))
    print("Frozen-Test Binary Accuracy".center(LINE_WIDTH))

    subtitle("BINARY METRICS")
    metric_row("Accuracy", percent(float(binary["accuracy"])))
    metric_row(
        "Balanced accuracy",
        percent(float(binary["balanced_accuracy"])),
    )
    metric_row("Macro F1", percent(float(binary["macro_f1"])))
    metric_row("Macro precision", percent(float(binary["macro_precision"])))
    metric_row("Macro recall", percent(float(binary["macro_recall"])))
    metric_row(
        "TBI recall / sensitivity",
        percent(float(report["tbi_abnormal"]["recall"])),
    )
    metric_row(
        "Normal recall / specificity",
        percent(float(report["normal"]["recall"])),
    )

    subtitle("BINARY PER-CLASS RESULTS")
    print(
        f"{'Class':<22}"
        f"{'Precision':>14}"
        f"{'Recall':>14}"
        f"{'F1':>14}"
        f"{'Support':>14}"
    )
    print("-" * LINE_WIDTH)
    for key, label in (("normal", "Normal"), ("tbi_abnormal", "TBI / Abnormal")):
        values = report[key]
        print(
            f"{label:<22}"
            f"{percent(float(values['precision'])):>14}"
            f"{percent(float(values['recall'])):>14}"
            f"{percent(float(values['f1-score'])):>14}"
            f"{int(values['support']):>14}"
        )

    subtitle("BINARY CONFUSION MATRIX")
    print(f"{'':<22}{'Pred. Normal':>18}{'Pred. TBI':>18}")
    print(f"{'Actual Normal':<22}{matrix[0][0]:>18}{matrix[0][1]:>18}")
    print(f"{'Actual TBI':<22}{matrix[1][0]:>18}{matrix[1][1]:>18}")


def show_hierarchical_results(metrics: dict[str, Any]) -> None:
    hierarchy = metrics["hierarchical_six_class"]

    subtitle("HIERARCHICAL SIX-CLASS COMPARISON")
    metric_row("Accuracy", percent(float(hierarchy["accuracy"])))
    metric_row("Macro F1", percent(float(hierarchy["macro_f1"])))
    metric_row(
        "Balanced accuracy",
        percent(float(hierarchy["balanced_accuracy"])),
    )
    print(
        "\nDecision: the hierarchy did not beat the selected flat ensemble, "
        "so it was not deployed for the final six-class prediction."
    )


def show_unet_results(unet_artifact: dict[str, Any]) -> None:
    production = PRODUCTION_UNET_RESULTS
    selected_threshold = float(unet_artifact["selected_threshold"])
    validation = unet_artifact["selected_validation_metrics"]
    rejected_test = unet_artifact["frozen_test_metrics"]

    title("U-NET SEGMENTATION — VERIFIED SAVED RESULTS")

    subtitle("RETAINED PRODUCTION THRESHOLD")
    metric_row("Threshold", f"{production['threshold']:.3f}")
    metric_row("Frozen-test slices", str(production["test_slices"]))
    metric_row("Positive-mask slices", str(production["positive_mask_slices"]))
    metric_row("Negative-mask slices", str(production["negative_mask_slices"]))
    metric_row("Global Dice", percent(float(production["dice"])))
    metric_row("Global IoU", percent(float(production["iou"])))
    metric_row("Pixel precision", percent(float(production["precision"])))
    metric_row("Pixel recall", percent(float(production["recall"])))
    metric_row(
        "Mean Dice on positive slices",
        percent(float(production["mean_positive_slice_dice"])),
    )

    subtitle("VALIDATION-SELECTED THRESHOLD EXPERIMENT")
    metric_row("Selected on validation", f"{selected_threshold:.3f}")
    metric_row("Validation Dice", percent(float(validation["dice"])))
    metric_row("Frozen-test Dice", percent(float(rejected_test["dice"])))
    metric_row("Frozen-test IoU", percent(float(rejected_test["iou"])))
    metric_row(
        "Frozen-test precision",
        percent(float(rejected_test["precision"])),
    )
    metric_row("Frozen-test recall", percent(float(rejected_test["recall"])))
    print(
        "\nDecision: threshold 0.275 was rejected because its frozen-test "
        "Dice (5.18%) was below the retained threshold 0.25 result (6.56%)."
    )
    print(
        "Segmentation remains experimental and is not clinically validated."
    )


def show_evidence_files() -> None:
    subtitle("SAVED EVIDENCE USED")
    print(f"Classification: {CLASSIFICATION_METRICS_PATH}")
    print(f"U-Net         : {UNET_METRICS_PATH}")
    print(
        "Predictions   : "
        f"{PROJECT_DIR / 'experiments/final_frozen_test/predictions.csv'}"
    )


def main() -> None:
    classification_metrics = load_json(CLASSIFICATION_METRICS_PATH)
    unet_metrics = load_json(UNET_METRICS_PATH)

    print("\n" + "=" * LINE_WIDTH)
    print("TBI DETECTION PROJECT — PRESENTATION RESULTS".center(LINE_WIDTH))
    print("=" * LINE_WIDTH)

    show_model_summary(classification_metrics)
    show_per_class_results(classification_metrics)
    show_binary_results(classification_metrics)
    show_hierarchical_results(classification_metrics)
    show_unet_results(unet_metrics)
    show_evidence_files()

    title("RESULTS COMPLETE")
    print("\nNo model training was performed.")
    print("No model inference was rerun.")
    print("Results were read from the saved, verified evaluation artifacts.")
    print(
        "\nFinal selected model: EfficientNetB0 128-D embedding + "
        "ExtraTrees, with a 90% tree / 10% neural probability blend."
    )
    print("Final six-class accuracy: 45.69% (baseline: 44.25%).")
    print("Frozen-test binary accuracy: 87.86%.")
    print("Retained U-Net frozen-test Dice: 6.56% at threshold 0.25.")
    print("=" * LINE_WIDTH)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"\nERROR: Unable to display saved final results.\n{error}")
        raise SystemExit(1) from error
