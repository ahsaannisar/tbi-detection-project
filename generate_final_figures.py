"""Generate faculty-ready figures from saved, measured final metrics."""

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    metrics = json.load(open("experiments/final_frozen_test/metrics.json", encoding="utf-8"))
    final = metrics["final_six_class"]
    baseline = metrics["baseline_six_class"]
    labels = ["Epidural", "Intraparenchymal", "Intraventricular", "Normal", "Subarachnoid", "Subdural"]

    matrix = np.asarray(final["confusion_matrix"])
    fig, axis = plt.subplots(figsize=(9, 7))
    image = axis.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set(xticks=np.arange(6), yticks=np.arange(6), xticklabels=labels, yticklabels=labels, xlabel="Predicted label", ylabel="True label", title="Final ensemble — frozen test confusion matrix")
    plt.setp(axis.get_xticklabels(), rotation=35, ha="right")
    threshold = matrix.max() / 2
    for row in range(6):
        for column in range(6):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center", color="white" if matrix[row, column] > threshold else "black")
    fig.tight_layout()
    fig.savefig("confusion_matrix.png", dpi=180)
    plt.close(fig)
    shutil.copy2("confusion_matrix.png", "frontend/confusion_matrix.png")

    metric_names = ["Accuracy", "Macro F1", "Balanced accuracy", "Macro precision", "Macro recall"]
    keys = ["accuracy", "macro_f1", "balanced_accuracy", "macro_precision", "macro_recall"]
    baseline_values = [baseline[key] * 100 for key in keys]
    final_values = [final[key] * 100 for key in keys]
    positions = np.arange(len(keys))
    width = 0.36
    fig, axis = plt.subplots(figsize=(10, 6))
    axis.bar(positions - width / 2, baseline_values, width, label="Baseline EfficientNetB0", color="#64748b")
    axis.bar(positions + width / 2, final_values, width, label="Selected ensemble", color="#06b6d4")
    axis.set_ylabel("Frozen test score (%)")
    axis.set_title("Baseline vs verified final model")
    axis.set_xticks(positions, metric_names, rotation=20, ha="right")
    axis.set_ylim(0, 55)
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    for offset, values in ((-width / 2, baseline_values), (width / 2, final_values)):
        for position, value in zip(positions, values):
            axis.text(position + offset, value + 0.6, f"{value:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig("model_improvement_comparison.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
