"""One-time frozen-test evaluation for validation-selected final models."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

from classification_experiments import CLASS_NAMES, ImageSequence, calculate_metrics, read_items


def binary_metrics(y_true, probability):
    prediction = (probability >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "macro_f1": float(f1_score(y_true, prediction, average="macro")),
        "macro_precision": float(precision_score(y_true, prediction, average="macro")),
        "macro_recall": float(recall_score(y_true, prediction, average="macro")),
        "confusion_matrix": confusion_matrix(y_true, prediction).tolist(),
        "classification_report": classification_report(y_true, prediction, target_names=["normal", "tbi_abnormal"], output_dict=True),
    }


def main():
    output = Path("experiments/final_frozen_test")
    output.mkdir(parents=True, exist_ok=True)
    train = np.load("experiments/exp03_feature_heads_nearest/train_features.npz")
    train_x, train_y = train["x"], train["y"]
    test_items = read_items(Path("tbi-split"), "test", originals_only=False)
    test_y = np.asarray([label for _, label in test_items])

    keras_model = tf.keras.models.load_model("tbi_efficientnetb0_best.keras", compile=False)
    embedder = tf.keras.Model(keras_model.input, keras_model.layers[-2].output)
    sequence = ImageSequence(test_items, 224, 32, "raw", False, False, "nearest")
    test_x = embedder.predict(sequence, verbose=2)
    neural_probabilities = keras_model.predict(sequence, verbose=2)

    artifact = joblib.load("experiments/exp08_embedding_ensemble/selected_ensemble.joblib")
    recipe = artifact["recipe"]
    tree = artifact["classifiers"][recipe["tree"]]
    tree_probabilities = tree.predict_proba(test_x)
    final_probabilities = recipe["neural_weight"] * neural_probabilities + (1.0 - recipe["neural_weight"]) * tree_probabilities

    baseline_metrics = calculate_metrics(test_y, neural_probabilities)
    final_metrics = calculate_metrics(test_y, final_probabilities)

    # Binary hyperparameters were fixed by original-validation results.
    binary = ExtraTreesClassifier(n_estimators=500, min_samples_leaf=3, class_weight="balanced", max_features="sqrt", n_jobs=-1, random_state=42)
    binary.fit(train_x, (train_y != 3).astype(int))
    abnormal_column = list(binary.classes_).index(1)
    abnormal_probability = binary.predict_proba(test_x)[:, abnormal_column]
    binary_result = binary_metrics((test_y != 3).astype(int), abnormal_probability)

    # Best validation-selected hierarchy: ExtraTrees binary + leaf-3 subtype,
    # with the stage-1 abnormal threshold fixed at 0.60.
    abnormal_train = train_y != 3
    subtype = ExtraTreesClassifier(n_estimators=500, min_samples_leaf=3, class_weight="balanced", max_features="sqrt", n_jobs=-1, random_state=42)
    subtype.fit(train_x[abnormal_train], train_y[abnormal_train])
    subtype_probability = subtype.predict_proba(test_x)
    hierarchy_probability = np.zeros((len(test_y), 6), dtype=np.float64)
    hierarchy_probability[:, 3] = 1.0 - abnormal_probability
    for column, class_index in enumerate(subtype.classes_):
        hierarchy_probability[:, int(class_index)] = abnormal_probability * subtype_probability[:, column]
    subtype_prediction = np.argmax(np.where(np.arange(6) == 3, -1, hierarchy_probability), axis=1)
    hierarchy_prediction = np.where(abnormal_probability >= 0.60, subtype_prediction, 3)
    hierarchy_decisions = np.eye(6)[hierarchy_prediction]
    hierarchy_metrics = calculate_metrics(test_y, hierarchy_decisions)

    original_mask = np.asarray(["_aug_" not in path.name for path, _ in test_items])
    subgroup_metrics = {
        "baseline_original_only": calculate_metrics(test_y[original_mask], neural_probabilities[original_mask]),
        "final_original_only": calculate_metrics(test_y[original_mask], final_probabilities[original_mask]),
    }
    payload = {
        "frozen_test_images": len(test_items),
        "selection_was_validation_only": True,
        "recipe": recipe,
        "baseline_six_class": baseline_metrics,
        "final_six_class": final_metrics,
        "binary_normal_vs_tbi": binary_result,
        "hierarchical_six_class": hierarchy_metrics,
        "descriptive_subgroups_not_used_for_selection": subgroup_metrics,
    }
    with (output / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filename", "true_label", "baseline_prediction", "final_prediction", "binary_prediction", "hierarchical_prediction", *[f"p_{name}" for name in CLASS_NAMES]])
        for index, ((path, label), probability) in enumerate(zip(test_items, final_probabilities)):
            writer.writerow([
                path.name,
                CLASS_NAMES[label],
                CLASS_NAMES[int(np.argmax(neural_probabilities[index]))],
                CLASS_NAMES[int(np.argmax(probability))],
                "tbi_abnormal" if abnormal_probability[index] >= 0.5 else "normal",
                CLASS_NAMES[int(hierarchy_prediction[index])],
                *[float(value) for value in probability],
            ])
    joblib.dump({"binary": binary, "subtype": subtype, "hierarchy_threshold": 0.60}, output / "auxiliary_classifiers.joblib")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
