"""Validation-selected classical and hierarchical classifiers on CNN embeddings.

The frozen test split is not enumerated or loaded unless --evaluate-test is
explicitly supplied. Model selection uses original validation images only.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from classification_experiments import (
    CLASS_NAMES,
    ImageSequence,
    calculate_metrics,
    read_items,
)


def embedding_model(model_path: Path):
    model = tf.keras.models.load_model(model_path, compile=False)
    # The production B0 head is Dense(128) -> Dropout -> Dense(6). Dropout is
    # inactive at inference, so its output is the compact learned CT embedding.
    output_shape = tuple(model.layers[-1].output.shape)
    if len(output_shape) != 2 or output_shape[-1] != 6:
        raise ValueError("Expected a six-class Keras classifier")
    return model, tf.keras.Model(model.input, model.layers[-2].output)


def extract(model, items, batch_size: int, interpolation: str):
    sequence = ImageSequence(
        items,
        image_size=224,
        batch_size=batch_size,
        preprocessing="raw",
        shuffle=False,
        augment=False,
        interpolation=interpolation,
    )
    return model.predict(sequence, verbose=2)


def candidates():
    for c in (0.01, 0.1, 1.0, 10.0, 100.0):
        yield f"logreg_C{c:g}", make_pipeline(
            StandardScaler(),
            LogisticRegression(C=c, class_weight="balanced", max_iter=3000, random_state=42),
        )
    for c in (0.1, 1.0, 10.0, 100.0):
        yield f"rbf_svc_C{c:g}", make_pipeline(
            StandardScaler(),
            SVC(C=c, class_weight="balanced", probability=True, random_state=42),
        )
    for leaves in (1, 3, 6):
        yield f"extra_trees_leaf{leaves}", ExtraTreesClassifier(
            n_estimators=500,
            min_samples_leaf=leaves,
            class_weight="balanced",
            max_features="sqrt",
            n_jobs=-1,
            random_state=42,
        )


def aligned_probabilities(classifier, features, class_count=6):
    raw = classifier.predict_proba(features)
    result = np.zeros((len(features), class_count), dtype=np.float64)
    for column, class_index in enumerate(classifier.classes_):
        result[:, int(class_index)] = raw[:, column]
    return result


def hierarchical_probabilities(binary, subtype, features, threshold: float):
    abnormal_column = list(binary.classes_).index(1)
    abnormal = binary.predict_proba(features)[:, abnormal_column]
    subtype_raw = subtype.predict_proba(features)
    probabilities = np.zeros((len(features), 6), dtype=np.float64)
    probabilities[:, 3] = 1.0 - abnormal
    for column, class_index in enumerate(subtype.classes_):
        probabilities[:, int(class_index)] = abnormal * subtype_raw[:, column]

    # The explicit threshold governs the stage-1 decision, while conditional
    # subtype probabilities govern stage 2. Encode that decision for metrics.
    predictions = np.argmax(probabilities, axis=1)
    subtype_prediction = np.argmax(np.where(np.arange(6) == 3, -1, probabilities), axis=1)
    predictions = np.where(abnormal >= threshold, subtype_prediction, 3)
    forced = np.full_like(probabilities, 1e-12)
    forced[np.arange(len(features)), predictions] = 1.0
    return probabilities, forced


def write_rows(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--model", default="tbi_efficientnetb0_best.keras")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--interpolation", choices=["nearest", "bilinear"], default="nearest")
    parser.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args()

    output = Path("experiments") / args.name
    output.mkdir(parents=True, exist_ok=True)
    split_root = Path("tbi-split")
    train_items = read_items(split_root, "train", originals_only=False)
    validation_items = read_items(split_root, "validation", originals_only=True)
    train_y = np.asarray([label for _, label in train_items])
    validation_y = np.asarray([label for _, label in validation_items])

    keras_model, embedder = embedding_model(Path(args.model))
    train_x = extract(embedder, train_items, args.batch_size, args.interpolation)
    validation_x = extract(embedder, validation_items, args.batch_size, args.interpolation)
    np.savez_compressed(output / "train_features.npz", x=train_x, y=train_y)
    np.savez_compressed(output / "validation_features.npz", x=validation_x, y=validation_y)

    # Include the unchanged neural-network head as the reference candidate.
    validation_sequence = ImageSequence(validation_items, 224, args.batch_size, "raw", False, False, args.interpolation)
    neural_probabilities = keras_model.predict(validation_sequence, verbose=0)
    neural_metrics = calculate_metrics(validation_y, neural_probabilities)
    results = [{"name": "keras_head", **{k: neural_metrics[k] for k in ("accuracy", "balanced_accuracy", "macro_f1", "macro_precision", "macro_recall")}}]

    best_name = "keras_head"
    best_score = neural_metrics["macro_f1"]
    best_classifier = None
    best_probabilities = neural_probabilities
    best_metrics = neural_metrics

    fitted = {}
    for name, classifier in candidates():
        classifier.fit(train_x, train_y)
        probabilities = aligned_probabilities(classifier, validation_x)
        metrics = calculate_metrics(validation_y, probabilities)
        results.append({"name": name, **{k: metrics[k] for k in ("accuracy", "balanced_accuracy", "macro_f1", "macro_precision", "macro_recall")}})
        fitted[name] = classifier
        print(name, json.dumps(results[-1]))
        if metrics["macro_f1"] > best_score:
            best_name, best_score = name, metrics["macro_f1"]
            best_classifier, best_probabilities = classifier, probabilities
            best_metrics = metrics

    # Binary experiment and hierarchical normal-vs-abnormal followed by
    # five-way injury subtype.
    subtype_mask = train_y != 3
    binary_candidates = []
    for c_binary in (0.01, 0.1, 1.0, 10.0):
        binary_candidates.append((f"logreg_C{c_binary:g}", make_pipeline(StandardScaler(), LogisticRegression(C=c_binary, class_weight="balanced", max_iter=3000, random_state=42))))
    binary_candidates.append(("extra_trees", ExtraTreesClassifier(n_estimators=500, min_samples_leaf=3, class_weight="balanced", max_features="sqrt", n_jobs=-1, random_state=42)))
    binary_results = []
    for binary_name, binary in binary_candidates:
        binary.fit(train_x, (train_y != 3).astype(int))
        binary_prediction = binary.predict(validation_x)
        binary_true = (validation_y != 3).astype(int)
        binary_results.append({
            "name": binary_name,
            "accuracy": float(accuracy_score(binary_true, binary_prediction)),
            "balanced_accuracy": float(balanced_accuracy_score(binary_true, binary_prediction)),
            "macro_f1": float(f1_score(binary_true, binary_prediction, average="macro")),
            "macro_precision": float(precision_score(binary_true, binary_prediction, average="macro")),
            "macro_recall": float(recall_score(binary_true, binary_prediction, average="macro")),
        })
        for flat_name in ("logreg_C0.1", "logreg_C1", "rbf_svc_C1", "rbf_svc_C10", "extra_trees_leaf3"):
            subtype = fitted[flat_name]
            # Refit the same estimator family on abnormal classes only.
            subtype = clone(subtype)
            subtype.fit(train_x[subtype_mask], train_y[subtype_mask])
            for threshold in np.arange(0.25, 0.76, 0.05):
                probabilities, decision_probabilities = hierarchical_probabilities(binary, subtype, validation_x, float(threshold))
                metrics = calculate_metrics(validation_y, decision_probabilities)
                name = f"hier_{binary_name}_{flat_name}_t{threshold:.2f}"
                results.append({"name": name, **{k: metrics[k] for k in ("accuracy", "balanced_accuracy", "macro_f1", "macro_precision", "macro_recall")}})
                if metrics["macro_f1"] > best_score:
                    best_name, best_score = name, metrics["macro_f1"]
                    best_classifier = {"binary": binary, "subtype": subtype, "threshold": float(threshold)}
                    best_probabilities = probabilities
                    best_metrics = metrics

    results.sort(key=lambda row: row["macro_f1"], reverse=True)
    binary_results.sort(key=lambda row: row["macro_f1"], reverse=True)
    write_rows(output / "validation_comparison.csv", results)
    with (output / "binary_validation.json").open("w", encoding="utf-8") as handle:
        json.dump(binary_results, handle, indent=2)
    with (output / "selection.json").open("w", encoding="utf-8") as handle:
        json.dump({"best_model": best_name, "selection_metric": "original-validation macro_f1", "metrics": best_metrics, "top_results": results[:20]}, handle, indent=2)
    if best_classifier is not None:
        joblib.dump(best_classifier, output / "best_feature_classifier.joblib")

    print("BEST", best_name, json.dumps(results[0]))

    if args.evaluate_test:
        if best_classifier is None:
            raise RuntimeError("Keras head remained best; evaluate it through the main evaluation path")
        test_items = read_items(split_root, "test", originals_only=False)
        test_y = np.asarray([label for _, label in test_items])
        test_x = extract(embedder, test_items, args.batch_size, args.interpolation)
        if isinstance(best_classifier, dict):
            test_probabilities, decision_probabilities = hierarchical_probabilities(
                best_classifier["binary"], best_classifier["subtype"], test_x, best_classifier["threshold"]
            )
            test_metrics = calculate_metrics(test_y, decision_probabilities)
        else:
            test_probabilities = aligned_probabilities(best_classifier, test_x)
            test_metrics = calculate_metrics(test_y, test_probabilities)
        with (output / "test_metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(test_metrics, handle, indent=2)
        print("FROZEN TEST", json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
