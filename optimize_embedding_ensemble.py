"""Focused validation-only optimization of the promising B0 embedding head.

This script intentionally has no test-set option. It stores the selected
classifier/ensemble recipe for a later, single frozen-test evaluation.
"""

from __future__ import annotations

import csv
import json
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

from classification_experiments import ImageSequence, calculate_metrics, read_items


OUTPUT = Path("experiments/exp08_embedding_ensemble")
FEATURES = Path("experiments/exp03_feature_heads_nearest")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    train = np.load(FEATURES / "train_features.npz")
    validation = np.load(FEATURES / "validation_features.npz")
    train_x, train_y = train["x"], train["y"]
    validation_x, validation_y = validation["x"], validation["y"]

    validation_items = read_items(Path("tbi-split"), "validation", originals_only=True)
    sequence = ImageSequence(validation_items, 224, 32, "raw", False, False, "nearest")
    keras_model = tf.keras.models.load_model("tbi_efficientnetb0_best.keras", compile=False)
    neural_probabilities = keras_model.predict(sequence, verbose=2)

    definitions = []
    for leaf in (1, 2, 3, 5, 8):
        for max_features in ("sqrt", 0.5):
            definitions.append(
                (
                    f"extra_leaf{leaf}_mf{max_features}",
                    ExtraTreesClassifier(
                        n_estimators=1000,
                        min_samples_leaf=leaf,
                        max_features=max_features,
                        class_weight="balanced",
                        n_jobs=-1,
                        random_state=42,
                    ),
                )
            )
    for leaf in (1, 3, 6):
        definitions.append(
            (
                f"random_forest_leaf{leaf}",
                RandomForestClassifier(
                    n_estimators=750,
                    min_samples_leaf=leaf,
                    max_features="sqrt",
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=42,
                ),
            )
        )

    rows = []
    fitted = {}
    probability_sets = {"keras_head": neural_probabilities}
    neural_metrics = calculate_metrics(validation_y, neural_probabilities)
    rows.append({"name": "keras_head", **{key: neural_metrics[key] for key in ("accuracy", "balanced_accuracy", "macro_f1", "macro_precision", "macro_recall")}})

    for name, classifier in definitions:
        classifier.fit(train_x, train_y)
        probabilities = classifier.predict_proba(validation_x)
        metrics = calculate_metrics(validation_y, probabilities)
        row = {"name": name, **{key: metrics[key] for key in ("accuracy", "balanced_accuracy", "macro_f1", "macro_precision", "macro_recall")}}
        print(json.dumps(row))
        rows.append(row)
        fitted[name] = classifier
        probability_sets[name] = probabilities

    ranked_trees = sorted(
        [row for row in rows if row["name"] != "keras_head"],
        key=lambda row: row["macro_f1"],
        reverse=True,
    )
    top_names = [row["name"] for row in ranked_trees[:4]]

    ensemble_recipes = []
    # Test whether the neural and tree errors are complementary.
    for tree_name in top_names:
        for neural_weight in np.arange(0.0, 0.51, 0.05):
            probabilities = neural_weight * neural_probabilities + (1.0 - neural_weight) * probability_sets[tree_name]
            name = f"blend_{tree_name}_neural{neural_weight:.2f}"
            metrics = calculate_metrics(validation_y, probabilities)
            rows.append({"name": name, **{key: metrics[key] for key in ("accuracy", "balanced_accuracy", "macro_f1", "macro_precision", "macro_recall")}})
            ensemble_recipes.append((name, metrics, {"type": "neural_tree", "tree": tree_name, "neural_weight": float(neural_weight)}))

    # Test compact tree ensembles; only individually top-ranked heads enter.
    for count in (2, 3, 4):
        for names in combinations(top_names, count):
            probabilities = np.mean([probability_sets[name] for name in names], axis=0)
            name = "average_" + "__".join(names)
            metrics = calculate_metrics(validation_y, probabilities)
            rows.append({"name": name, **{key: metrics[key] for key in ("accuracy", "balanced_accuracy", "macro_f1", "macro_precision", "macro_recall")}})
            ensemble_recipes.append((name, metrics, {"type": "tree_average", "trees": list(names)}))

    rows.sort(key=lambda row: row["macro_f1"], reverse=True)
    best = rows[0]
    recipe = next((recipe for name, _, recipe in ensemble_recipes if name == best["name"]), None)
    if recipe is None:
        recipe = {"type": "single_tree", "tree": best["name"]}

    required_trees = [recipe["tree"]] if "tree" in recipe else recipe.get("trees", [])
    artifact = {"recipe": recipe, "classifiers": {name: fitted[name] for name in required_trees}}
    joblib.dump(artifact, OUTPUT / "selected_ensemble.joblib")

    with (OUTPUT / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    best_probabilities = probabilities_for_recipe(recipe, fitted, probability_sets, neural_probabilities)
    best_metrics = calculate_metrics(validation_y, best_probabilities)
    with (OUTPUT / "selection.json").open("w", encoding="utf-8") as handle:
        json.dump({"best": best, "recipe": recipe, "metrics": best_metrics, "top_results": rows[:20]}, handle, indent=2)
    np.save(OUTPUT / "validation_probabilities.npy", best_probabilities)
    print("BEST", json.dumps(best), json.dumps(recipe))


def probabilities_for_recipe(recipe, fitted, probability_sets, neural_probabilities):
    if recipe["type"] == "single_tree":
        return probability_sets[recipe["tree"]]
    if recipe["type"] == "tree_average":
        return np.mean([probability_sets[name] for name in recipe["trees"]], axis=0)
    tree_probability = probability_sets[recipe["tree"]]
    weight = recipe["neural_weight"]
    return weight * neural_probabilities + (1.0 - weight) * tree_probability


if __name__ == "__main__":
    main()
