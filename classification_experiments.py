"""Reproducible classification experiments for the TBI project.

The test split is never read unless --evaluate-test is explicitly supplied.
Model selection uses original (non ``_aug_``) validation images and macro F1.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageEnhance, ImageOps
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


CLASS_NAMES = [
    "epidural",
    "intraparenchymal",
    "intraventricular",
    "normal",
    "subarachnoid",
    "subdural",
]
SEED = 42


def set_reproducibility(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def read_items(root: Path, split: str, originals_only: bool = False):
    items = []
    for label, class_name in enumerate(CLASS_NAMES):
        for path in sorted((root / split / class_name).glob("*.png")):
            if originals_only and "_aug_" in path.name:
                continue
            items.append((path, label))
    return items


def _window(array: np.ndarray, low: float, high: float) -> np.ndarray:
    array = np.clip(array, low, high)
    return ((array - low) * (255.0 / (high - low))).astype(np.uint8)


def preprocess_pil(image: Image.Image, mode: str) -> Image.Image:
    """Return a three-channel image in the 0..255 range.

    TBI-6000 contains 8-bit windowed PNGs rather than HU-valued DICOMs. The
    ``multi_window`` mode is therefore an intensity-window representation, not
    a claim that true DICOM brain/subdural windows were reconstructed.
    """
    gray = image.convert("L")

    if mode == "raw":
        return Image.merge("RGB", (gray, gray, gray))

    if mode == "autocontrast":
        adjusted = ImageOps.autocontrast(gray, cutoff=1)
        return Image.merge("RGB", (adjusted, adjusted, adjusted))

    if mode == "center_crop":
        width, height = gray.size
        side = int(min(width, height) * 0.90)
        left = (width - side) // 2
        top = (height - side) // 2
        cropped = gray.crop((left, top, left + side, top + side))
        return Image.merge("RGB", (cropped, cropped, cropped))

    if mode == "multi_window":
        array = np.asarray(gray, dtype=np.float32)
        raw = array.astype(np.uint8)
        brain = _window(array, 50.0, 180.0)
        high_density = _window(array, 80.0, 220.0)
        return Image.fromarray(np.stack([raw, brain, high_density], axis=-1), "RGB")

    if mode == "multi_contrast":
        auto = ImageOps.autocontrast(gray, cutoff=1)
        equalized = ImageOps.equalize(gray)
        return Image.merge("RGB", (gray, auto, equalized))

    raise ValueError(f"Unknown preprocessing mode: {mode}")


class ImageSequence(tf.keras.utils.Sequence):
    def __init__(
        self,
        items,
        image_size: int,
        batch_size: int,
        preprocessing: str,
        shuffle: bool,
        augment: bool,
        interpolation: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.items = list(items)
        self.image_size = image_size
        self.batch_size = batch_size
        self.preprocessing = preprocessing
        self.shuffle = shuffle
        self.augment = augment
        self.interpolation = interpolation
        self.indices = np.arange(len(self.items))
        self.rng = np.random.default_rng(SEED)
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.items) / self.batch_size))

    def _augment(self, image: Image.Image) -> Image.Image:
        if not self.augment:
            return image
        if self.rng.random() < 0.5:
            image = ImageOps.mirror(image)
        angle = float(self.rng.uniform(-7.0, 7.0))
        image = image.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=0)
        width, height = image.size
        dx = float(self.rng.uniform(-0.05, 0.05) * width)
        dy = float(self.rng.uniform(-0.05, 0.05) * height)
        image = image.transform(
            image.size,
            Image.Transform.AFFINE,
            (1, 0, -dx, 0, 1, -dy),
            resample=Image.Resampling.BILINEAR,
            fillcolor=0,
        )
        zoom = float(self.rng.uniform(0.90, 1.10))
        scaled_size = (max(1, int(width * zoom)), max(1, int(height * zoom)))
        scaled = image.resize(scaled_size, Image.Resampling.BILINEAR)
        if zoom >= 1.0:
            left = (scaled.width - width) // 2
            top = (scaled.height - height) // 2
            image = scaled.crop((left, top, left + width, top + height))
        else:
            canvas = Image.new("RGB", (width, height), 0)
            canvas.paste(scaled, ((width - scaled.width) // 2, (height - scaled.height) // 2))
            image = canvas
        if self.rng.random() < 0.7:
            image = ImageEnhance.Contrast(image).enhance(float(self.rng.uniform(0.90, 1.10)))
        if self.rng.random() < 0.5:
            image = ImageEnhance.Brightness(image).enhance(float(self.rng.uniform(0.95, 1.05)))
        return image

    def __getitem__(self, batch_index):
        start = batch_index * self.batch_size
        selected = self.indices[start : start + self.batch_size]
        images, labels = [], []
        resample = (
            Image.Resampling.BILINEAR
            if self.interpolation == "bilinear"
            else Image.Resampling.NEAREST
        )
        for index in selected:
            path, label = self.items[index]
            with Image.open(path) as source:
                image = preprocess_pil(source, self.preprocessing)
                image = self._augment(image)
                image = image.resize((self.image_size, self.image_size), resample=resample)
                images.append(np.asarray(image, dtype=np.float32))
            labels.append(label)
        return np.asarray(images, dtype=np.float32), tf.keras.utils.to_categorical(
            labels, num_classes=len(CLASS_NAMES)
        )

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.indices)


def build_model(architecture: str, image_size: int):
    inputs = tf.keras.layers.Input((image_size, image_size, 3), name="ct_image")

    constructors = {
        "efficientnetb0": tf.keras.applications.EfficientNetB0,
        "efficientnetb2": tf.keras.applications.EfficientNetB2,
        "efficientnetb3": tf.keras.applications.EfficientNetB3,
        "densenet121": tf.keras.applications.DenseNet121,
        "resnet50": tf.keras.applications.ResNet50,
    }
    constructor = constructors[architecture]
    base = constructor(
        include_top=False,
        weights="imagenet",
        input_shape=(image_size, image_size, 3),
    )
    base.trainable = False

    x = inputs
    if architecture == "densenet121":
        x = tf.keras.layers.Rescaling(1.0 / 255.0)(x)
        x = tf.keras.layers.Normalization(
            mean=[0.485, 0.456, 0.406],
            variance=[0.229**2, 0.224**2, 0.225**2],
        )(x)
    elif architecture == "resnet50":
        # Equivalent to Keras' caffe-mode RGB->BGR preprocessing.
        x = tf.keras.layers.Lambda(
            lambda value: value[..., ::-1]
            - tf.constant([103.939, 116.779, 123.68], dtype=tf.float32),
            name="resnet_preprocess",
        )(x)

    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    x = tf.keras.layers.Dense(256, activation="swish", kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    outputs = tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax", name="classification")(x)
    return tf.keras.Model(inputs, outputs), base


def compile_model(model, learning_rate: float, label_smoothing: float, optimizer_name: str):
    if optimizer_name == "adamw":
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=1e-5,
            clipnorm=1.0,
        )
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
        metrics=["accuracy"],
    )


def calculate_metrics(y_true, probabilities):
    predictions = np.argmax(probabilities, axis=1)
    result = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, predictions, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y_true, predictions, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, predictions, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, predictions).tolist(),
        "classification_report": classification_report(
            y_true,
            predictions,
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        ),
    }
    try:
        result["macro_ovr_auc"] = float(
            roc_auc_score(y_true, probabilities, multi_class="ovr", average="macro")
        )
    except ValueError:
        result["macro_ovr_auc"] = None
    return result


class ValidationMacroF1(tf.keras.callbacks.Callback):
    def __init__(self, validation_sequence, labels, output_dir: Path, patience: int):
        super().__init__()
        self.validation_sequence = validation_sequence
        self.labels = np.asarray(labels)
        self.output_dir = output_dir
        self.best = -np.inf
        self.best_epoch = 0
        self.epoch_offset = 0
        self.patience = patience
        self.wait = 0

    def reset_stage_patience(self):
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs is not None else {}
        probabilities = self.model.predict(self.validation_sequence, verbose=0)
        metrics = calculate_metrics(self.labels, probabilities)
        logs["val_macro_f1"] = metrics["macro_f1"]
        logs["val_balanced_accuracy"] = metrics["balanced_accuracy"]
        absolute_epoch = self.epoch_offset + epoch + 1
        print(
            f"\noriginal-validation epoch={absolute_epoch} "
            f"accuracy={metrics['accuracy']:.4f} "
            f"balanced_accuracy={metrics['balanced_accuracy']:.4f} "
            f"macro_f1={metrics['macro_f1']:.4f}"
        )
        if metrics["macro_f1"] > self.best:
            self.best = metrics["macro_f1"]
            self.best_epoch = absolute_epoch
            self.model.save_weights(self.output_dir / "best.weights.h5")
            with open(self.output_dir / "best_validation_metrics.json", "w", encoding="utf-8") as handle:
                json.dump({"epoch": absolute_epoch, **metrics}, handle, indent=2)
            self.wait = 0
        else:
            self.wait += 1
            if self.patience > 0 and self.wait >= self.patience:
                print(f"Stopping stage after {self.wait} epochs without macro-F1 improvement")
                self.model.stop_training = True


def save_predictions(path: Path, items, probabilities) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filename", "true_label", "predicted_label", *CLASS_NAMES])
        for (image_path, true_label), probability in zip(items, probabilities):
            writer.writerow(
                [
                    image_path.name,
                    CLASS_NAMES[true_label],
                    CLASS_NAMES[int(np.argmax(probability))],
                    *[float(value) for value in probability],
                ]
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--architecture",
        choices=["efficientnetb0", "efficientnetb2", "efficientnetb3", "densenet121", "resnet50"],
        required=True,
    )
    parser.add_argument("--preprocessing", choices=["raw", "autocontrast", "center_crop", "multi_window", "multi_contrast"], default="raw")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--head-epochs", type=int, default=6)
    parser.add_argument("--fine-tune-epochs", type=int, default=8)
    parser.add_argument("--fine-tune-layers", type=int, default=40)
    parser.add_argument("--early-stop-patience", type=int, default=3)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--optimizer", choices=["adam", "adamw"], default="adam")
    parser.add_argument("--interpolation", choices=["nearest", "bilinear"], default="bilinear")
    parser.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args()

    set_reproducibility()
    split_root = Path("tbi-split")
    output_dir = Path("experiments") / args.name
    output_dir.mkdir(parents=True, exist_ok=True)

    train_items = read_items(split_root, "train", originals_only=False)
    validation_items = read_items(split_root, "validation", originals_only=True)
    train_sequence = ImageSequence(
        train_items,
        args.image_size,
        args.batch_size,
        args.preprocessing,
        shuffle=True,
        augment=True,
        interpolation=args.interpolation,
    )
    validation_sequence = ImageSequence(
        validation_items,
        args.image_size,
        args.batch_size,
        args.preprocessing,
        shuffle=False,
        augment=False,
        interpolation=args.interpolation,
    )
    validation_labels = [label for _, label in validation_items]

    config = vars(args) | {
        "selection_split": "validation originals only",
        "training_images": len(train_items),
        "validation_images": len(validation_items),
        "class_names": CLASS_NAMES,
        "seed": SEED,
        "tensorflow_version": tf.__version__,
    }
    with open(output_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    model, base = build_model(args.architecture, args.image_size)
    metric_callback = ValidationMacroF1(validation_sequence, validation_labels, output_dir, args.early_stop_patience)

    print("Training classification head")
    compile_model(model, learning_rate=3e-4, label_smoothing=args.label_smoothing, optimizer_name=args.optimizer)
    history_head = model.fit(
        train_sequence,
        epochs=args.head_epochs,
        callbacks=[metric_callback],
        verbose=2,
    )

    print(f"Fine-tuning final {args.fine_tune_layers} backbone layers")
    # Fine-tune from the best frozen-head checkpoint, not merely the final
    # head epoch (which may have triggered early stopping after regression).
    model.load_weights(output_dir / "best.weights.h5")
    metric_callback.epoch_offset = len(history_head.history["loss"])
    metric_callback.reset_stage_patience()
    base.trainable = True
    for layer in base.layers[:-args.fine_tune_layers]:
        layer.trainable = False
    compile_model(model, learning_rate=1e-5, label_smoothing=args.label_smoothing, optimizer_name=args.optimizer)
    history_fine = model.fit(
        train_sequence,
        epochs=args.fine_tune_epochs,
        callbacks=[metric_callback],
        verbose=2,
    )

    model.load_weights(output_dir / "best.weights.h5")
    model.save(output_dir / "best_model.keras")
    validation_probabilities = model.predict(validation_sequence, verbose=0)
    validation_metrics = calculate_metrics(validation_labels, validation_probabilities)
    with open(output_dir / "final_validation_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(validation_metrics, handle, indent=2)
    save_predictions(output_dir / "validation_predictions.csv", validation_items, validation_probabilities)

    histories = {"head": history_head.history, "fine_tune": history_fine.history}
    with open(output_dir / "history.json", "w", encoding="utf-8") as handle:
        json.dump(histories, handle, indent=2)

    print("Best validation metrics:")
    print(json.dumps(validation_metrics, indent=2))

    if args.evaluate_test:
        test_items = read_items(split_root, "test", originals_only=False)
        test_sequence = ImageSequence(
            test_items,
            args.image_size,
            args.batch_size,
            args.preprocessing,
            shuffle=False,
            augment=False,
            interpolation=args.interpolation,
        )
        test_labels = [label for _, label in test_items]
        test_probabilities = model.predict(test_sequence, verbose=0)
        test_metrics = calculate_metrics(test_labels, test_probabilities)
        with open(output_dir / "test_metrics.json", "w", encoding="utf-8") as handle:
            json.dump(test_metrics, handle, indent=2)
        save_predictions(output_dir / "test_predictions.csv", test_items, test_probabilities)
        print("Frozen test metrics:")
        print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
