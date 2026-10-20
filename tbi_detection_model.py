"""
TBI-6000
6-Class Intracranial Hemorrhage Classification

Classes:
1. Epidural
2. Intraparenchymal
3. Intraventricular
4. Normal
5. Subarachnoid
6. Subdural

Dataset split:
    tbi-split/
        train/
        validation/
        test/

The test set is NOT used during training.
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_DIR = "tbi-split/train"
VAL_DIR = "tbi-split/validation"
TEST_DIR = "tbi-split/test"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16

INITIAL_EPOCHS = 15
FINE_TUNE_EPOCHS = 10

MODEL_FILE = "tbi_efficientnetb0_best.keras"

SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# CHECK DATASET
# ============================================================

if not os.path.exists(TRAIN_DIR):
    raise FileNotFoundError(
        f"Training directory not found: {TRAIN_DIR}"
    )

if not os.path.exists(VAL_DIR):
    raise FileNotFoundError(
        f"Validation directory not found: {VAL_DIR}"
    )

if not os.path.exists(TEST_DIR):
    raise FileNotFoundError(
        f"Test directory not found: {TEST_DIR}"
    )


# ============================================================
# DATA GENERATORS
# ============================================================

print("\nLoading datasets...")

train_datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.10,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator()

test_datagen = ImageDataGenerator()


train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True,
    seed=SEED
)


val_data = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMAGE_SIZE,
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)


test_data = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMAGE_SIZE,
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)


NUM_CLASSES = len(train_data.class_indices)

CLASS_NAMES = list(train_data.class_indices.keys())

print("\nClass mapping:")
print(train_data.class_indices)

print(f"\nNumber of classes: {NUM_CLASSES}")
print(f"Training images: {train_data.samples}")
print(f"Validation images: {val_data.samples}")
print(f"Test images: {test_data.samples}")


# ============================================================
# CLASS WEIGHTS
# ============================================================

classes = train_data.classes

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(classes),
    y=classes
)

CLASS_WEIGHTS = {
    i: weight
    for i, weight in enumerate(class_weights_array)
}

print("\nClass weights:")
print(CLASS_WEIGHTS)


# ============================================================
# BUILD MODEL
# ============================================================

print("\nBuilding EfficientNetB0 model...")


base_model = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3)
)

# Freeze pretrained layers initially
base_model.trainable = False


inputs = layers.Input(
    shape=(224, 224, 3)
)

x = base_model(
    inputs,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.BatchNormalization()(x)

x = layers.Dropout(0.4)(x)

x = layers.Dense(
    128,
    activation="relu"
)(x)

x = layers.Dropout(0.3)(x)

outputs = layers.Dense(
    NUM_CLASSES,
    activation="softmax"
)(x)


model = models.Model(
    inputs,
    outputs
)


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-3
    ),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


print("\nModel summary:")
model.summary()


# ============================================================
# CALLBACKS
# ============================================================

callbacks = [

    ModelCheckpoint(
        MODEL_FILE,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    ),

    EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
        verbose=1
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=2,
        min_lr=1e-7,
        verbose=1
    )
]


# ============================================================
# PHASE 1 — TRANSFER LEARNING
# ============================================================

print("\n")
print("=" * 60)
print("PHASE 1 — TRANSFER LEARNING")
print("=" * 60)

history1 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=INITIAL_EPOCHS,
    class_weight=CLASS_WEIGHTS,
    callbacks=callbacks,
    verbose=1
)


# ============================================================
# PHASE 2 — FINE TUNING
# ============================================================

print("\n")
print("=" * 60)
print("PHASE 2 — FINE TUNING")
print("=" * 60)


base_model.trainable = True


# Freeze most early layers.
# Fine-tune only the later EfficientNet layers.

for layer in base_model.layers[:-30]:
    layer.trainable = False


model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


history2 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=FINE_TUNE_EPOCHS,
    class_weight=CLASS_WEIGHTS,
    callbacks=callbacks,
    verbose=1
)


# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\nLoading best saved model...")

model = tf.keras.models.load_model(
    MODEL_FILE
)


# ============================================================
# TEST EVALUATION
# ============================================================

print("\n")
print("=" * 60)
print("FINAL TEST SET EVALUATION")
print("=" * 60)


test_data.reset()

test_loss, test_accuracy = model.evaluate(
    test_data,
    verbose=1
)

print("\nTest Loss:", test_loss)
print("Test Accuracy:", test_accuracy)


# ============================================================
# PREDICTIONS
# ============================================================

test_data.reset()

predictions = model.predict(
    test_data,
    verbose=1
)

y_pred = np.argmax(
    predictions,
    axis=1
)

y_true = test_data.classes


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n")
print("=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

report = classification_report(
    y_true,
    y_pred,
    target_names=CLASS_NAMES,
    zero_division=0
)

print(report)

with open(
    "classification_report.txt",
    "w"
) as f:
    f.write(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)


plt.figure(
    figsize=(10, 8)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)

plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.title("TBI-6000 Test Set Confusion Matrix")

plt.tight_layout()

plt.savefig(
    "confusion_matrix.png",
    dpi=200
)

plt.close()

print(
    "\nSaved confusion matrix to: confusion_matrix.png"
)


# ============================================================
# TRAINING CURVES
# ============================================================

train_accuracy = (
    history1.history["accuracy"]
    + history2.history["accuracy"]
)

val_accuracy = (
    history1.history["val_accuracy"]
    + history2.history["val_accuracy"]
)

train_loss = (
    history1.history["loss"]
    + history2.history["loss"]
)

val_loss = (
    history1.history["val_loss"]
    + history2.history["val_loss"]
)


epochs = range(
    1,
    len(train_accuracy) + 1
)


# Accuracy

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    epochs,
    train_accuracy,
    label="Training Accuracy"
)

plt.plot(
    epochs,
    val_accuracy,
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title(
    "Training and Validation Accuracy"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "training_accuracy.png",
    dpi=200
)

plt.close()


# Loss

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    epochs,
    train_loss,
    label="Training Loss"
)

plt.plot(
    epochs,
    val_loss,
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "Training and Validation Loss"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "training_loss.png",
    dpi=200
)

plt.close()


print("Saved training accuracy graph.")
print("Saved training loss graph.")


# ============================================================
# FINAL INFORMATION
# ============================================================

print("\n")
print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(f"Best model: {MODEL_FILE}")
print(f"Test accuracy: {test_accuracy:.4f}")
print("\nGenerated files:")

print("  -", MODEL_FILE)
print("  - classification_report.txt")
print("  - confusion_matrix.png")
print("  - training_accuracy.png")
print("  - training_loss.png")

print("\nClass mapping:")
for index, class_name in enumerate(CLASS_NAMES):
    print(f"  {index}: {class_name}")

print("=" * 60)