# ============================================================
# TBI SEGMENTATION - U-NET TRAINING
# Corrected version with differentiable Soft Dice Loss
# ============================================================

import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau
)


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = "segmentation-slices"

TRAIN_IMAGE_DIR = os.path.join(
    DATASET_DIR,
    "train",
    "images"
)

TRAIN_MASK_DIR = os.path.join(
    DATASET_DIR,
    "train",
    "masks"
)

VAL_IMAGE_DIR = os.path.join(
    DATASET_DIR,
    "validation",
    "images"
)

VAL_MASK_DIR = os.path.join(
    DATASET_DIR,
    "validation",
    "masks"
)

TEST_IMAGE_DIR = os.path.join(
    DATASET_DIR,
    "test",
    "images"
)

TEST_MASK_DIR = os.path.join(
    DATASET_DIR,
    "test",
    "masks"
)


# ============================================================
# OUTPUTS
# ============================================================

MODEL_PATH = "unet_best_v2.keras"

RESULTS_DIR = "unet_results_v2"

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# TRAINING SETTINGS
# ============================================================

IMAGE_SIZE = 256

BATCH_SIZE = 8

EPOCHS = 25

LEARNING_RATE = 1e-4

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)

np.random.seed(SEED)

tf.random.set_seed(SEED)


# ============================================================
# GPU INFORMATION
# ============================================================

print()
print("=" * 70)
print("U-NET TBI SEGMENTATION TRAINING")
print("=" * 70)

print()

print(
    "TensorFlow version:",
    tf.__version__
)

print(
    "Available GPUs:",
    tf.config.list_physical_devices("GPU")
)

print()


# ============================================================
# DATASET FILES
# ============================================================

def get_files(directory):

    files = []

    for filename in os.listdir(directory):

        if filename.lower().endswith(".png"):

            files.append(filename)

    files.sort()

    return files


train_images = get_files(
    TRAIN_IMAGE_DIR
)

train_masks = get_files(
    TRAIN_MASK_DIR
)

val_images = get_files(
    VAL_IMAGE_DIR
)

val_masks = get_files(
    VAL_MASK_DIR
)

test_images = get_files(
    TEST_IMAGE_DIR
)

test_masks = get_files(
    TEST_MASK_DIR
)


# ============================================================
# DATASET VERIFICATION
# ============================================================

def verify_pairs(
    image_files,
    mask_files,
    name
):

    print(
        f"{name} images:",
        len(image_files)
    )

    print(
        f"{name} masks :",
        len(mask_files)
    )

    if len(image_files) != len(mask_files):

        raise ValueError(
            f"{name}: image/mask count mismatch"
        )

    image_set = set(image_files)

    mask_set = set(mask_files)

    missing_masks = image_set - mask_set

    missing_images = mask_set - image_set

    if missing_masks:

        raise ValueError(
            f"{name}: missing masks: "
            f"{list(missing_masks)[:10]}"
        )

    if missing_images:

        raise ValueError(
            f"{name}: missing images: "
            f"{list(missing_images)[:10]}"
        )

    print(
        f"{name}: pairing verified"
    )

    print()


verify_pairs(
    train_images,
    train_masks,
    "TRAIN"
)

verify_pairs(
    val_images,
    val_masks,
    "VALIDATION"
)

verify_pairs(
    test_images,
    test_masks,
    "TEST"
)


# ============================================================
# DATA GENERATOR
# ============================================================

class SegmentationSequence(
    tf.keras.utils.Sequence
):

    def __init__(
        self,
        image_dir,
        mask_dir,
        image_files,
        batch_size=8,
        image_size=256,
        shuffle=False,
        **kwargs
    ):

        super().__init__(**kwargs)

        self.image_dir = image_dir

        self.mask_dir = mask_dir

        self.image_files = image_files

        self.batch_size = batch_size

        self.image_size = image_size

        self.shuffle = shuffle

        self.indices = np.arange(
            len(self.image_files)
        )

        self.on_epoch_end()


    def __len__(self):

        return int(
            np.ceil(
                len(self.image_files)
                /
                self.batch_size
            )
        )


    def __getitem__(
        self,
        index
    ):

        start = (
            index
            *
            self.batch_size
        )

        end = min(
            start + self.batch_size,
            len(self.image_files)
        )

        batch_indices = self.indices[
            start:end
        ]

        batch_images = []

        batch_masks = []


        for i in batch_indices:

            filename = self.image_files[i]

            image_path = os.path.join(
                self.image_dir,
                filename
            )

            mask_path = os.path.join(
                self.mask_dir,
                filename
            )


            # ------------------------------------------------
            # LOAD IMAGE
            # ------------------------------------------------

            image = tf.keras.utils.load_img(
                image_path,
                color_mode="grayscale",
                target_size=(
                    self.image_size,
                    self.image_size
                )
            )

            image = tf.keras.utils.img_to_array(
                image
            )

            image = image.astype(
                np.float32
            )

            image = image / 255.0


            # ------------------------------------------------
            # LOAD MASK
            # ------------------------------------------------

            mask = tf.keras.utils.load_img(
                mask_path,
                color_mode="grayscale",
                target_size=(
                    self.image_size,
                    self.image_size
                )
            )

            mask = tf.keras.utils.img_to_array(
                mask
            )

            mask = mask.astype(
                np.float32
            )

            # Masks are binary 0/1
            mask = (
                mask > 0
            ).astype(
                np.float32
            )


            batch_images.append(
                image
            )

            batch_masks.append(
                mask
            )


        return (
            np.array(
                batch_images,
                dtype=np.float32
            ),
            np.array(
                batch_masks,
                dtype=np.float32
            )
        )


    def on_epoch_end(self):

        if self.shuffle:

            np.random.shuffle(
                self.indices
            )


# ============================================================
# CREATE GENERATORS
# ============================================================

train_generator = SegmentationSequence(

    TRAIN_IMAGE_DIR,

    TRAIN_MASK_DIR,

    train_images,

    batch_size=BATCH_SIZE,

    image_size=IMAGE_SIZE,

    shuffle=True
)


val_generator = SegmentationSequence(

    VAL_IMAGE_DIR,

    VAL_MASK_DIR,

    val_images,

    batch_size=BATCH_SIZE,

    image_size=IMAGE_SIZE,

    shuffle=False
)


test_generator = SegmentationSequence(

    TEST_IMAGE_DIR,

    TEST_MASK_DIR,

    test_images,

    batch_size=BATCH_SIZE,

    image_size=IMAGE_SIZE,

    shuffle=False
)


# ============================================================
# DIFFERENTIABLE SOFT DICE
# ============================================================

def dice_coefficient(
    y_true,
    y_pred,
    smooth=1e-6
):

    # IMPORTANT:
    # Do NOT threshold y_pred here.
    # This keeps Dice differentiable.

    y_true = tf.cast(
        y_true,
        tf.float32
    )

    y_pred = tf.cast(
        y_pred,
        tf.float32
    )


    # Flatten each image separately

    y_true = tf.reshape(
        y_true,
        [
            tf.shape(y_true)[0],
            -1
        ]
    )

    y_pred = tf.reshape(
        y_pred,
        [
            tf.shape(y_pred)[0],
            -1
        ]
    )


    intersection = tf.reduce_sum(
        y_true * y_pred,
        axis=1
    )


    denominator = (

        tf.reduce_sum(
            y_true,
            axis=1
        )

        +

        tf.reduce_sum(
            y_pred,
            axis=1
        )

    )


    dice = (

        (
            2.0
            *
            intersection
            +
            smooth
        )

        /

        (
            denominator
            +
            smooth
        )

    )


    return tf.reduce_mean(
        dice
    )


# ============================================================
# BINARY DICE METRIC
# ============================================================

def binary_dice_metric(
    y_true,
    y_pred,
    smooth=1e-6
):

    # Threshold ONLY for reporting metric.
    # This metric is NOT used as the loss.

    y_true = tf.cast(
        y_true > 0.5,
        tf.float32
    )

    y_pred = tf.cast(
        y_pred > 0.5,
        tf.float32
    )


    y_true = tf.reshape(
        y_true,
        [
            tf.shape(y_true)[0],
            -1
        ]
    )

    y_pred = tf.reshape(
        y_pred,
        [
            tf.shape(y_pred)[0],
            -1
        ]
    )


    intersection = tf.reduce_sum(
        y_true * y_pred,
        axis=1
    )


    denominator = (

        tf.reduce_sum(
            y_true,
            axis=1
        )

        +

        tf.reduce_sum(
            y_pred,
            axis=1
        )

    )


    dice = (

        (
            2.0
            *
            intersection
            +
            smooth
        )

        /

        (
            denominator
            +
            smooth
        )

    )


    return tf.reduce_mean(
        dice
    )


# ============================================================
# IOU METRIC
# ============================================================

def iou_metric(
    y_true,
    y_pred,
    smooth=1e-6
):

    y_true = tf.cast(
        y_true > 0.5,
        tf.float32
    )

    y_pred = tf.cast(
        y_pred > 0.5,
        tf.float32
    )


    y_true = tf.reshape(
        y_true,
        [
            tf.shape(y_true)[0],
            -1
        ]
    )

    y_pred = tf.reshape(
        y_pred,
        [
            tf.shape(y_pred)[0],
            -1
        ]
    )


    intersection = tf.reduce_sum(
        y_true * y_pred,
        axis=1
    )


    union = (

        tf.reduce_sum(
            y_true,
            axis=1
        )

        +

        tf.reduce_sum(
            y_pred,
            axis=1
        )

        -

        intersection

    )


    iou = (

        (
            intersection
            +
            smooth
        )

        /

        (
            union
            +
            smooth
        )

    )


    return tf.reduce_mean(
        iou
    )


# ============================================================
# DICE LOSS
# ============================================================

def dice_loss(
    y_true,
    y_pred
):

    return (

        1.0
        -
        dice_coefficient(
            y_true,
            y_pred
        )

    )


# ============================================================
# COMBINED BCE + DICE LOSS
# ============================================================

def combined_loss(
    y_true,
    y_pred
):

    bce = tf.keras.losses.binary_crossentropy(
        y_true,
        y_pred
    )

    bce = tf.reduce_mean(
        bce
    )


    d_loss = dice_loss(
        y_true,
        y_pred
    )


    return (
        bce
        +
        d_loss
    )


# ============================================================
# U-NET CONVOLUTIONAL BLOCK
# ============================================================

def conv_block(
    x,
    filters
):

    x = layers.Conv2D(
        filters,
        3,
        padding="same",
        kernel_initializer="he_normal"
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.Activation(
        "relu"
    )(x)


    x = layers.Conv2D(
        filters,
        3,
        padding="same",
        kernel_initializer="he_normal"
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.Activation(
        "relu"
    )(x)


    return x


# ============================================================
# BUILD U-NET
# ============================================================

def build_unet():

    inputs = layers.Input(
        shape=(
            IMAGE_SIZE,
            IMAGE_SIZE,
            1
        )
    )


    # --------------------------------------------------------
    # ENCODER
    # --------------------------------------------------------

    c1 = conv_block(
        inputs,
        32
    )

    p1 = layers.MaxPooling2D(
        2
    )(c1)


    c2 = conv_block(
        p1,
        64
    )

    p2 = layers.MaxPooling2D(
        2
    )(c2)


    c3 = conv_block(
        p2,
        128
    )

    p3 = layers.MaxPooling2D(
        2
    )(c3)


    c4 = conv_block(
        p3,
        256
    )

    p4 = layers.MaxPooling2D(
        2
    )(c4)


    # --------------------------------------------------------
    # BOTTLENECK
    # --------------------------------------------------------

    bottleneck = conv_block(
        p4,
        512
    )


    # --------------------------------------------------------
    # DECODER
    # --------------------------------------------------------

    u6 = layers.Conv2DTranspose(
        256,
        2,
        strides=2,
        padding="same"
    )(bottleneck)

    u6 = layers.concatenate(
        [
            u6,
            c4
        ]
    )

    c6 = conv_block(
        u6,
        256
    )


    u7 = layers.Conv2DTranspose(
        128,
        2,
        strides=2,
        padding="same"
    )(c6)

    u7 = layers.concatenate(
        [
            u7,
            c3
        ]
    )

    c7 = conv_block(
        u7,
        128
    )


    u8 = layers.Conv2DTranspose(
        64,
        2,
        strides=2,
        padding="same"
    )(c7)

    u8 = layers.concatenate(
        [
            u8,
            c2
        ]
    )

    c8 = conv_block(
        u8,
        64
    )


    u9 = layers.Conv2DTranspose(
        32,
        2,
        strides=2,
        padding="same"
    )(c8)

    u9 = layers.concatenate(
        [
            u9,
            c1
        ]
    )

    c9 = conv_block(
        u9,
        32
    )


    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    outputs = layers.Conv2D(
        1,
        1,
        activation="sigmoid"
    )(c9)


    model = models.Model(
        inputs=inputs,
        outputs=outputs
    )


    return model


# ============================================================
# BUILD MODEL
# ============================================================

print()
print("=" * 70)
print("BUILDING U-NET")
print("=" * 70)

model = build_unet()


# ============================================================
# COMPILE
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),

    loss=combined_loss,

    metrics=[
        binary_dice_metric,
        iou_metric
    ]
)


model.summary()


# ============================================================
# CALLBACKS
# ============================================================

checkpoint = ModelCheckpoint(

    MODEL_PATH,

    monitor="val_binary_dice_metric",

    mode="max",

    save_best_only=True,

    verbose=1
)


early_stopping = EarlyStopping(

    monitor="val_loss",

    mode="min",

    patience=5,

    restore_best_weights=True,

    verbose=1
)


reduce_lr = ReduceLROnPlateau(

    monitor="val_loss",

    mode="min",

    factor=0.5,

    patience=2,

    min_lr=1e-7,

    verbose=1
)


# ============================================================
# TRAIN
# ============================================================

print()
print("=" * 70)
print("STARTING U-NET TRAINING")
print("=" * 70)

print()

print(
    "Training slices:",
    len(train_images)
)

print(
    "Validation slices:",
    len(val_images)
)

print(
    "Test slices:",
    len(test_images)
)

print()

print(
    "Epochs:",
    EPOCHS
)

print(
    "Batch size:",
    BATCH_SIZE
)

print(
    "Learning rate:",
    LEARNING_RATE
)

print()

print(
    "Corrected loss:"
)

print(
    "Binary Cross Entropy + Differentiable Soft Dice Loss"
)

print()


history = model.fit(

    train_generator,

    validation_data=val_generator,

    epochs=EPOCHS,

    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ],

    verbose=1
)


# ============================================================
# LOAD BEST MODEL
# ============================================================

print()
print("=" * 70)
print("LOADING BEST U-NET MODEL")
print("=" * 70)

print()

model = tf.keras.models.load_model(

    MODEL_PATH,

    custom_objects={
        "combined_loss": combined_loss,
        "dice_loss": dice_loss,
        "dice_coefficient": dice_coefficient,
        "binary_dice_metric": binary_dice_metric,
        "iou_metric": iou_metric
    }
)


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

print()
print("=" * 70)
print("FINAL U-NET TEST EVALUATION")
print("=" * 70)

print()


test_results = model.evaluate(

    test_generator,

    verbose=1
)


print()

print(
    "U-NET TEST RESULTS"
)

print(
    "-" * 50
)

for metric_name, value in zip(
    model.metrics_names,
    test_results
):

    print(
        f"{metric_name:<30}: {value:.4f}"
    )


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_file = os.path.join(

    RESULTS_DIR,

    "training_history.npz"

)


np.savez(

    history_file,

    **history.history

)


# ============================================================
# TRAINING CURVES
# ============================================================

history_data = history.history


# ------------------------------------------------------------
# LOSS
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    history_data["loss"],
    label="Training Loss"
)

plt.plot(
    history_data["val_loss"],
    label="Validation Loss"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.title(
    "U-Net Training and Validation Loss"
)

plt.legend()

plt.grid(
    True
)

plt.tight_layout()


plt.savefig(

    os.path.join(
        RESULTS_DIR,
        "training_loss.png"
    ),

    dpi=150
)

plt.close()


# ------------------------------------------------------------
# DICE
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    history_data["binary_dice_metric"],
    label="Training Dice"
)

plt.plot(
    history_data["val_binary_dice_metric"],
    label="Validation Dice"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Dice"
)

plt.title(
    "U-Net Training and Validation Dice"
)

plt.legend()

plt.grid(
    True
)

plt.tight_layout()


plt.savefig(

    os.path.join(
        RESULTS_DIR,
        "training_dice.png"
    ),

    dpi=150
)

plt.close()


# ------------------------------------------------------------
# IOU
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    history_data["iou_metric"],
    label="Training IoU"
)

plt.plot(
    history_data["val_iou_metric"],
    label="Validation IoU"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "IoU"
)

plt.title(
    "U-Net Training and Validation IoU"
)

plt.legend()

plt.grid(
    True
)

plt.tight_layout()


plt.savefig(

    os.path.join(
        RESULTS_DIR,
        "training_iou.png"
    ),

    dpi=150
)

plt.close()


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("U-NET TRAINING COMPLETE")
print("=" * 70)

print()

print(
    "Best model saved:"
)

print(
    MODEL_PATH
)

print()

print(
    "Results saved:"
)

print(
    RESULTS_DIR
)

print()

print(
    "Next step:"
)

print(
    "Evaluate the saved U-Net predictions using a standardized"
)

print(
    "pixel-wise test evaluation and inspect segmentation overlays."
)

print()

print("=" * 70)