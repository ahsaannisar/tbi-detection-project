from pathlib import Path
import csv
import random
import shutil
import re

# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = Path("TBI-6000 data")
OUTPUT_DIR = Path("tbi-split")

CLASSES = [
    "epidural",
    "intraparenchymal",
    "intraventricular",
    "normal",
    "subarachnoid",
    "subdural",
]

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

SEED = 42


# ============================================================
# GET AUGMENTATION FAMILY
# ============================================================

def get_family(filename):
    """
    Groups an original image with its augmented versions.

    Example:
        ID_169_epidural_UNIQUE.png
        ID_169_epidural_UNIQUE_aug_266.png

    Both belong to the same family.
    """

    match = re.match(r"(.+?)_aug_\d+\.png$", filename)

    if match:
        return match.group(1)

    return filename[:-4] if filename.endswith(".png") else filename


# ============================================================
# MAIN
# ============================================================

def main():

    random.seed(SEED)

    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {DATASET_DIR}"
        )

    print("=" * 60)
    print("TBI-6000 DATASET SPLITTER")
    print("=" * 60)

    # --------------------------------------------------------
    # Check dataset
    # --------------------------------------------------------

    print(f"\nDataset: {DATASET_DIR}")
    print(f"Output : {OUTPUT_DIR}")

    total_images = 0

    for class_name in CLASSES:

        class_dir = DATASET_DIR / class_name

        if not class_dir.exists():
            raise FileNotFoundError(
                f"Missing class directory: {class_dir}"
            )

        images = list(class_dir.glob("*.png"))

        print(f"{class_name:20s}: {len(images)} images")

        total_images += len(images)

    print(f"\nTotal images: {total_images}")

    # --------------------------------------------------------
    # Create output directories
    # --------------------------------------------------------

    for split in ["train", "validation", "test"]:

        for class_name in CLASSES:

            directory = OUTPUT_DIR / split / class_name

            directory.mkdir(
                parents=True,
                exist_ok=True
            )

    # --------------------------------------------------------
    # CSV manifest
    # --------------------------------------------------------

    manifest_path = OUTPUT_DIR / "split_manifest.csv"

    manifest_rows = []

    # --------------------------------------------------------
    # Process each class
    # --------------------------------------------------------

    for class_name in CLASSES:

        class_dir = DATASET_DIR / class_name

        images = list(class_dir.glob("*.png"))

        # ----------------------------------------------------
        # Group augmented images with original
        # ----------------------------------------------------

        families = {}

        for image_path in images:

            family = get_family(image_path.name)

            if family not in families:
                families[family] = []

            families[family].append(image_path)

        family_list = list(families.keys())

        random.shuffle(family_list)

        total_families = len(family_list)

        train_end = int(total_families * TRAIN_RATIO)

        val_end = train_end + int(
            total_families * VAL_RATIO
        )

        train_families = family_list[:train_end]

        val_families = family_list[train_end:val_end]

        test_families = family_list[val_end:]

        split_families = {
            "train": train_families,
            "validation": val_families,
            "test": test_families,
        }

        print(
            f"\n{class_name}"
        )

        for split_name, selected_families in split_families.items():

            split_image_count = 0

            for family in selected_families:

                for image_path in families[family]:

                    destination = (
                        OUTPUT_DIR
                        / split_name
                        / class_name
                        / image_path.name
                    )

                    shutil.copy2(
                        image_path,
                        destination
                    )

                    manifest_rows.append(
                        [
                            image_path.name,
                            class_name,
                            split_name,
                            family,
                        ]
                    )

                    split_image_count += 1

            print(
                f"  {split_name:12s}: "
                f"{split_image_count} images"
            )

    # --------------------------------------------------------
    # Save manifest
    # --------------------------------------------------------

    with open(
        manifest_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "filename",
                "class",
                "split",
                "augmentation_family",
            ]
        )

        writer.writerows(manifest_rows)

    # --------------------------------------------------------
    # Final verification
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("FINAL SPLIT")
    print("=" * 60)

    grand_total = 0

    for split in ["train", "validation", "test"]:

        split_total = 0

        for class_name in CLASSES:

            directory = (
                OUTPUT_DIR
                / split
                / class_name
            )

            count = len(
                list(directory.glob("*.png"))
            )

            print(
                f"{split:12s} | "
                f"{class_name:20s} | "
                f"{count}"
            )

            split_total += count

        print(
            f"{split:12s} TOTAL: {split_total}"
        )

        grand_total += split_total

    print("\n" + "=" * 60)
    print(f"TOTAL COPIED: {grand_total}")
    print(f"MANIFEST: {manifest_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()