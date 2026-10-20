# TBI model-improvement report

## Final verified result

Model selection used only the 527 original (non-`_aug_`) validation images. The selected recipe was then evaluated once on the complete, unchanged 626-image test split.

| Metric | Original EfficientNetB0 | Selected ensemble | Change |
|---|---:|---:|---:|
| Accuracy | 44.25% | **45.69%** | +1.44 pp |
| Macro F1 | 42.72% | **44.89%** | +2.17 pp |
| Balanced accuracy | 43.79% | **45.36%** | +1.57 pp |
| Macro precision | 44.53% | **45.83%** | +1.30 pp |
| Macro recall | 43.79% | **45.36%** | +1.57 pp |
| Correct predictions | 277 / 626 | **286 / 626** | +9 |

This is a genuine but modest six-class improvement. It should not be presented as a large breakthrough.

The separately selected binary detector performed substantially better:

| Binary normal-vs-TBI metric | Frozen test result |
|---|---:|
| Accuracy | **87.86%** |
| Macro F1 | **76.22%** |
| Balanced accuracy | **74.96%** |
| TBI/abnormal recall | **93.92%** |
| Normal recall | 56.00% |

The binary confusion matrix was `[[56, 44], [32, 494]]` for `[normal, TBI/abnormal]`. It is sensitive to abnormal scans but still produces many false-positive TBI alerts, so it is not clinically deployable.

## Final model selection

- **Best model:** EfficientNetB0 learned embedding + ExtraTrees + neural probability blend
- **Preprocessing:** raw grayscale PNG copied to RGB; nearest-neighbor resize; pixel range 0–255 (EfficientNet performs its own rescaling)
- **Input size:** 224×224×3
- **Six-class probability rule:** 90% ExtraTrees probability + 10% EfficientNet neural-head probability
- **Binary stage:** ExtraTrees classifier trained on the same 128-dimensional EfficientNet embedding
- **Selection criterion:** original-validation macro F1
- **Frozen-test accuracy:** **45.69%**
- **Frozen-test macro F1:** **44.89%**
- **Frozen-test balanced accuracy:** **45.36%**
- **Why selected:** it achieved the strongest validation macro F1 (47.20%) and validation accuracy (49.34%) among tested candidates, then retained a positive improvement on the untouched test set.

## Validation experiment comparison

All values below are from original validation images and were used for model selection. Failed experiments are retained.

| Experiment | Accuracy | Balanced accuracy | Macro F1 | Outcome |
|---|---:|---:|---:|---|
| Original EfficientNetB0 | 45.35% | 45.69% | 42.39% | Baseline |
| Revised B0, raw, 224 | 41.94% | 43.14% | 40.25% | Worse |
| B0, pseudo-multi-window, 224 | 39.09% | 39.29% | 38.62% | Worse |
| B0, raw, 320 | 41.94% | 41.79% | 40.05% | Worse; much slower |
| DenseNet121, raw, 224 | 44.59% | 41.87% | 40.44% | Worse; ~3.5 min/epoch |
| EfficientNetB2, raw, 260 | 38.14% | 38.01% | 35.27% | Worse |
| Continued low-LR B0 fine-tuning | 44.78% | 44.74% | 42.41% | No meaningful gain |
| B0 embedding + ExtraTrees | 48.58% | 48.52% | 46.89% | Improved |
| **B0 embedding ensemble (selected)** | **49.34%** | **48.29%** | **47.20%** | Best validation result |
| Best hierarchical pipeline | 47.82% | 46.96% | 45.30% | Did not beat flat ensemble |

## CT preprocessing finding

The source classification dataset contains 8-bit PNGs, not original HU-valued DICOM data. Therefore true clinical CT windows cannot be reconstructed. The tested three-channel representation (raw, mid-density clipping, high-density clipping) is correctly described as **pseudo-windowing**. It reduced validation macro F1 to 38.62%, so raw input was retained.

For future work, the highest-value data improvement is obtaining original DICOM series with Hounsfield units and patient/study identifiers. That would enable real brain/subdural/bone windows and defensible patient-level splitting.

## Binary and hierarchical findings

- Binary normal-vs-TBI classification generalized much better than six-way subtype classification: 87.86% test accuracy and 93.92% abnormal recall.
- The validation-selected two-stage hierarchy did not improve the final six-class task: 44.25% test accuracy and 43.21% macro F1.
- Production therefore exposes the strong binary screening result and the better flat six-class ensemble side by side; it does not pretend the hierarchy won.

## Segmentation result

Validation-only threshold selection chose 0.275, with 31.98% validation Dice. On the frozen segmentation test set it achieved only 5.18% Dice, below the existing threshold 0.25 result of 6.56%. The production threshold remains 0.25. This failed generalization is recorded rather than using the test set to retune the threshold.

The segmentation model remains experimental and is the weakest part of the system. A defensible next model-development step is a patient-level split followed by a 2D/3D nnU-Net-style baseline or a smaller U-Net trained with lesion-aware sampling and Focal-Tversky loss.

## Reproducibility and leakage controls

- Model selection used original validation images only because only the epidural class contains pre-generated augmented files.
- No test image or test label was used to select architectures, preprocessing, thresholds, tree hyperparameters, or ensemble weight.
- The complete 626-image classification test split was retained; no difficult cases were removed.
- The primary test result includes the dataset's 97 augmented epidural test images because removing them after seeing the split would change the frozen test definition. A descriptive original-only subgroup is stored in the metrics file, but it was not used for selection.
- Patient-level independence cannot be claimed because patient/study metadata is unavailable.

## Saved evidence

- Final test metrics: `experiments/final_frozen_test/metrics.json`
- Per-image final predictions: `experiments/final_frozen_test/predictions.csv`
- Validation ensemble comparison: `experiments/exp08_embedding_ensemble/comparison.csv`
- Selected production ensemble: `experiments/exp08_embedding_ensemble/selected_ensemble.joblib`
- Binary/hierarchical classifiers: `experiments/final_frozen_test/auxiliary_classifiers.joblib`
- Segmentation threshold sweep: `experiments/unet_threshold_optimization/metrics.json`
- Reproducible training harness: `classification_experiments.py`

## Demonstration flow

`Upload CT → binary TBI/normal probability → six-class injury prediction → confidence and all probabilities → experimental U-Net mask/overlay`

All displayed values are produced by the saved models; none are hard-coded predictions.
