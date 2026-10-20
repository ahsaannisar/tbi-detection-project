# TBI Detection and Hemorrhage Segmentation

An end-to-end deep-learning project for analyzing brain CT images through a browser. The system performs:

1. Binary screening: **Normal vs TBI/abnormal**
2. Six-class classification:
   - Epidural hemorrhage
   - Intraparenchymal hemorrhage
   - Intraventricular hemorrhage
   - Normal
   - Subarachnoid hemorrhage
   - Subdural hemorrhage
3. U-Net-based hemorrhage segmentation
4. Display of confidence, class probabilities, and segmentation results

The original six-class EfficientNetB0 model achieved **44.25% frozen-test accuracy**. After controlled validation experiments, the selected EfficientNetB0-embedding ensemble achieved **45.69% accuracy**, **44.89% macro F1**, and **45.36% balanced accuracy** on the same untouched 626-image test set.

> **Important:** This is an academic/research prototype, not a certified medical device. Its predictions must not be used as a substitute for diagnosis by a qualified medical professional.

---

## Contents

- [Quick start](#quick-start)
- [System architecture](#system-architecture)
- [Final measured results](#final-measured-results)
- [Dataset and split strategy](#dataset-and-split-strategy)
- [Development process from start to finish](#development-process-from-start-to-finish)
- [Final classification implementation](#final-classification-implementation)
- [Segmentation implementation](#segmentation-implementation)
- [Backend implementation](#backend-implementation)
- [Frontend implementation](#frontend-implementation)
- [Tools and technologies](#tools-and-technologies)
- [Project structure](#project-structure)
- [Installation](#installation)
- [Running and using the application](#running-and-using-the-application)
- [Reproducing experiments](#reproducing-experiments)
- [Evidence and generated artifacts](#evidence-and-generated-artifacts)
- [Verification performed](#verification-performed)
- [Known limitations](#known-limitations)
- [Recommended future improvements](#recommended-future-improvements)

---

## Quick start

From the project directory:

```bash
source .venv/bin/activate
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Then open:

```text
http://127.0.0.1:8000
```

Upload a PNG, JPG/JPEG, BMP, WEBP, or DICOM CT image and select **Analyze Scan**.

Useful endpoints:

- Application: `http://127.0.0.1:8000/`
- Health check: `http://127.0.0.1:8000/api/health`
- API documentation: `http://127.0.0.1:8000/docs`

---

## System architecture

```text
CT IMAGE / DICOM
       |
       v
Browser frontend (HTML, CSS, JavaScript)
       |
       | multipart/form-data
       v
FastAPI backend
       |
       +--> Image/DICOM decoding and preprocessing
       |
       +--> EfficientNetB0 feature extractor
       |       |
       |       +--> 128-dimensional embedding
       |       |       |
       |       |       +--> ExtraTrees six-class classifier
       |       |       +--> ExtraTrees binary classifier
       |       |
       |       +--> Original six-class neural softmax head
       |
       +--> Final probability blend
       |       90% ExtraTrees + 10% neural head
       |
       +--> U-Net segmentation
       |
       v
JSON response
       |
       v
TBI status + injury type + confidence +
class probabilities + segmentation overlay
```

---

## Final measured results

All final classification numbers below were measured once on the frozen test split after selecting the model and ensemble weight using validation data only.

### Six-class baseline vs selected model

| Metric | Original EfficientNetB0 baseline | Selected ensemble | Absolute change |
|---|---:|---:|---:|
| Accuracy | 44.25% | **45.69%** | **+1.44 percentage points** |
| Macro F1 | 42.72% | **44.89%** | **+2.17 points** |
| Balanced accuracy | 43.79% | **45.36%** | **+1.57 points** |
| Macro precision | 44.53% | **45.83%** | **+1.30 points** |
| Macro recall | 43.79% | **45.36%** | **+1.57 points** |
| Correct predictions | 277 / 626 | **286 / 626** | **+9 images** |

The improvement is real but modest. The README reports the measured result rather than presenting an artificial target.

### Selected model: per-class frozen-test performance

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| Epidural | 76% | 53% | 63% |
| Intraparenchymal | 31% | 28% | 30% |
| Intraventricular | 44% | 62% | 51% |
| Normal | 58% | 64% | 61% |
| Subarachnoid | 31% | 26% | 28% |
| Subdural | 34% | 39% | 36% |

The most difficult distinctions remain among intraparenchymal, subarachnoid, and subdural hemorrhage.

### Binary Normal vs TBI result

The binary classifier uses the same EfficientNetB0 embedding and a separately trained ExtraTrees classifier.

| Metric | Frozen-test result |
|---|---:|
| Accuracy | **87.86%** |
| Balanced accuracy | **74.96%** |
| Macro F1 | **76.22%** |
| Macro precision | **77.73%** |
| TBI recall/sensitivity | **93.92%** |
| Normal recall/specificity | **56.00%** |

Binary confusion matrix, with rows as actual class and columns as predicted class:

```text
                 Predicted Normal   Predicted TBI
Actual Normal            56               44
Actual TBI               32              494
```

The high TBI recall is useful for screening, but the lower normal recall means false-positive TBI alerts remain common.

### Hierarchical classifier result

The evaluated hierarchy was:

```text
Stage 1: Normal vs TBI
Stage 2: If TBI, predict one of five hemorrhage classes
```

| Metric | Frozen-test result |
|---|---:|
| Accuracy | 44.25% |
| Macro F1 | 43.21% |
| Balanced accuracy | 44.03% |

The hierarchical classifier did **not** outperform the selected flat six-class ensemble, so it was not used as the production six-class prediction path. The binary stage is still displayed as a separate screening result.

### Segmentation result

Existing U-Net at its production threshold of `0.25`:

| Metric | Frozen-test result |
|---|---:|
| Test slices | 1,538 |
| Positive-mask slices | 591 |
| Negative-mask slices | 947 |
| Global Dice | **6.56%** |
| Global IoU | **3.39%** |
| Precision | **20.45%** |
| Recall | **3.91%** |
| Mean Dice on positive slices | **7.90%** |

A validation-only threshold search selected `0.275` and reached 31.98% validation Dice, but it achieved only 5.18% Dice on the frozen test set. It was therefore rejected, and the existing production threshold of `0.25` was retained.

Segmentation is integrated and functional, but its measured quality is currently too low for clinical interpretation.

---

## Dataset and split strategy

### Classification dataset

The classifier uses six folders/classes.

| Split | Epidural | Intraparenchymal | Intraventricular | Normal | Subarachnoid | Subdural | Total |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 784 | 800 | 800 | 800 | 800 | 800 | **4,784** |
| Validation | 90 | 100 | 100 | 100 | 100 | 100 | **590** |
| Test | 126 | 100 | 100 | 100 | 100 | 100 | **626** |

The validation folder contains generated variants. Model selection therefore used an **original-only validation subset of 527 images** to reduce the influence of synthetic variants.

The original-only validation distribution was:

- Epidural: 27
- Each of the other five classes: 100
- Total: 527

### How the split was created

`split_tbi6000_dataset.py`:

- Uses a fixed random seed of `42`
- Groups an original image with filenames containing its `_aug_` variants
- Assigns each family to only one split
- Targets an approximate 80/10/10 train/validation/test division
- Copies the selected files into the split directories
- Writes a manifest for traceability

Grouping augmentation families is essential. Splitting individual augmented images independently would leak nearly identical images between training and evaluation.

### Test-set policy

The test set was treated as frozen:

- Architecture and ensemble choices were made on validation data
- Ensemble weights were selected on validation data
- The test labels were not edited
- Difficult examples were not removed
- Predictions were not fabricated or hard-coded
- Final metrics were produced by `evaluate_final_models.py`

### Data-integrity caveats

- The audit found **one exact train/test duplicate**. It was documented but not silently removed after the test protocol had been frozen.
- No patient or study identifiers are available. Therefore, this project **cannot claim patient-level independence**.
- Classification inputs are 8-bit PNG images rather than original CT voxel data. True Hounsfield Unit windows cannot be reconstructed reliably from these PNG files.
- The tested multi-window preprocessing is therefore a **pseudo-window representation**, not a clinically exact HU brain/subdural/bone window.

---

## Development process from start to finish

### 1. Existing end-to-end application

The starting project already had a functioning pipeline:

```text
CT image -> frontend -> FastAPI -> EfficientNetB0 ->
U-Net -> result display
```

The main problem was model quality, especially the original six-class accuracy of approximately 44.25%.

### 2. Baseline verification

The original `tbi_efficientnetb0_best.keras` model was evaluated on the complete frozen test set:

- Accuracy: 44.25%
- Macro F1: 42.72%
- Balanced accuracy: 43.79%
- Correct: 277 of 626

This became the official comparison baseline.

### 3. Reproducible experiment framework

`classification_experiments.py` was created to test controlled combinations of:

- EfficientNetB0, EfficientNetB2, EfficientNetB3
- DenseNet121
- ResNet50
- Raw grayscale-to-RGB preprocessing
- Pseudo multi-window preprocessing
- Higher input resolutions
- Class weighting
- Data augmentation
- Frozen-head and fine-tuning stages
- Early stopping
- Learning-rate reduction
- Macro F1 model selection

By default, experiments do not touch the test set. Test evaluation requires an explicit `--evaluate-test` flag.

### 4. CT preprocessing experiments

Two classification representations were compared:

1. **Raw grayscale replicated into RGB**
2. **Pseudo multi-window channels** derived from 8-bit intensity ranges

Background handling and normalization were included in the experiment pipeline. Because source images were already converted to PNG, these transformations cannot reproduce true Hounsfield Unit windows.

Validation results:

| Experiment | Accuracy | Balanced accuracy | Macro F1 |
|---|---:|---:|---:|
| Revised EfficientNetB0, raw 224 | 41.94% | 43.14% | 40.25% |
| EfficientNetB0, pseudo-window 224 | 39.09% | 39.29% | 38.62% |

The pseudo-window experiment performed worse and was rejected.

### 5. Higher-resolution transfer learning

EfficientNetB0 was tested at `320x320`.

| Experiment | Accuracy | Balanced accuracy | Macro F1 |
|---|---:|---:|---:|
| EfficientNetB0, raw 320 | 41.94% | 41.79% | 40.05% |

The higher resolution did not improve validation performance and substantially increased CPU training time and memory use. A `384x384` run was therefore not prioritized on the available 8 GB machine.

### 6. Alternative architectures

| Experiment | Accuracy | Balanced accuracy | Macro F1 |
|---|---:|---:|---:|
| DenseNet121, raw 224 | 44.59% | 41.87% | 40.44% |
| EfficientNetB2, raw 260 | 38.14% | 38.01% | 35.27% |

DenseNet121 received particular attention because of its medical-imaging reputation, but it did not beat the stronger downstream classifier built from the existing model's learned embedding. On the available CPU, DenseNet121 took roughly 3.5 minutes per epoch, and EfficientNetB2 roughly 2.8–3.9 minutes per epoch.

EfficientNetB3 and ResNet50 remain supported by the experiment script, but full runs were not prioritized after larger tested candidates failed to improve the validation result and hardware limits became significant.

### 7. Conservative EfficientNetB0 continuation

The existing EfficientNetB0 model was fine-tuned conservatively at a low learning rate:

| Experiment | Accuracy | Balanced accuracy | Macro F1 |
|---|---:|---:|---:|
| Continued existing EfficientNetB0 | 44.78% | 44.74% | 42.41% |

This was better than several fresh transfer-learning runs, confirming that the existing model contained useful learned features.

### 8. Learned-embedding classifiers

The original EfficientNetB0 contains a 128-dimensional dense feature layer before its six-class output. `feature_classifier_experiments.py` extracts those embeddings and compares classical classifiers including:

- Logistic regression
- Support vector machine
- ExtraTrees
- Binary classifiers
- Hierarchical classifiers

The ExtraTrees classifier performed best among these focused candidates:

| Experiment | Accuracy | Balanced accuracy | Macro F1 |
|---|---:|---:|---:|
| EfficientNetB0 embedding + ExtraTrees | 48.58% | 48.52% | 46.89% |

### 9. Probability ensemble selection

`optimize_embedding_ensemble.py` compared focused tree configurations and blends. The selected six-class probability was:

```text
final_probability =
    0.90 * ExtraTrees_probability
  + 0.10 * original_neural_head_probability
```

The weight was selected using the original-only validation set, not the test set.

| Validation model | Accuracy | Balanced accuracy | Macro F1 |
|---|---:|---:|---:|
| Original baseline | 45.35% | 45.69% | 42.39% |
| ExtraTrees embedding classifier | 48.58% | 48.52% | 46.89% |
| **Selected 90/10 ensemble** | **49.34%** | **48.29%** | **47.20%** |

The blend had the strongest macro F1 and accuracy among the selected validation candidates.

### 10. Binary and hierarchical evaluation

Separate binary and hierarchical models were trained from the same 128-dimensional embeddings:

- Binary: Normal vs TBI
- Hemorrhage subtype: five abnormal classes
- Hierarchy: binary decision followed by subtype classification

The hierarchy did not beat the selected flat six-class model. The binary classifier was retained because it provides a useful high-sensitivity screening result.

### 11. One-time frozen-test evaluation

After selecting the complete strategy on validation data, `evaluate_final_models.py` evaluated:

- Original six-class baseline
- ExtraTrees six-class head
- Selected 90/10 ensemble
- Binary classifier
- Hierarchical classifier

It generated immutable evidence files containing metrics and per-image predictions.

### 12. U-Net threshold evaluation

`optimize_unet_threshold.py`:

1. Swept thresholds only on validation masks
2. Selected the validation-best threshold
3. Applied that fixed threshold to the test set
4. Compared it with the existing production threshold

The validation-selected threshold generalized poorly, so it was not deployed.

### 13. Production integration

`backend/main.py` was updated to load the verified selected classifiers at startup and return:

- Binary TBI status
- Binary TBI probability
- Six-class injury type
- Six-class confidence
- Six class probabilities
- U-Net segmentation data

`frontend/script.js` was updated to display these real model outputs. No displayed prediction or confidence is hard-coded.

### 14. Project cleanup

Only clearly disposable generated files were removed:

- `__pycache__` directories
- Compiled `.pyc` files
- `.DS_Store` metadata

Approximately 176 MiB of disposable cache data was removed. Datasets, models, source code, results, plots, reports, and experiment evidence were preserved.

---

## Final classification implementation

### Selected six-class model

| Item | Selected configuration |
|---|---|
| Feature extractor | Existing EfficientNetB0 |
| Feature layer | 128-dimensional dense embedding |
| Main classifier | ExtraTrees |
| Ensemble | 90% ExtraTrees + 10% neural softmax head |
| Input size | 224 x 224 |
| Input representation | Grayscale copied to three RGB channels |
| Resize interpolation | Nearest-neighbor |
| Input range | 0–255; EfficientNet performs its internal scaling |
| Test accuracy | **45.69%** |
| Test macro F1 | **44.89%** |
| Test balanced accuracy | **45.36%** |

### Why this model was selected

- It produced the strongest original-only validation macro F1 among the completed candidates.
- The ensemble weight was determined without using the test set.
- It improved every reported aggregate six-class metric over the original baseline.
- It reuses the existing neural representation, making deployment practical on the available hardware.
- Its measured test improvement is documented by saved metrics and predictions.

### Runtime artifacts

The backend requires these files:

| Artifact | Purpose |
|---|---|
| `tbi_efficientnetb0_best.keras` | Original EfficientNetB0, neural probabilities, and embedding extraction |
| `experiments/exp08_embedding_ensemble/selected_ensemble.joblib` | Six-class ExtraTrees model and ensemble metadata |
| `experiments/final_frozen_test/auxiliary_classifiers.joblib` | Binary and hierarchical classifier artifacts |
| `unet_best_v2.keras` | Segmentation model |

Do not rename or remove these files unless the corresponding paths in `backend/main.py` are also updated.

### Classification preprocessing

For ordinary images:

1. Decode with Pillow
2. Apply EXIF orientation correction
3. Convert to grayscale
4. Convert back to three identical RGB channels
5. Resize to 224 x 224 using nearest-neighbor
6. Pass values in the 0–255 range to EfficientNetB0

For DICOM:

1. Decode the pixel array with pydicom
2. Apply rescale slope and intercept when present
3. Normalize/clip the decoded values for display and model input
4. Use the first frame for multi-frame input
5. Convert to the same three-channel 224 x 224 representation

### Inference logic

```python
neural_probabilities = efficientnet_model(image)
embedding = embedding_model(image)
tree_probabilities = six_class_extra_trees.predict_proba(embedding)

final_probabilities = (
    0.90 * tree_probabilities
    + 0.10 * neural_probabilities
)

injury_type = argmax(final_probabilities)
tbi_probability = binary_extra_trees.predict_proba(embedding)[TBI]
```

---

## Segmentation implementation

`train_unet.py` defines and trains a custom two-dimensional U-Net.

### Training configuration

| Setting | Value |
|---|---|
| Input size | 256 x 256 |
| Batch size | 8 |
| Maximum epochs | 25 |
| Initial learning rate | 0.0001 |
| Loss | Binary cross-entropy + soft Dice loss |
| Checkpoint monitor | Validation Dice |
| Training controls | Early stopping and learning-rate reduction |

### Segmentation dataset

| Split | Image/mask pairs |
|---|---:|
| Train | 7,628 |
| Validation | 2,009 |
| Test | 1,538 |

### Runtime behavior

- The uploaded image is resized to 256 x 256.
- U-Net returns a pixel probability map.
- The production threshold `0.25` converts probabilities into a binary mask.
- The backend returns both the mask and a color overlay as base64-encoded PNG data.
- If the mask is empty, the UI reports that no segmentation region was produced.

Because the frozen-test Dice is low, the mask should be treated as experimental visualization only.

---

## Backend implementation

The backend uses FastAPI and TensorFlow/Keras.

### Main file

`backend/main.py`

### Startup behavior

At startup the backend:

1. Resolves project-relative artifact paths
2. Loads the EfficientNetB0 classifier
3. Constructs an embedding extractor from the 128-dimensional feature layer
4. Loads the six-class ExtraTrees ensemble artifact
5. Loads binary and hierarchical auxiliary classifiers
6. Loads the U-Net model with its custom Dice functions
7. Exposes model-load status through the health endpoint

### Routes

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/` | Serves the frontend |
| `GET` | `/api/health` | Reports application and model readiness |
| `POST` | `/api/predict` | Runs classification and segmentation |
| `GET` | `/docs` | Interactive OpenAPI/Swagger documentation |

### Prediction response

The prediction endpoint returns a structure similar to:

```json
{
  "prediction": "normal",
  "confidence": 0.0,
  "tbi_status": "NORMAL",
  "tbi_probability": 0.0,
  "injury_type": "normal",
  "class_probabilities": {
    "epidural": 0.0,
    "intraparenchymal": 0.0,
    "intraventricular": 0.0,
    "normal": 0.0,
    "subarachnoid": 0.0,
    "subdural": 0.0
  },
  "segmentation_available": true,
  "segmentation_detected": false,
  "segmentation_mask": "data:image/png;base64,...",
  "segmentation_overlay": "data:image/png;base64,..."
}
```

The zeros above only illustrate the response schema. Actual requests return probabilities generated by the loaded models.

### Validation and error handling

The API validates:

- File presence
- Supported image/DICOM type
- Decode success
- Model availability
- Prediction output shape

Errors are returned as clear HTTP responses rather than silently substituting fake results.

---

## Frontend implementation

The frontend is implemented with:

- HTML
- CSS
- Vanilla JavaScript
- Browser Fetch API
- Local browser storage for recent scan history

Main files:

- `frontend/index.html`
- `frontend/style.css`
- `frontend/script.js`

The interface supports:

- Drag-and-drop or file-picker upload
- Image preview
- DICOM filename/status display
- Analyze button and loading state
- Binary TBI/Normal result
- Injury-type result
- Confidence display
- Six-class probability bars
- Segmentation mask/overlay display
- Recent analysis history
- Error and readiness messages

The browser does not run the ML models directly. It sends the selected file to FastAPI and renders the returned model outputs.

---

## Tools and technologies

The project was implemented and evaluated locally on an Apple Silicon MacBook Air with 8 GB memory using CPU TensorFlow.

### Core stack

| Tool | Role | Version used during final verification |
|---|---|---:|
| Python | Training, evaluation, backend | 3.12 virtual environment |
| TensorFlow / Keras | EfficientNet and U-Net | 2.21.0 |
| scikit-learn | ExtraTrees, metrics, classical heads | 1.9.0 |
| joblib | Serialized classical models | 1.5.3 |
| FastAPI | Prediction API and static-app server | 0.143.0 |
| Uvicorn | ASGI development server | 0.54.0 |
| NumPy | Array processing | 2.5.2 |
| Pillow | Image decoding and preprocessing | 12.3.0 |
| pydicom | DICOM decoding | 3.0.2 |
| pandas | Experiment tables and prediction files | 3.0.5 |
| Matplotlib | Confusion matrices and comparison figures | 3.11.1 |
| HTML/CSS/JavaScript | Browser interface | Native browser stack |

The serialized joblib classifiers should be loaded with a compatible scikit-learn version, preferably the version recorded above.

### Model architectures and methods tested

- EfficientNetB0
- EfficientNetB2
- DenseNet121
- EfficientNetB3 and ResNet50 support in the experiment framework
- Custom 2-D U-Net
- ExtraTrees
- Logistic regression
- Support vector machine
- Random forest candidates
- Class weighting
- Data augmentation
- Transfer learning
- Fine-tuning
- Higher-resolution input
- Pseudo multi-window input
- Binary classification
- Hierarchical classification
- Probability averaging/ensembling
- Validation-only threshold selection

---

## Project structure

```text
tbi-detection-project/
|
|-- README.md
|-- MODEL_IMPROVEMENT_REPORT.md
|
|-- backend/
|   |-- main.py
|
|-- frontend/
|   |-- index.html
|   |-- styles.css
|   |-- script.js
|   |-- confusion_matrix.png
|
|-- tbi-split/
|   |-- train/
|   |-- validation/
|   |-- test/
|
|-- segmentation-slices/
|   |-- train/
|   |-- validation/
|   |-- test/
|
|-- experiments/
|   |-- experiment_summary.csv
|   |-- exp08_embedding_ensemble/
|   |   |-- selected_ensemble.joblib
|   |-- final_frozen_test/
|       |-- metrics.json
|       |-- predictions.csv
|       |-- auxiliary_classifiers.joblib
|
|-- tbi_efficientnetb0_best.keras
|-- unet_best_v2.keras
|
|-- split_tbi6000_dataset.py
|-- tbi_detection_model.py
|-- classification_experiments.py
|-- continue_b0_finetuning.py
|-- feature_classifier_experiments.py
|-- optimize_embedding_ensemble.py
|-- evaluate_final_models.py
|-- train_unet.py
|-- optimize_unet_threshold.py
|-- generate_final_figures.py
|-- show_results.py
|
|-- model_improvement_comparison.png
|-- confusion_matrix.png
|-- tbi-det-pro-2.ipynb
```

Large dataset and experiment directories may contain additional checkpoints, histories, manifests, plots, and logs.

---

## Installation

### Prerequisites

- Python 3.12 recommended
- Sufficient disk space for datasets and model artifacts
- At least 8 GB RAM for inference; more memory or a GPU is recommended for retraining

### 1. Enter the project

```bash
cd /Users/ahsaansheikh/Desktop/tbi-detection-project
```

### 2. Create and activate a virtual environment

If `.venv` already exists:

```bash
source .venv/bin/activate
```

To create it from scratch:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 3. Install dependencies

Install the runtime and experiment dependencies:

```bash
python -m pip install tensorflow fastapi 'uvicorn[standard]' python-multipart numpy pillow pydicom scikit-learn joblib matplotlib pandas
```

### 4. Confirm required models

```bash
ls -lh tbi_efficientnetb0_best.keras
ls -lh unet_best_v2.keras
ls -lh experiments/exp08_embedding_ensemble/selected_ensemble.joblib
ls -lh experiments/final_frozen_test/auxiliary_classifiers.joblib
```

---

## Running and using the application

### Start the server

```bash
source .venv/bin/activate
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Wait until startup completes. TensorFlow model loading can take some time.

For ordinary development, avoid `--reload` unless needed because the reloader can temporarily load large models more than once.

### Check readiness

Open:

```text
http://127.0.0.1:8000/api/health
```

The health response should show the CNN, ensemble, binary classifier, and U-Net as loaded.

### Use the browser application

1. Open `http://127.0.0.1:8000`.
2. Upload a supported CT image or DICOM file.
3. Review the preview.
4. Select **Analyze Scan**.
5. Review:
   - TBI/Normal screening result
   - TBI probability
   - Six-class injury type
   - Prediction confidence
   - Class-probability distribution
   - Segmentation overlay, when produced

### Stop the server

In the terminal running Uvicorn, press:

```text
Ctrl+C
```

### Direct API example

```bash
curl -X POST \
  -F "file=@/absolute/path/to/scan.png" \
  http://127.0.0.1:8000/api/predict
```

---

## Reproducing experiments

Training can be slow on an 8 GB CPU-only system. Run one major experiment at a time and monitor memory usage.

Use `--help` before a script to see its current supported arguments:

```bash
python classification_experiments.py --help
python feature_classifier_experiments.py --help
python optimize_embedding_ensemble.py --help
```

### Create/recreate classification splits

```bash
python split_tbi6000_dataset.py
```

Review the generated manifest and counts before training. Do not recreate the split after beginning final model comparison unless intentionally starting a new experiment protocol.

### Train/evaluate the original EfficientNetB0 pipeline

```bash
python tbi_detection_model.py
```

### Run transfer-learning experiments

```bash
python classification_experiments.py --help
```

The script supports the architecture, preprocessing, resolution, training-stage, and optional test-evaluation settings used in the research runs. Keep test evaluation disabled while selecting models.

### Continue the existing EfficientNetB0 carefully

```bash
python continue_b0_finetuning.py
```

### Extract embeddings and compare classical heads

```bash
python feature_classifier_experiments.py \
  --name exp03_feature_heads_nearest \
  --model tbi_efficientnetb0_best.keras \
  --interpolation nearest
```

### Optimize the embedding ensemble on validation data

```bash
python optimize_embedding_ensemble.py
```

### Run the locked final evaluation

```bash
python evaluate_final_models.py
```

This script reads the frozen test split and writes the final evidence. It should be run only after model and weight selection is complete.

### Train U-Net

```bash
python train_unet.py
```

### Evaluate segmentation thresholds

```bash
python optimize_unet_threshold.py
```

### Regenerate final figures

```bash
python generate_final_figures.py
```

---

## Evidence and generated artifacts

| File | What it contains |
|---|---|
| `MODEL_IMPROVEMENT_REPORT.md` | Detailed experiment narrative and decisions |
| `experiments/experiment_summary.csv` | Consolidated validation experiment results |
| `experiments/final_frozen_test/metrics.json` | Exact final metrics for baseline and selected models |
| `experiments/final_frozen_test/predictions.csv` | Per-image frozen-test predictions and probabilities |
| `model_improvement_comparison.png` | Baseline vs improved metric chart |
| `confusion_matrix.png` | Selected model confusion matrix |
| `frontend/confusion_matrix.png` | Frontend copy of the confusion matrix |

These files are the primary evidence for faculty review. The exact machine-readable metrics should be taken from `metrics.json`, rather than manually transcribing values.

---

## Verification performed

The final implementation was checked for:

- Complete baseline evaluation on all 626 test images
- Validation-only model and ensemble selection
- Frozen-test evaluation through a dedicated script
- Saved per-image predictions
- Probability outputs summing to approximately 1.0
- Successful loading of the final neural and joblib artifacts
- Direct backend prediction with real model output
- Binary status and six-class injury type in the API response
- Segmentation mask/overlay response
- Frontend rendering of confidence and class probabilities
- Updated confusion matrix and comparison figure
- Removal of disposable cache files only

---

## Known limitations

1. **Six-class performance remains limited.**  
   Accuracy improved from 44.25% to 45.69%, but the system still confuses visually similar hemorrhage types.

2. **The dataset is image-level, not verified patient-level.**  
   Without patient/study identifiers, patient-level leakage cannot be ruled out.

3. **One exact train/test duplicate was detected.**  
   It remains documented in the fixed benchmark instead of being removed after seeing the test split.

4. **PNG data cannot support genuine HU windowing.**  
   Clinical brain/subdural/bone window experiments require original DICOM pixel data and metadata.

5. **Segmentation quality is poor.**  
   The U-Net result is suitable only as an experimental visualization in its current form.

6. **Binary specificity is weaker than sensitivity.**  
   The binary model detects most TBI images but incorrectly marks many normal images as TBI.

7. **Hardware constrained the search.**  
   Full-resolution and larger-architecture searches were limited by CPU speed and 8 GB memory.

8. **This is not clinically validated.**  
   Results come from the available local dataset and must not be generalized to hospitals, scanners, or populations without external validation.

---

## Recommended future improvements

The most promising next work is data-focused rather than simply adding more architectures:

1. Acquire original DICOM studies with reliable Hounsfield Units.
2. Split by patient/study before any training.
3. Remove duplicates before freezing a new benchmark.
4. Review and correct uncertain class labels with clinical guidance.
5. Use 2.5-D or 3-D context from adjacent CT slices.
6. Pretrain on a larger head-CT hemorrhage dataset.
7. Use lesion localization or segmentation as an auxiliary training task.
8. Rebalance and calibrate the binary classifier to improve normal specificity.
9. Train the most promising models across multiple random seeds.
10. Report confidence intervals and external-test performance.
11. Improve segmentation annotations and use foreground-aware sampling or loss functions.
12. Revisit 320/384 input sizes and stronger ensembles on GPU hardware.

---

## Final conclusion

The project now provides a complete, working, and evidence-backed CT analysis pipeline:

```text
Upload CT
   |
   v
Normal/TBI screening
   |
   v
Six-class injury prediction
   |
   v
Confidence and class probabilities
   |
   v
Experimental U-Net segmentation
```

The production classifier was replaced only after validation-based selection and frozen-test verification. The final six-class ensemble improves the original EfficientNetB0 baseline from **44.25% to 45.69% accuracy**, while the separate binary model achieves **87.86% accuracy** and **93.92% TBI recall**.

The outcome is a genuine measured improvement—not a fabricated score—and the saved metrics, predictions, models, code, and figures provide a reproducible technical record of the work.
