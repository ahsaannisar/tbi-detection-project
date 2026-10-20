import base64
import io
import os
import sys
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
import joblib
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps

try:
    import pydicom
    from pydicom import dcmread
except Exception:  # pragma: no cover
    pydicom = None
    dcmread = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tbi_backend")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
BACKEND_STATIC_DIR = os.path.join(BASE_DIR, "backend", "static")
CNN_MODEL_PATH = os.path.join(BASE_DIR, "tbi_efficientnetb0_best.keras")
UNET_MODEL_PATH = os.path.join(BASE_DIR, "unet_best_v2.keras")
ENSEMBLE_PATH = os.path.join(BASE_DIR, "experiments", "exp08_embedding_ensemble", "selected_ensemble.joblib")
AUXILIARY_PATH = os.path.join(BASE_DIR, "experiments", "final_frozen_test", "auxiliary_classifiers.joblib")

CLASS_NAMES = [
    "Epidural",
    "Intraparenchymal",
    "Intraventricular",
    "Normal",
    "Subarachnoid",
    "Subdural",
]
UNET_THRESHOLD = 0.25

app = FastAPI(title="TBI Detection Research Prototype", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure static directories exist
os.makedirs(BACKEND_STATIC_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
app.mount("/assets", StaticFiles(directory=BACKEND_STATIC_DIR), name="assets")

# Global model references
cnn_model = None
cnn_embedder = None
ensemble_artifact = None
binary_classifier = None
unet_model = None


@app.on_event("startup")
def load_models():
    global cnn_model, cnn_embedder, ensemble_artifact, binary_classifier, unet_model
    logger.info("Loading saved models...")
    if os.path.exists(CNN_MODEL_PATH):
        cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH, compile=False)
        cnn_embedder = tf.keras.Model(cnn_model.input, cnn_model.layers[-2].output)
        logger.info(f"CNN model loaded successfully from {CNN_MODEL_PATH}")
    else:
        logger.error(f"CNN model file not found at {CNN_MODEL_PATH}")

    if os.path.exists(ENSEMBLE_PATH):
        ensemble_artifact = joblib.load(ENSEMBLE_PATH)
        logger.info(f"Validation-selected ensemble loaded from {ENSEMBLE_PATH}")
    else:
        logger.error(f"Ensemble artifact not found at {ENSEMBLE_PATH}")

    if os.path.exists(AUXILIARY_PATH):
        auxiliary = joblib.load(AUXILIARY_PATH)
        binary_classifier = auxiliary["binary"]
        logger.info(f"Binary normal-vs-TBI classifier loaded from {AUXILIARY_PATH}")
    else:
        logger.error(f"Auxiliary classifier artifact not found at {AUXILIARY_PATH}")

    if os.path.exists(UNET_MODEL_PATH):
        unet_model = tf.keras.models.load_model(UNET_MODEL_PATH, compile=False)
        logger.info(f"U-Net model loaded successfully from {UNET_MODEL_PATH}")
    else:
        logger.error(f"U-Net model file not found at {UNET_MODEL_PATH}")


@app.get("/health")
def health() -> Dict[str, str]:
    return {
        "status": "ok",
        "backend": "online",
        "cnn_model": "loaded" if cnn_model is not None else "not_loaded",
        "ensemble": "loaded" if ensemble_artifact is not None else "not_loaded",
        "binary_model": "loaded" if binary_classifier is not None else "not_loaded",
        "unet_model": "loaded" if unet_model is not None else "not_loaded",
    }


def _process_dicom_bytes(file_bytes: bytes) -> Tuple[Image.Image, Tuple[int, int]]:
    if pydicom is None or dcmread is None:
        raise HTTPException(status_code=400, detail="DICOM support (pydicom) is not available.")

    try:
        ds = dcmread(io.BytesIO(file_bytes))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Unable to parse DICOM file: {exc}") from exc

    if not hasattr(ds, "pixel_array") or ds.pixel_array is None:
        raise HTTPException(status_code=400, detail="DICOM file contains no pixel data.")

    arr = ds.pixel_array.astype(np.float32)
    if arr.ndim == 3:
        arr = arr[0]

    # Apply rescale slope and intercept if present
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    arr = arr * slope + intercept

    # Perform CT brain windowing (Window Width: 80, Window Level: 40 => range 0..80 HU)
    vmin, vmax = 0.0, 80.0
    if arr.max() > 80.0 or arr.min() < 0.0:
        arr_clipped = np.clip(arr, vmin, vmax)
        arr_norm = ((arr_clipped - vmin) / (vmax - vmin) * 255.0).astype(np.uint8)
    else:
        arr_norm = np.clip(arr, 0, 255).astype(np.uint8)

    orig_dimensions = (int(arr.shape[1]), int(arr.shape[0]))
    pil_img = Image.fromarray(arr_norm)
    return pil_img, orig_dimensions


def _process_standard_image_bytes(file_bytes: bytes) -> Tuple[Image.Image, Tuple[int, int]]:
    try:
        pil_img = Image.open(io.BytesIO(file_bytes))
        pil_img = ImageOps.exif_transpose(pil_img)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Unable to decode image file: {exc}") from exc

    orig_dimensions = (pil_img.width, pil_img.height)
    return pil_img, orig_dimensions


def _image_to_base64_data_url(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{fmt.lower()};base64,{b64_str}"


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    allowed_exts = {".png", ".jpg", ".jpeg", ".dcm"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension '{ext}'. Allowed extensions: PNG, JPG, JPEG, DCM.",
        )

    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Error reading file bytes: {exc}") from exc

    # Load PIL image and get original dimensions
    if ext == ".dcm":
        pil_img, orig_dims = _process_dicom_bytes(contents)
    else:
        pil_img, orig_dims = _process_standard_image_bytes(contents)

    # Convert to RGB representation for CNN
    rgb_img = pil_img.convert("RGB")

    # Prepare CNN input tensor: target size (224, 224), float32, range [0.0, 255.0] UN-SCALED
    cnn_pil = rgb_img.resize((224, 224), Image.Resampling.NEAREST)
    cnn_array = np.array(cnn_pil, dtype=np.float32)  # Shape (224, 224, 3), range [0, 255]
    cnn_batch = np.expand_dims(cnn_array, axis=0)  # Shape (1, 224, 224, 3)

    if cnn_model is None or cnn_embedder is None or ensemble_artifact is None or binary_classifier is None:
        raise HTTPException(status_code=500, detail="Verified classification ensemble is not fully loaded.")

    try:
        neural_probs = cnn_model.predict(cnn_batch, verbose=0)[0]
        embedding = cnn_embedder.predict(cnn_batch, verbose=0)
        recipe = ensemble_artifact["recipe"]
        tree_model = ensemble_artifact["classifiers"][recipe["tree"]]
        tree_probs = tree_model.predict_proba(embedding)[0]
        neural_weight = float(recipe["neural_weight"])
        cnn_probs_raw = neural_weight * neural_probs + (1.0 - neural_weight) * tree_probs
        class_idx = int(np.argmax(cnn_probs_raw))
        predicted_class_name = CLASS_NAMES[class_idx]
        confidence = float(cnn_probs_raw[class_idx])
        class_probabilities = {name: float(prob) for name, prob in zip(CLASS_NAMES, cnn_probs_raw)}
        abnormal_column = list(binary_classifier.classes_).index(1)
        tbi_probability = float(binary_classifier.predict_proba(embedding)[0, abnormal_column])
        tbi_status = "TBI / ABNORMAL" if tbi_probability >= 0.5 else "NORMAL"
        abnormal_indices = [index for index, name in enumerate(CLASS_NAMES) if name != "Normal"]
        injury_idx = max(abnormal_indices, key=lambda index: cnn_probs_raw[index])
        injury_type = CLASS_NAMES[injury_idx] if tbi_probability >= 0.5 else None
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"CNN classification inference failed: {exc}") from exc

    # REQUIRED DEBUG LOGGING
    logger.info("=" * 60)
    logger.info("INFERENCE DEBUG LOGGING:")
    logger.info(f"Original image size: {orig_dims}")
    logger.info(f"Processed image shape: {cnn_batch.shape}")
    logger.info(f"Processed image dtype: {cnn_batch.dtype}")
    logger.info(f"Prediction array: {cnn_probs_raw.tolist()}")
    logger.info(f"Predicted class index: {class_idx}")
    logger.info(f"Predicted class name: {predicted_class_name}")
    logger.info(f"Binary status: {tbi_status} ({tbi_probability:.6f} abnormal probability)")
    logger.info("All six probabilities:")
    for c_name, c_prob in class_probabilities.items():
        logger.info(f"  - {c_name:<16}: {c_prob * 100:.2f}% ({c_prob:.6f})")
    logger.info("=" * 60)

    # Prepare U-Net input tensor: target size (256, 256), grayscale float32, range [0.0, 1.0] SCALED
    gray_img = pil_img.convert("L").resize((256, 256), Image.Resampling.NEAREST)
    unet_array = np.array(gray_img, dtype=np.float32) / 255.0  # Shape (256, 256), range [0, 1]
    unet_batch = np.expand_dims(np.expand_dims(unet_array, axis=-1), axis=0)  # Shape (1, 256, 256, 1)

    mask_data_url = ""
    overlay_data_url = ""
    positive_pixels = 0
    positive_percentage = 0.0

    if unet_model is not None:
        try:
            unet_out = unet_model.predict(unet_batch, verbose=0)[0, :, :, 0]  # Shape (256, 256)
            binary_mask = (unet_out >= UNET_THRESHOLD).astype(np.uint8)
            positive_pixels = int(np.sum(binary_mask))
            positive_percentage = round(float((positive_pixels / (256 * 256)) * 100), 2)

            # Generate Mask image (255 for positive, 0 for negative)
            mask_img_pil = Image.fromarray(binary_mask * 255).convert("L")
            mask_data_url = _image_to_base64_data_url(mask_img_pil, "PNG")

            # Generate Overlay image (RGB): Original gray image with bright RED lesion overlay
            gray_256_rgb = np.stack([np.array(gray_img)] * 3, axis=-1)
            overlay_array = gray_256_rgb.copy()
            # Highlight lesion pixels in red (R=255, G=40, B=40) with 60/40 blend
            lesion_indices = binary_mask > 0
            overlay_array[lesion_indices, 0] = np.clip(
                gray_256_rgb[lesion_indices, 0] * 0.4 + 255 * 0.6, 0, 255
            ).astype(np.uint8)
            overlay_array[lesion_indices, 1] = np.clip(
                gray_256_rgb[lesion_indices, 1] * 0.4, 0, 255
            ).astype(np.uint8)
            overlay_array[lesion_indices, 2] = np.clip(
                gray_256_rgb[lesion_indices, 2] * 0.4, 0, 255
            ).astype(np.uint8)

            overlay_img_pil = Image.fromarray(overlay_array)
            overlay_data_url = _image_to_base64_data_url(overlay_img_pil, "PNG")

        except Exception as exc:
            logger.error(f"U-Net inference failed: {exc}")

    # Generate original preview data URL
    preview_img = rgb_img.copy()
    preview_img.thumbnail((512, 512))
    original_data_url = _image_to_base64_data_url(preview_img, "PNG")

    # Format JSON response
    response_payload = {
        "prediction": predicted_class_name,
        "confidence": round(confidence, 6),
        "tbi_status": tbi_status,
        "tbi_probability": round(tbi_probability, 6),
        "injury_type": injury_type,
        "class_probabilities": {k: round(v, 6) for k, v in class_probabilities.items()},
        "segmentation": {
            "available": unet_model is not None,
            "threshold": UNET_THRESHOLD,
            "positive_pixels": positive_pixels,
            "positive_percentage": positive_percentage,
            "mask_data_url": mask_data_url,
            "overlay_data_url": overlay_data_url,
            "warning": "Experimental segmentation — current model requires further optimization and is not validated for clinical use.",
        },
        "classification": {
            "predicted_class": predicted_class_name,
            "confidence": round(confidence, 6),
            "tbi_status": tbi_status,
            "tbi_probability": round(tbi_probability, 6),
            "normal_probability": round(1.0 - tbi_probability, 6),
            "injury_type": injury_type,
            "probabilities": {k: round(v, 6) for k, v in class_probabilities.items()},
            "all_classes": CLASS_NAMES,
        },
        "original_image_url": original_data_url,
        "filename": file.filename,
        "image_size": f"{orig_dims[0]} × {orig_dims[1]}",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_info": {
            "cnn": {
                "name": "EfficientNetB0 embedding + ExtraTrees probability ensemble",
                "input_size": "224x224 RGB",
                "classes": CLASS_NAMES,
                "neural_weight": neural_weight,
                "tree_weight": round(1.0 - neural_weight, 6),
                "binary_stage": "ExtraTrees normal-vs-TBI",
                "verified_frozen_test_accuracy": 0.456869,
                "verified_frozen_test_macro_f1": 0.448909,
                "verified_binary_accuracy": 0.878594,
            },
            "unet": {
                "name": "U-Net",
                "input_size": "256x256 grayscale",
                "threshold": UNET_THRESHOLD,
                "experimental": True,
            },
        },
        "disclaimer": "This system is an academic research prototype and is not intended for clinical diagnosis or treatment decisions.",
    }

    return JSONResponse(content=response_payload)


@app.get("/confusion_matrix.png")
@app.get("/static/confusion_matrix.png")
@app.get("/assets/confusion_matrix.png")
async def get_confusion_matrix():
    candidates = [
        os.path.join(BASE_DIR, "confusion_matrix.png"),
        os.path.join(BACKEND_STATIC_DIR, "confusion_matrix.png"),
        os.path.join(FRONTEND_DIR, "confusion_matrix.png"),
    ]
    for cand in candidates:
        if os.path.exists(cand):
            return FileResponse(cand, media_type="image/png")
    raise HTTPException(status_code=404, detail="confusion_matrix.png not found")


@app.get("/training_accuracy.png")
@app.get("/static/training_accuracy.png")
async def get_training_accuracy():
    cand = os.path.join(BASE_DIR, "training_accuracy.png")
    if os.path.exists(cand):
        return FileResponse(cand, media_type="image/png")
    raise HTTPException(status_code=404, detail="training_accuracy.png not found")


@app.get("/training_loss.png")
@app.get("/static/training_loss.png")
async def get_training_loss():
    cand = os.path.join(BASE_DIR, "training_loss.png")
    if os.path.exists(cand):
        return FileResponse(cand, media_type="image/png")
    raise HTTPException(status_code=404, detail="training_loss.png not found")


@app.get("/")
async def serve_index():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "TBI Detection API online"}


@app.get("/{full_path:path}")
async def spa_fallback(full_path: str):
    # Serve requested file if it exists in frontend directory
    file_path = os.path.join(FRONTEND_DIR, full_path.replace("static/", "", 1))
    if os.path.isfile(file_path):
        return FileResponse(file_path)

    # SPA Fallback to index.html
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    raise HTTPException(status_code=404, detail="Page not found")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
