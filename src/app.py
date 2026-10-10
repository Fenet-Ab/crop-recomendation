from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
FEATURE_COLS_PATH = BASE_DIR / "data" / "processed" / "feature_cols.csv"

app = FastAPI(
    title="Ethiopian Crop Recommendation System",
    description="Recommends the best crop group for given soil and climate conditions in Ethiopia (based on real localized data).",
    version="2.1",
)

# Serve static files (HTML, future CSS/JS/images)
app.mount("/static", StaticFiles(directory="static", html=True), name="static")


@app.get("/")
def read_root():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/static/index.html")


model = None
scaler = None
le = None
feature_cols = None

GROUP_CROPS = {
    "Major_Cereals": ["Teff", "Maize", "Wheat", "Barley"],
    "Cereals": ["Dagussa", "Sorghum"],
    "Pulses": ["Bean", "Pea"],
    "Specialty": ["Niger seed", "Potato", "Red Pepper", "Fallow"],
}

CLASS_EXPLANATIONS = {
    "Major_Cereals": "Major Cereals – staple grains like Teff (for injera), Maize, Wheat, and Barley. Highly suitable for most Ethiopian farms.",
    "Cereals": "Minor Cereals – drought-resistant alternatives like Dagussa and Sorghum, good for drier areas.",
    "Pulses": "Pulses – nutritious legumes (Bean, Pea) that fix nitrogen and improve soil health.",
    "Specialty": "Specialty Crops – high-value or niche options (Niger seed, Potato, Red Pepper, Fallow rotation).",
}

SOIL_COLOR_CHOICES = ("black", "brown", "gray", "red", "other")


def normalize_soil_color(value: str) -> str:
    s = str(value).lower().strip()
    if "black" in s:
        return "black"
    if "red" in s:
        return "red"
    if "gray" in s or "grey" in s:
        return "gray"
    if "brown" in s or "yellow" in s:
        return "brown"
    return "other"


class CropInput(BaseModel):
    N: float = Field(
        ..., ge=0, le=700, description="Nitrogen mg/kg / ppm (0-700)", example=180
    )
    P: float = Field(..., ge=0, le=800, description="Phosphorus (0-800)", example=40)
    K: float = Field(..., ge=0, le=2200, description="Potassium (0-2200)", example=300)
    ph: float = Field(..., ge=3, le=10, description="Soil pH", example=6.5)
    temperature: float = Field(
        ..., ge=5, le=45, description="Average Temperature °C", example=22.0
    )
    humidity: float = Field(
        ..., ge=35, le=95, description="Relative Humidity % (35-95)", example=65.0
    )
    rainfall: float = Field(
        ..., ge=0, le=2500, description="Rainfall (seasonal sum / annual proxy)", example=25.0
    )
    altitude_m: float = Field(
        ..., ge=0, le=4500, description="Altitude (meters)", example=2400.0
    )
    Zn: float = Field(..., ge=0, le=50, description="Zinc level", example=5.0)
    S: float = Field(..., ge=0, le=100, description="Sulfur level", example=20.0)
    soil_moisture: float = Field(
        ..., ge=0, le=1, description="Topsoil moisture (0-1)", example=0.6
    )
    soil_color: str = Field(
        "brown",
        description="Soil color: black | brown | gray | red | other",
        example="brown",
    )


def build_feature_row(input_data: CropInput) -> np.ndarray:
    """Build a single feature row aligned to the training feature_cols order."""
    color = normalize_soil_color(input_data.soil_color)
    values = {
        "N": input_data.N,
        "P": input_data.P,
        "K": input_data.K,
        "ph": input_data.ph,
        "temperature": input_data.temperature,
        "humidity": input_data.humidity,
        "rainfall": input_data.rainfall,
        "altitude_m": input_data.altitude_m,
        "Zn": input_data.Zn,
        "S": input_data.S,
        "soil_moisture": input_data.soil_moisture,
    }
    for choice in SOIL_COLOR_CHOICES:
        values[f"soilcolor_{choice}"] = 1.0 if color == choice else 0.0

    row = [float(values.get(col, 0.0)) for col in feature_cols]
    return np.array([row])


@app.on_event("startup")
def load_models():
    global model, scaler, le, feature_cols
    try:
        model_path = MODELS_DIR / "best_crop_model.pkl"
        scaler_path = MODELS_DIR / "scaler_merged.pkl"
        le_path = MODELS_DIR / "label_encoder_merged.pkl"

        print(f"DEBUG: Loading from: {MODELS_DIR.absolute()}")
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        le = joblib.load(le_path)

        if FEATURE_COLS_PATH.exists():
            cols = pd.read_csv(FEATURE_COLS_PATH)["feature"].tolist()
        else:
            cols = [
                "N",
                "P",
                "K",
                "ph",
                "temperature",
                "humidity",
                "rainfall",
                "altitude_m",
                "Zn",
                "S",
                "soil_moisture",
            ]

        expected_n = getattr(scaler, "n_features_in_", len(cols))
        numeric_defaults = [
            "N", "P", "K", "ph", "temperature", "humidity",
            "rainfall", "altitude_m", "Zn", "S", "soil_moisture"
        ]
        if len(cols) == expected_n:
            feature_cols = cols
        elif len(numeric_defaults) == expected_n:
            feature_cols = numeric_defaults
        else:
            feature_cols = cols[:expected_n]

        print(f"Feature columns ({len(feature_cols)}): {feature_cols}")
        print("All models loaded successfully!")
    except Exception as e:
        print(f"CRITICAL: Failed to load models: {e}")
        raise RuntimeError(f"Model loading failed: {e}")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = []
    for error in exc.errors():
        field = " → ".join(str(loc) for loc in error["loc"])
        msg = error["msg"]
        errors.append(f"{field}: {msg}")

    friendly_message = (
        "Input validation failed:\n"
        + "\n".join(errors)
        + "\n\nPlease check all fields and try again."
    )
    return JSONResponse(status_code=422, content={"detail": friendly_message})


@app.post("/predict")
def predict_crop(input_data: CropInput):
    if model is None or scaler is None or le is None or feature_cols is None:
        raise HTTPException(status_code=500, detail="Server error: Models not loaded.")

    try:
        input_array = build_feature_row(input_data)
        if input_array.shape[1] != getattr(scaler, "n_features_in_", input_array.shape[1]):
            raise HTTPException(
                status_code=500,
                detail=(
                    f"Feature mismatch: built {input_array.shape[1]} features but "
                    f"scaler expects {scaler.n_features_in_}. "
                    "Re-run notebooks 02→05 after the data-artifact fixes."
                ),
            )

        scaled_input = scaler.transform(input_array)
        probs = model.predict_proba(scaled_input)[0]

        top3_idx = np.argsort(probs)[-3:][::-1]
        top3_crops = le.inverse_transform(top3_idx)
        top3_probs = probs[top3_idx]

        best_idx = np.argmax(probs)
        best_crop = le.inverse_transform([best_idx])[0]
        best_crops_list = GROUP_CROPS.get(best_crop, [])

        top3_details = []
        for crop, prob in zip(top3_crops, top3_probs):
            top3_details.append(
                {
                    "crop": crop,
                    "crops_in_group": GROUP_CROPS.get(crop, []),
                    "probability_pct": round(float(prob) * 100, 2),
                }
            )

        response = {
            "recommended_crop": best_crop,
            "crops_in_group": best_crops_list,
            "explanation": CLASS_EXPLANATIONS.get(best_crop, ""),
            "confidence_pct": round(float(probs[best_idx]) * 100, 2),
            "top3_recommendations": top3_details,
        }
        return response
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/health")
def health():
    return {
        "status": "healthy" if model is not None else "models not loaded",
        "n_features": len(feature_cols) if feature_cols else 0,
    }
