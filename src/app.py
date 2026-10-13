from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from typing import Any, List, Optional
from sqlalchemy.orm import Session, joinedload
from src.database import Base, get_db, get_engine
from src.models_db import Farmer, SmsLog
from src.localization import build_sms_message
from src.sms_service import normalize_msisdn, send_sms_notification
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
    adm1: str = Field(
        "Oromia",
        description="Administrative Region: Oromia | Amhara | Tigray | SNNPRS | Afar",
        example="Oromia",
    )
    soil_type: str = Field(
        "Vertisols",
        description="Soil classification: Vertisols | Nitisols | Cambisols | Luvisols | Unknown",
        example="Vertisols",
    )
    fertilizer_N: Optional[float] = Field(
        64.0, ge=0, le=300, description="Elemental N applied kg/ha", example=64.0
    )
    fertilizer_P: Optional[float] = Field(
        20.0, ge=0, le=100, description="Elemental P applied kg/ha", example=20.0
    )
    fertilizer_S: Optional[float] = Field(
        7.0, ge=0, le=80, description="Elemental S applied kg/ha", example=7.0
    )
    fertilizer_K: Optional[float] = Field(
        0.0, ge=0, le=100, description="Elemental K applied kg/ha", example=0.0
    )
    soil_SOC: Optional[float] = Field(
        2.0, ge=0.1, le=10.0, description="Soil Organic Carbon %", example=2.0
    )
    soil_clay: Optional[float] = Field(
        45.0, ge=0.0, le=100.0, description="Soil clay percentage %", example=45.0
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
        try:
            from src.seed import seed_data

            seed_data()
            print("Farmer SMS tables ready.")
        except Exception as db_exc:
            print(f"Farmer DB not available yet: {db_exc}")
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


from src.fertilizer_service import wheat_service, crop_fertilizer_recommender


class FertilizerRequest(BaseModel):
    N: float = Field(64.0, ge=0, le=300, description="Elemental Nitrogen applied (kg N/ha)", example=64.0)
    P: float = Field(20.0, ge=0, le=100, description="Elemental Phosphorus applied (kg P/ha)", example=20.0)
    K: float = Field(0.0, ge=0, le=100, description="Elemental Potassium applied (kg K/ha)", example=0.0)
    S: float = Field(7.0, ge=0, le=80, description="Elemental Sulfur applied (kg S/ha)", example=7.0)
    latitude: float = Field(8.5, ge=3.0, le=15.0, description="Latitude (°N)", example=8.5)
    longitude: float = Field(39.2, ge=33.0, le=48.0, description="Longitude (°E)", example=39.2)
    adm1: str = Field("Oromia", description="Administrative Region (Oromia, Amhara, Tigray, SNNPRS, Afar)", example="Oromia")
    soil_pH: Optional[float] = Field(6.5, ge=3.5, le=9.5, description="Soil pH", example=6.5)
    soil_SOC: Optional[float] = Field(2.0, ge=0.2, le=8.0, description="Soil Organic Carbon %", example=2.0)
    soil_P: Optional[float] = Field(15.0, ge=0.0, le=100.0, description="Available Soil P (mg/kg / ppm)", example=15.0)
    soil_clay: Optional[float] = Field(45.0, ge=0.0, le=100.0, description="Soil Clay %", example=45.0)
    soil_sand: Optional[float] = Field(20.0, ge=0.0, le=100.0, description="Soil Sand %", example=20.0)
    soil_type: str = Field("Vertisols", description="Soil type classification", example="Vertisols")
    simulate_curve: bool = Field(True, description="Whether to include simulated N-response curve", example=True)


@app.post("/recommend-fertilizer")
def recommend_fertilizer(req: FertilizerRequest):
    """Predict wheat grain yield and return commercial fertilizer recommendations and coverage notices."""
    try:
        pred_res = wheat_service.predict(
            N=req.N,
            P=req.P,
            K=req.K,
            S=req.S,
            latitude=req.latitude,
            longitude=req.longitude,
            adm1=req.adm1,
            soil_pH=req.soil_pH,
            soil_SOC=req.soil_SOC,
            soil_P=req.soil_P,
            soil_clay=req.soil_clay,
            soil_sand=req.soil_sand,
            soil_type=req.soil_type,
        )

        curve_data = None
        if req.simulate_curve:
            curve_data = wheat_service.simulate_n_response(
                P=req.P,
                K=req.K,
                S=req.S,
                latitude=req.latitude,
                longitude=req.longitude,
                adm1=req.adm1,
                soil_pH=req.soil_pH,
                soil_SOC=req.soil_SOC,
            )

        pred_res["simulated_n_response_curve"] = curve_data
        return pred_res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fertilizer recommendation error: {str(e)}")


@app.get("/fertilizer-presets")
def get_fertilizer_presets(adm1: str = "Oromia", soil_pH: float = 6.5, soil_SOC: float = 2.0, soil_P: float = 15.0):
    """Returns curated Ethiopian extension fertilizer packages and simulations for smallholder farmers."""
    strategies = ["low_input", "balanced_recommended", "high_potential"]
    packages = []
    for strat in strategies:
        pkg = wheat_service.recommend_optimal_treatment(
            adm1=adm1,
            soil_pH=soil_pH,
            soil_SOC=soil_SOC,
            soil_P=soil_P,
            target_strategy=strat,
        )
        packages.append(pkg)
    return {"adm1": adm1, "packages": packages}


class CropFertilizerRequestModel(BaseModel):
    crop: str = Field(..., description="Specific crop (Wheat, Maize, Teff, Barley, Bean, Potato, ...)", example="Wheat")
    soil_N_ppm: float = Field(150.0, ge=0, le=700, description="Soil available Nitrogen (mg/kg / ppm)", example=150.0)
    soil_P_ppm: float = Field(15.0, ge=0, le=200, description="Soil available Phosphorus (mg/kg / ppm)", example=15.0)
    soil_K_ppm: float = Field(280.0, ge=0, le=2200, description="Soil available Potassium (mg/kg)", example=280.0)
    soil_pH: float = Field(6.5, ge=3.0, le=10.0, description="Soil pH", example=6.5)
    soil_SOC_pct: float = Field(2.0, ge=0.1, le=10.0, description="Soil Organic Carbon %", example=2.0)
    strategy: str = Field("balanced", description="Fertilizer strategy: low | balanced | high", example="balanced")
    adm1: str = Field("Oromia", description="Administrative Region", example="Oromia")


def fertilizer_from_soil(crop: str, input_data: CropInput, strategy: str = "balanced") -> dict:
    """Build a Urea + DAP package from soil-test values and a specific crop."""
    return crop_fertilizer_recommender.recommend(
        crop=crop,
        soil_N_ppm=input_data.N,
        soil_P_ppm=input_data.P,
        soil_K_ppm=input_data.K,
        soil_pH=input_data.ph,
        soil_SOC_pct=input_data.soil_SOC or 2.0,
        strategy=strategy,
        adm1=input_data.adm1 or "Oromia",
    )


@app.post("/recommend-crop-fertilizer")
def recommend_crop_fertilizer(req: CropFertilizerRequestModel):
    """Return scientific Urea + DAP fertilizer recommendation for a specific crop,
    adjusted for soil test results (N, P, pH, SOC). Includes split application timing."""
    try:
        result = crop_fertilizer_recommender.recommend(
            crop=req.crop,
            soil_N_ppm=req.soil_N_ppm,
            soil_P_ppm=req.soil_P_ppm,
            soil_K_ppm=req.soil_K_ppm,
            soil_pH=req.soil_pH,
            soil_SOC_pct=req.soil_SOC_pct,
            strategy=req.strategy,
            adm1=req.adm1,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Crop fertilizer recommendation error: {str(e)}")


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

        fertilizer_packages = []
        for crop_name in best_crops_list:
            try:
                fertilizer_packages.append(
                    fertilizer_from_soil(crop_name, input_data, strategy="balanced")
                )
            except Exception:
                continue

        response = {
            "recommended_crop": best_crop,
            "crops_in_group": best_crops_list,
            "explanation": CLASS_EXPLANATIONS.get(best_crop, ""),
            "confidence_pct": round(float(probs[best_idx]) * 100, 2),
            "top3_recommendations": top3_details,
            "fertilizer_for_group": fertilizer_packages,
            "primary_fertilizer": fertilizer_packages[0] if fertilizer_packages else None,
        }
        return response
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict-unified")
def predict_unified(input_data: CropInput):
    """Predict crop suitability and crop-specific Urea + DAP rates from the same soil input."""
    crop_res = predict_crop(input_data)
    primary = crop_res.get("primary_fertilizer")
    if primary is None:
        crops = crop_res.get("crops_in_group") or ["Wheat"]
        primary = fertilizer_from_soil(crops[0], input_data, strategy="balanced")

    return {
        "selected_area": {
            "adm1": input_data.adm1,
            "soil_type": input_data.soil_type,
            "soil_color": input_data.soil_color,
            "altitude_m": input_data.altitude_m,
            "rainfall": input_data.rainfall,
            "ph": input_data.ph,
        },
        "crop_suitability": crop_res,
        "fertilizer_recommendation": primary,
        "fertilizer_for_group": crop_res.get("fertilizer_for_group", []),
    }


@app.get("/health")
def health():
    return {
        "status": "healthy" if model is not None else "crop model not loaded",
        "crop_model_loaded": model is not None,
        "fertilizer_model_engine": wheat_service.model is not None and "ml_model" or "empirical_trial_prior",
        "n_crop_features": len(feature_cols) if feature_cols else 0,
    }


class FarmerCreate(BaseModel):
    full_name: str = Field(..., min_length=2, example="Abebe Kebede")
    phone_number: str = Field(..., example="0911234567")
    region: str = Field("Oromia", example="Oromia")
    language_pref: str = Field("am", description="am | om", example="am")


class SmsPreviewRequest(BaseModel):
    farmer_id: Optional[int] = None
    language: Optional[str] = None
    farmer_name: Optional[str] = "Farmer"
    crop_name: str = "Wheat"
    crop_group: Optional[str] = None
    dap_kg: float = 0
    urea_kg: float = 0
    n_kg: Optional[float] = None
    p_kg: Optional[float] = None
    is_legume: bool = False
    confidence_pct: Optional[float] = None
    splits: Optional[List[dict[str, Any]]] = None


class SendSmsRequest(BaseModel):
    farmer_id: Optional[int] = None
    send_to_all: bool = False
    crop_name: str
    crop_group: Optional[str] = None
    dap_kg: float
    urea_kg: float
    n_kg: Optional[float] = None
    p_kg: Optional[float] = None
    is_legume: bool = False
    confidence_pct: Optional[float] = None
    splits: Optional[List[dict[str, Any]]] = None
    message_override: Optional[str] = None


def _farmer_dict(farmer: Farmer) -> dict:
    return {
        "id": farmer.id,
        "name": farmer.full_name,
        "phone": farmer.phone_number,
        "language": farmer.language_pref,
        "region": farmer.region,
    }


def _select_farmers(payload: SendSmsRequest, db: Session) -> list:
    if payload.send_to_all:
        return db.query(Farmer).all()
    if payload.farmer_id:
        farmer = db.query(Farmer).filter(Farmer.id == payload.farmer_id).first()
        return [farmer] if farmer else []
    return []


@app.get("/farmers/list")
def get_farmers_list(db: Session = Depends(get_db)):
    try:
        farmers = db.query(Farmer).order_by(Farmer.id).all()
        return [_farmer_dict(f) for f in farmers]
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {exc}")


@app.get("/farmers/sms-logs")
def list_sms_logs(limit: int = 15, db: Session = Depends(get_db)):
    logs = (
        db.query(SmsLog)
        .options(joinedload(SmsLog.farmer))
        .order_by(SmsLog.sent_at.desc())
        .limit(max(1, min(limit, 50)))
        .all()
    )
    return [
        {
            "id": row.id,
            "farmer_id": row.farmer_id,
            "farmer": row.farmer.full_name if row.farmer else None,
            "crop_name": row.crop_name,
            "message_body": row.message_body,
            "gateway_status": row.gateway_status,
            "sent_at": row.sent_at.isoformat() if row.sent_at else None,
        }
        for row in logs
    ]


@app.post("/farmers/")
def register_farmer(body: FarmerCreate, db: Session = Depends(get_db)):
    phone = normalize_msisdn(body.phone_number)
    if not phone.startswith("251"):
        raise HTTPException(status_code=400, detail="Phone must be Ethiopian (09... or 2519...)")
    existing = db.query(Farmer).filter(Farmer.phone_number == phone).first()
    if existing:
        raise HTTPException(status_code=409, detail="This phone number is already registered")
    lang = (body.language_pref or "am").lower()
    if lang not in ("am", "om", "en"):
        lang = "am"
    farmer = Farmer(
        full_name=body.full_name.strip(),
        phone_number=phone,
        region=body.region.strip() or "Oromia",
        language_pref=lang,
    )
    db.add(farmer)
    db.commit()
    db.refresh(farmer)
    return {"status": "success", "farmer": _farmer_dict(farmer)}


@app.post("/farmers/preview-sms")
def preview_farmer_sms(payload: SmsPreviewRequest, db: Session = Depends(get_db)):
    """Build the localized SMS from the latest crop/fertilizer result without sending."""
    farmer = None
    if payload.farmer_id:
        farmer = db.query(Farmer).filter(Farmer.id == payload.farmer_id).first()
        if not farmer:
            raise HTTPException(status_code=404, detail="Farmer not found")
    lang = farmer.language_pref if farmer else (payload.language or "am")
    name = farmer.full_name if farmer else (payload.farmer_name or "Farmer")
    message = build_sms_message(
        lang=lang,
        name=name,
        crop=payload.crop_name,
        dap_kg=payload.dap_kg,
        urea_kg=payload.urea_kg,
        crop_group=payload.crop_group,
        splits=payload.splits,
        is_legume=payload.is_legume,
        confidence_pct=payload.confidence_pct,
        n_kg=payload.n_kg,
        p_kg=payload.p_kg,
    )
    return {
        "language": lang,
        "message": message,
        "farmer": _farmer_dict(farmer) if farmer else None,
        "crop_name": payload.crop_name,
        "dap_kg": payload.dap_kg,
        "urea_kg": payload.urea_kg,
    }


@app.post("/farmers/send-sms")
def send_custom_recommendation_sms(payload: SendSmsRequest, db: Session = Depends(get_db)):
    """Send the crop + Urea/DAP recommendation in the farmer's language (am/om)."""
    if payload.dap_kg <= 0 and payload.urea_kg <= 0 and not payload.message_override:
        raise HTTPException(
            status_code=400,
            detail="Run crop/fertilizer recommendation first so DAP and Urea rates are available.",
        )

    target_farmers = _select_farmers(payload, db)
    if not target_farmers:
        raise HTTPException(status_code=404, detail="No target farmers found")

    results = []
    sent = 0
    for farmer in target_farmers:
        edited = (payload.message_override or "").strip()
        if edited and not payload.send_to_all:
            msg = edited
        else:
            msg = build_sms_message(
                lang=farmer.language_pref,
                name=farmer.full_name,
                crop=payload.crop_name,
                dap_kg=payload.dap_kg,
                urea_kg=payload.urea_kg,
                crop_group=payload.crop_group,
                splits=payload.splits,
                is_legume=payload.is_legume,
                confidence_pct=payload.confidence_pct,
                n_kg=payload.n_kg,
                p_kg=payload.p_kg,
            )
        gateway_res = send_sms_notification(farmer.phone_number, msg)
        status_label = "success" if gateway_res.get("ok") else "error"
        db.add(
            SmsLog(
                farmer_id=farmer.id,
                crop_name=payload.crop_name,
                message_body=msg,
                gateway_status=status_label,
            )
        )
        ok = bool(gateway_res.get("ok"))
        if ok:
            sent += 1
        results.append(
            {
                "farmer": farmer.full_name,
                "phone": farmer.phone_number,
                "language": farmer.language_pref,
                "ok": ok,
                "message": msg,
                "chars": len(msg),
                "response": gateway_res,
            }
        )

    db.commit()
    return {
        "status": "success" if sent else "error",
        "dispatched_count": sent,
        "attempted": len(results),
        "details": results,
    }