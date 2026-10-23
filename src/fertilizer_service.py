"""
EthioSoil-AI: Wheat Fertilizer Recommendation & Yield Prediction Service.

Provides:
- Machine learning inference for wheat grain yield response given soil, site, and fertilizer treatments.
- Agronomic yield simulation curves (Mitscherlich-type response for N, P, S).
- Conversion to Ethiopian commercial fertilizer bags (Urea, DAP, NPS).
- Regional data coverage guards & agronomic safety notices.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import joblib
except ImportError:
    joblib = None

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "wheat_fertilizer_model.pkl"
METADATA_PATH = MODELS_DIR / "wheat_fertilizer_metadata.json"

# Verified regions represented in wheat trials
REPRESENTED_REGIONS = {"oromia", "amhara", "tigray", "snnprs", "afar"}


class WheatFertilizerService:
    """Service handling wheat fertilizer response prediction, simulation, and recommendations."""

    def __init__(self):
        self.model = None
        self.metadata: Dict[str, Any] = {}
        self.is_ml_loaded = False
        self._load_model_if_available()

    def _load_model_if_available(self) -> None:
        """Attempt to load trained model bundle and metadata."""
        if joblib and MODEL_PATH.exists():
            try:
                self.model = joblib.load(MODEL_PATH)
                self.is_ml_loaded = True
                print(f"[EthioSoil] Loaded trained wheat fertilizer model from {MODEL_PATH}")
            except Exception as e:
                print(f"[EthioSoil] Warning: Could not load {MODEL_PATH}: {e}")
                self.model = None
                self.is_ml_loaded = False

        if METADATA_PATH.exists():
            try:
                with open(METADATA_PATH, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
            except Exception:
                self.metadata = {}

    def _empirical_prior_prediction(
        self,
        N: float,
        P: float,
        K: float,
        S: float,
        soil_pH: Optional[float] = None,
        soil_SOC: Optional[float] = None,
        soil_P: Optional[float] = None,
        adm1: Optional[str] = None,
    ) -> float:
        """Calibrated empirical agronomic prior derived from the 1,535 EthioSoil wheat trial records.

        Mean trial unfertilized control yield = ~1,840 kg/ha.
        Maximum response plateau = ~3,600 - 4,400 kg/ha under optimal balanced fertilization.
        """
        # Base control yield by region
        base_yield = 1840.0
        reg = str(adm1).lower().strip() if adm1 else ""
        if "oromia" in reg:
            base_yield = 1920.0
        elif "amhara" in reg:
            base_yield = 1750.0
        elif "tigray" in reg:
            base_yield = 1580.0

        # Soil factor modifiers
        pH_mod = 1.0
        if soil_pH is not None and not np.isnan(soil_pH):
            # Optimal wheat pH is 6.0 - 7.5; acidic soils (<5.5) fix P and inhibit uptake
            if soil_pH < 5.5:
                pH_mod = 0.82 + 0.18 * max(0.0, (soil_pH - 4.5) / 1.0)
            elif soil_pH > 8.0:
                pH_mod = 0.90

        soc_mod = 1.0
        if soil_SOC is not None and not np.isnan(soil_SOC):
            # Higher SOC gives higher base mineralization
            soc_mod = 0.85 + min(0.35, 0.12 * soil_SOC)

        # Nitrogen response: Diminishing marginal returns (quadratic with plateau)
        # 1 kg N produces ~14 kg grain at low rates, tapering to ~4 kg grain near 120 kg N/ha
        n_gain = 15.5 * N - 0.052 * (N**2)
        n_gain = max(0.0, min(1450.0, n_gain))

        # Phosphorus response: Essential for root development and tillering
        # 1 kg P produces ~28 kg grain at low rates (P is potent)
        p_gain = 28.0 * P - 0.32 * (P**2)
        p_gain = max(0.0, min(950.0, p_gain))

        # NP Synergistic Interaction: Nitrogen response is stunted without P
        # If P is 0, N efficiency drops by 35%
        synergy = min(1.0, (P + 5.0) / 20.0)
        n_gain *= synergy

        # Sulfur response (relevant in Ethiopian volcanic / Vertisol soils)
        s_gain = min(220.0, 6.5 * S)

        # Potassium response (minimal in Ethiopian vertisols with high native clay K)
        k_gain = min(120.0, 3.2 * K)

        pred = (base_yield * soc_mod * pH_mod) + n_gain + p_gain + s_gain + k_gain
        return float(np.clip(pred, 400.0, 6500.0))

    def predict(
        self,
        N: float,
        P: float,
        K: float = 0.0,
        S: float = 0.0,
        latitude: float = 8.5,
        longitude: float = 39.2,
        adm1: str = "Oromia",
        soil_pH: Optional[float] = 6.5,
        soil_SOC: Optional[float] = 2.0,
        soil_P: Optional[float] = 15.0,
        soil_clay: Optional[float] = 45.0,
        soil_sand: Optional[float] = 20.0,
        soil_type: str = "Vertisols",
    ) -> Dict[str, Any]:
        """Predict wheat grain yield (kg/ha) for a given fertilizer treatment and field conditions."""
        # 1. Prediction under requested treatment
        if self.is_ml_loaded and self.model is not None:
            try:
                row = pd.DataFrame([{
                    "N_fertilizer": float(N),
                    "P_fertilizer": float(P),
                    "K_fertilizer": float(K),
                    "S_fertilizer": float(S),
                    "latitude": float(latitude),
                    "longitude": float(longitude),
                    "adm1": str(adm1),
                    "soil_pH": float(soil_pH) if soil_pH is not None else np.nan,
                    "soil_SOC": float(soil_SOC) if soil_SOC is not None else np.nan,
                    "soil_P": float(soil_P) if soil_P is not None else np.nan,
                    "soil_clay": float(soil_clay) if soil_clay is not None else np.nan,
                    "soil_sand": float(soil_sand) if soil_sand is not None else np.nan,
                    "soil_type": str(soil_type),
                    "NP_interaction": float(N * P),
                    "N_sq": float(N**2),
                    "P_sq": float(P**2),
                }])
                pred_yield = float(self.model.predict(row)[0])
                model_used = "trained_groupkfold_ml_model"
            except Exception as e:
                pred_yield = self._empirical_prior_prediction(
                    N=N, P=P, K=K, S=S, soil_pH=soil_pH, soil_SOC=soil_SOC, soil_P=soil_P, adm1=adm1
                )
                model_used = f"empirical_trial_prior (fallback: {e})"
        else:
            pred_yield = self._empirical_prior_prediction(
                N=N, P=P, K=K, S=S, soil_pH=soil_pH, soil_SOC=soil_SOC, soil_P=soil_P, adm1=adm1
            )
            model_used = "empirical_ethiosoil_trial_prior"

        # 2. Baseline unfertilized control (N=0, P=0, K=0, S=0)
        control_yield = self._empirical_prior_prediction(
            N=0.0, P=0.0, K=0.0, S=0.0, soil_pH=soil_pH, soil_SOC=soil_SOC, soil_P=soil_P, adm1=adm1
        )

        yield_gain_kg = max(0.0, pred_yield - control_yield)
        gain_pct = (yield_gain_kg / control_yield * 100.0) if control_yield > 0 else 0.0

        # Commercial bag equivalents
        commercial = self.convert_to_commercial_fertilizers(N=N, P=P, S=S)

        # Coverage notices
        warnings = self.check_coverage_warnings(
            latitude=latitude, longitude=longitude, adm1=adm1, N=N, P=P, K=K, S=S, Zn=0.0
        )

        return {
            "crop": "Wheat (Grain)",
            "unit": "kg/ha",
            "predicted_yield_kg_ha": round(pred_yield, 1),
            "predicted_yield_quintals_ha": round(pred_yield / 100.0, 1),
            "unfertilized_control_yield_kg_ha": round(control_yield, 1),
            "expected_yield_gain_kg_ha": round(yield_gain_kg, 1),
            "yield_gain_percentage": round(gain_pct, 1),
            "treatment_applied": {
                "N_kg_ha": float(N),
                "P_kg_ha": float(P),
                "K_kg_ha": float(K),
                "S_kg_ha": float(S),
            },
            "commercial_product_equivalent": commercial,
            "coverage_warnings": warnings,
            "model_engine": model_used,
        }

    def simulate_n_response(
        self,
        P: float = 20.0,
        K: float = 0.0,
        S: float = 0.0,
        latitude: float = 8.5,
        longitude: float = 39.2,
        adm1: str = "Oromia",
        soil_pH: Optional[float] = 6.5,
        soil_SOC: Optional[float] = 2.0,
        n_steps: List[float] = None,
    ) -> List[Dict[str, Any]]:
        """Simulate yield response across increasing Nitrogen levels (kg N/ha) holding other nutrients fixed."""
        if n_steps is None:
            n_steps = [0.0, 23.0, 46.0, 69.0, 92.0, 115.0, 138.0, 180.0]

        curve = []
        for n_rate in n_steps:
            res = self.predict(
                N=n_rate,
                P=P,
                K=K,
                S=S,
                latitude=latitude,
                longitude=longitude,
                adm1=adm1,
                soil_pH=soil_pH,
                soil_SOC=soil_SOC,
            )
            curve.append({
                "N_rate_kg_ha": n_rate,
                "predicted_yield_kg_ha": res["predicted_yield_kg_ha"],
                "yield_gain_over_control_kg_ha": res["expected_yield_gain_kg_ha"],
                "urea_kg_ha": round(n_rate / 0.46, 1),
            })
        return curve

    def recommend_optimal_treatment(
        self,
        adm1: str = "Oromia",
        soil_pH: Optional[float] = 6.5,
        soil_SOC: Optional[float] = 2.0,
        soil_P: Optional[float] = 15.0,
        target_strategy: str = "balanced_recommended",  # low_input | balanced_recommended | high_potential
    ) -> Dict[str, Any]:
        """Provides tailored agronomic fertilizer recommendations based on soil state and target strategy."""
        # 1. Base agronomic baseline
        if target_strategy == "low_input":
            # 50 kg DAP + 50 kg Urea equivalent
            rec_N = 32.0
            rec_P = 10.0
            rec_S = 0.0
            rec_K = 0.0
            name = "Economic Low-Risk Package"
            desc = "Designed for moisture-constrained or low-budget smallholders (1 bag DAP + 1 bag Urea/ha)."
        elif target_strategy == "high_potential":
            # High highland vertisol potential (e.g., Bale, Arsi, Debre Zeit with good rain)
            rec_N = 92.0
            rec_P = 25.0
            rec_S = 10.0
            rec_K = 0.0
            name = "High-Input Intensive Package"
            desc = "Optimized for high-yield highland zones with reliable moisture and improved bread wheat varieties."
        else:
            # Standard balanced recommendation (e.g. 100 kg NPS + 100 kg Urea)
            rec_N = 64.0
            rec_P = 20.0
            rec_S = 7.0
            rec_K = 0.0
            name = "Balanced Extension Package (NPS + Urea)"
            desc = "Standard Ethiopian MoA & EIAR recommended practice for bread wheat in highlands."

        # Modulate by soil tests if available
        if soil_P is not None and not np.isnan(soil_P):
            if soil_P > 25.0:
                # Soil has ample native P; reduce P input
                rec_P = max(5.0, rec_P * 0.6)
            elif soil_P < 8.0:
                # Highly P-deficient soil; boost P
                rec_P = min(35.0, rec_P * 1.3)

        if soil_pH is not None and not np.isnan(soil_pH) and soil_pH < 5.5:
            # Acid soil note
            desc += " Note: Soil pH is acidic (<5.5); consider agricultural lime (CaCO3) application to reduce P fixation."

        prediction = self.predict(
            N=rec_N,
            P=rec_P,
            K=rec_K,
            S=rec_S,
            adm1=adm1,
            soil_pH=soil_pH,
            soil_SOC=soil_SOC,
            soil_P=soil_P,
        )

        return {
            "strategy": target_strategy,
            "package_name": name,
            "description": desc,
            "recommended_rates": {
                "N_kg_ha": round(rec_N, 1),
                "P_kg_ha": round(rec_P, 1),
                "K_kg_ha": round(rec_K, 1),
                "S_kg_ha": round(rec_S, 1),
            },
            "commercial_bags": prediction["commercial_product_equivalent"],
            "expected_yield": prediction["predicted_yield_kg_ha"],
            "expected_yield_quintals": prediction["predicted_yield_quintals_ha"],
            "yield_gain_kg": prediction["expected_yield_gain_kg_ha"],
            "yield_gain_pct": prediction["yield_gain_percentage"],
        }

    @staticmethod
    def convert_to_commercial_fertilizers(N: float, P: float, S: float) -> Dict[str, Any]:
        """Convert elemental N, P, S rates into real commercial Ethiopian products (Urea, DAP, NPS).

        Product compositions:
        - Urea: 46% N
        - DAP: 18% N, 20.1% elemental P (46% P2O5)
        - NPS: 19% N, 16.7% elemental P (38% P2O5), 7% S
        Standard bag weight: 50 kg per bag.
        """
        # Formulation 1: Traditional DAP + Urea
        # 1. P is delivered by DAP
        dap_kg = (P / 0.201) if P > 0 else 0.0
        n_from_dap = dap_kg * 0.18
        remaining_n = max(0.0, N - n_from_dap)
        urea_kg = (remaining_n / 0.46) if remaining_n > 0 else 0.0

        # Formulation 2: Modern NPS + Urea (if S is requested)
        if S > 0:
            nps_kg = S / 0.07
            n_from_nps = nps_kg * 0.19
            p_from_nps = nps_kg * 0.167
            rem_n_nps = max(0.0, N - n_from_nps)
            urea_for_nps = (rem_n_nps / 0.46) if rem_n_nps > 0 else 0.0
        else:
            nps_kg = (P / 0.167) if P > 0 else 0.0
            n_from_nps = nps_kg * 0.19
            rem_n_nps = max(0.0, N - n_from_nps)
            urea_for_nps = (rem_n_nps / 0.46) if rem_n_nps > 0 else 0.0

        return {
            "traditional_dap_urea": {
                "dap_kg_ha": round(dap_kg, 1),
                "dap_50kg_bags_ha": round(dap_kg / 50.0, 1),
                "urea_kg_ha": round(urea_kg, 1),
                "urea_50kg_bags_ha": round(urea_kg / 50.0, 1),
                "total_bags_ha": round((dap_kg + urea_kg) / 50.0, 1),
            },
            "modern_nps_urea": {
                "nps_kg_ha": round(nps_kg, 1),
                "nps_50kg_bags_ha": round(nps_kg / 50.0, 1),
                "urea_kg_ha": round(urea_for_nps, 1),
                "urea_50kg_bags_ha": round(urea_for_nps / 50.0, 1),
                "total_bags_ha": round((nps_kg + urea_for_nps) / 50.0, 1),
            },
        }

    @staticmethod
    def check_coverage_warnings(
        latitude: float,
        longitude: float,
        adm1: str,
        N: float,
        P: float,
        K: float,
        S: float,
        Zn: float,
    ) -> List[str]:
        """Evaluate whether inputs fall within the empirical trial envelope and return explicit agronomic warnings."""
        warnings = []
        reg = str(adm1).lower().strip() if adm1 else ""

        # 1. Geographic checks
        if reg not in REPRESENTED_REGIONS:
            warnings.append(
                f"Region '{adm1}' has no wheat trials in this database. "
                "Predictions rely on environmental coordinate extrapolations."
            )
        elif reg in {"afar", "snnprs"}:
            warnings.append(
                f"Region '{adm1}' has limited trial representation (<1% of subset). Interpret with caution."
            )

        # Coordinate bounds check for Ethiopia (~3°N - 15°N, 33°E - 48°E)
        if not (3.0 <= latitude <= 15.0 and 33.0 <= longitude <= 48.0):
            warnings.append(
                f"Coordinates ({latitude:.2f}°N, {longitude:.2f}°E) fall outside Ethiopia's agricultural boundaries."
            )

        # 2. Nutrient rate envelope checks
        if N > 180.0:
            warnings.append(
                f"Nitrogen rate ({N:.0f} kg N/ha) exceeds optimal economic trial response (<180 kg/ha). Risk of lodging."
            )
        if P > 45.0:
            warnings.append(
                f"Phosphorus rate ({P:.0f} kg P/ha) exceeds standard trial treatments. Marginal return will plateau."
            )
        if K > 0:
            warnings.append(
                "Potassium trials are scarce in this database (93.3% of wheat records applied 0 K). "
                "Ethiopian highland Vertisols generally exhibit high native potassium availability."
            )
        if S > 40.0:
            warnings.append(
                f"Sulfur rate ({S:.0f} kg S/ha) is high relative to trial coverage."
            )
        if Zn > 0:
            warnings.append(
                "Zinc trials are absent (0.0 kg Zn/ha across all wheat records). Zinc prediction is uncalibrated."
            )

        return warnings


# Singleton instance for application reuse
wheat_service = WheatFertilizerService()


# ─────────────────────────────────────────────────────────────────────────────
# CROP-SPECIFIC FERTILIZER RECOMMENDER  (Urea + DAP, soil-adjusted)
# ─────────────────────────────────────────────────────────────────────────────

# Crop-specific nutrient requirements (kg elemental nutrient / ha)
# Sources: Ethiopian EIAR / MoA extension guidelines, CIMMYT, ICARDA
CROP_NUTRIENT_REQUIREMENTS: Dict[str, Dict[str, Any]] = {
    # ── Major Cereals ──────────────────────────────────────────────────────
    "Wheat": {
        "N_base": 64.0, "N_max": 92.0,
        "P_base": 20.0, "P_max": 25.0,
        "is_legume": False,
        "growth_days": 120,
        "n_timing": [
            {"stage": "Basal (sowing)", "pct": 50, "note": "Apply all DAP + 50% Urea at planting in furrows 5 cm below seed"},
            {"stage": "Tillering (25–30 DAS)", "pct": 30, "note": "Top-dress 30% Urea when 3–4 tillers visible"},
            {"stage": "Stem elongation (50–60 DAS)", "pct": 20, "note": "Final 20% Urea for grain filling boost"},
        ],
        "p_timing": "All P applied at sowing as basal via DAP",
        "science": (
            "Wheat has a high N demand at tillering and stem elongation. "
            "P is critical for root development and tillering. "
            "Split N application prevents leaching and lodging. "
            "DAP provides both P and a starter N dose; Urea supplies the bulk nitrogen."
        ),
    },
    "Maize": {
        "N_base": 110.0, "N_max": 138.0,
        "P_base": 25.0, "P_max": 32.0,
        "is_legume": False,
        "growth_days": 150,
        "n_timing": [
            {"stage": "Basal (sowing)", "pct": 33, "note": "Apply all DAP + 1/3 Urea at planting in furrows beside seed"},
            {"stage": "V6 – Knee-High (6-leaf stage)", "pct": 33, "note": "Side-dress 1/3 Urea when plants are knee-high (~30 cm)"},
            {"stage": "V12 – Tasseling (50–60 DAS)", "pct": 34, "note": "Final Urea top-dress; critical for cob fill"},
        ],
        "p_timing": "All P applied at sowing as DAP basal dressing",
        "science": (
            "Maize is the highest N-demand cereal. Three-split N application "
            "maximises uptake efficiency and minimises volatilisation. "
            "P ensures early root establishment and V6 cob initiation. "
            "Highland Ethiopian maize (BHQPY-545, JITU) responds strongly to 110–138 kg N/ha."
        ),
    },
    "Teff": {
        "N_base": 46.0, "N_max": 69.0,
        "P_base": 13.0, "P_max": 18.0,
        "is_legume": False,
        "growth_days": 90,
        "n_timing": [
            {"stage": "Basal (sowing)", "pct": 50, "note": "Apply all DAP + half of Urea broadcast before last tillage"},
            {"stage": "Active tillering (25 DAS)", "pct": 50, "note": "Top-dress remaining Urea; avoid late application to prevent lodging"},
        ],
        "p_timing": "All P applied at sowing as DAP",
        "science": (
            "Teff has moderate N requirements but is extremely susceptible to lodging "
            "at high N rates. Never exceed 69 kg N/ha. P enhances panicle development "
            "and grain set. Broadcast application suits the tiny teff seed drilled at "
            "shallow depth. DAP + Urea at 1:1 bag ratio is the MoA standard package."
        ),
    },
    "Barley": {
        "N_base": 46.0, "N_max": 69.0,
        "P_base": 15.0, "P_max": 20.0,
        "is_legume": False,
        "growth_days": 100,
        "n_timing": [
            {"stage": "Basal (sowing)", "pct": 50, "note": "Apply all DAP + half Urea at sowing in furrows"},
            {"stage": "Early stem elongation (35–40 DAS)", "pct": 50, "note": "Top-dress remaining Urea; apply before rain"},
        ],
        "p_timing": "All P applied as basal DAP at sowing",
        "science": (
            "Barley is efficient at lower N inputs and suited to Tigray and Bale highlands. "
            "Excess N causes lodging and reduces malt quality. P is critical for root anchoring "
            "in shallow montane soils. The 1 bag DAP + 1 bag Urea (50 kg each) per hectare "
            "is the standard EIAR recommendation for food barley."
        ),
    },
    # ── Cereals (Minor) ────────────────────────────────────────────────────
    "Sorghum": {
        "N_base": 60.0, "N_max": 90.0,
        "P_base": 15.0, "P_max": 22.0,
        "is_legume": False,
        "growth_days": 140,
        "n_timing": [
            {"stage": "Basal (sowing)", "pct": 40, "note": "Apply all DAP + 40% Urea at planting"},
            {"stage": "Panicle initiation (45–55 DAS)", "pct": 60, "note": "Apply remaining 60% Urea at panicle initiation for grain fill"},
        ],
        "p_timing": "All P applied at sowing as DAP",
        "science": (
            "Sorghum is drought-tolerant and deep-rooted; it delays N uptake relative to cereals. "
            "Large N top-dress at panicle initiation is key. Avoid early over-fertilisation in "
            "semi-arid zones to minimise aphid pressure. P aids early root development in "
            "shallow Cambisol and Vertisol soils of Tigray and Harari zones."
        ),
    },
    "Dagussa": {
        "N_base": 40.0, "N_max": 60.0,
        "P_base": 10.0, "P_max": 15.0,
        "is_legume": False,
        "growth_days": 100,
        "n_timing": [
            {"stage": "Basal (sowing)", "pct": 50, "note": "Apply all DAP + half Urea at sowing"},
            {"stage": "Tillering (30 DAS)", "pct": 50, "note": "Top-dress remaining Urea at tillering"},
        ],
        "p_timing": "All P applied at sowing as DAP",
        "science": (
            "Dagussa (finger millet) is adapted to lower-fertility acidic soils of SNNPRS and "
            "Sidama zones. Moderate N and P inputs suffice; high N causes rank growth. "
            "DAP provides essential P for the fine-root system. Urea top-dress at tillering "
            "improves tiller number and panicle weight."
        ),
    },
    # ── Pulses ─────────────────────────────────────────────────────────────
    "Bean": {
        "N_base": 0.0, "N_max": 20.0,
        "P_base": 22.0, "P_max": 30.0,
        "is_legume": True,
        "growth_days": 90,
        "n_timing": [
            {"stage": "Starter N at sowing (optional)", "pct": 100, "note": "Small starter: 20 kg Urea/ha only; rhizobium inoculation is preferred over Urea for N"},
        ],
        "p_timing": "All P as basal DAP at sowing",
        "science": (
            "Beans fix atmospheric N via Rhizobium symbiosis — Urea is NOT recommended unless "
            "rhizobia are absent or soil N is critically low. Apply only a starter N dose "
            "(≤20 kg N/ha) if needed. P is the main limiting nutrient: DAP at 100–150 kg/ha "
            "strongly increases nodulation, growth and pod set. Inoculate seed with Rhizobium "
            "phaseoli liquid inoculant before sowing for best results."
        ),
    },
    "Pea": {
        "N_base": 0.0, "N_max": 15.0,
        "P_base": 20.0, "P_max": 26.0,
        "is_legume": True,
        "growth_days": 95,
        "n_timing": [
            {"stage": "Starter N at sowing (optional)", "pct": 100, "note": "15 kg Urea/ha maximum starter dose; prefer Rhizobium inoculant"},
        ],
        "p_timing": "All P as basal DAP at sowing",
        "science": (
            "Field peas are strong N-fixers. Do not apply full Urea rates — excess N suppresses "
            "nodule formation and N fixation. Apply only small starter N if soil is cold (<12°C) "
            "at emergence. DAP is essential: P boosts nodulation and seed protein content. "
            "Ethiopian highland peas (Shekla, Tegegnework) respond well to 100–130 kg DAP/ha."
        ),
    },
    # ── Specialty ──────────────────────────────────────────────────────────
    "Niger seed": {
        "N_base": 30.0, "N_max": 46.0,
        "P_base": 12.0, "P_max": 18.0,
        "is_legume": False,
        "growth_days": 110,
        "n_timing": [
            {"stage": "Basal (sowing)", "pct": 50, "note": "Apply all DAP + half Urea broadcast before final harrowing"},
            {"stage": "Branching stage (35 DAS)", "pct": 50, "note": "Top-dress remaining Urea at branching for seed set"},
        ],
        "p_timing": "All P as basal DAP at sowing",
        "science": (
            "Niger seed (nug) has modest fertiliser requirements but responds well to balanced "
            "N+P application. Excessive N promotes vegetative growth at the expense of oil content. "
            "Apply moderate Urea rates split between sowing and branching. DAP supplies P critical "
            "for fat synthesis and seed oil accumulation. Key in Amhara, SNNPRS, and Welega zones."
        ),
    },
    "Potato": {
        "N_base": 110.0, "N_max": 138.0,
        "P_base": 28.0, "P_max": 38.0,
        "is_legume": False,
        "growth_days": 100,
        "n_timing": [
            {"stage": "Basal (planting)", "pct": 40, "note": "Apply all DAP + 40% Urea in furrow at 10 cm below tuber"},
            {"stage": "Stolon initiation (30–35 DAP)", "pct": 35, "note": "Side-dress Urea when haulms are 20–25 cm; critical stage"},
            {"stage": "Tuber bulking (55–60 DAP)", "pct": 25, "note": "Final Urea application for starch accumulation in tubers"},
        ],
        "p_timing": "All P applied at planting as basal DAP",
        "science": (
            "Potato has the highest N and P demand of Ethiopian crops. Three-split N application "
            "is essential to match crop uptake curve. P is critical for tuber bulking, starch synthesis "
            "and skin quality. DAP + Urea in-furrow placement maximises tuber number. Highland "
            "zones (Bale, Jeldu, Chencha) consistently show strong response to 110–138 kg N/ha."
        ),
    },
    "Red Pepper": {
        "N_base": 80.0, "N_max": 110.0,
        "P_base": 22.0, "P_max": 30.0,
        "is_legume": False,
        "growth_days": 180,
        "n_timing": [
            {"stage": "Transplanting (basal)", "pct": 30, "note": "Apply all DAP + 30% Urea at transplanting in holes"},
            {"stage": "Flowering (60–70 DAT)", "pct": 40, "note": "Apply 40% Urea at onset of flowering; critical for fruit set"},
            {"stage": "Fruit development (100–110 DAT)", "pct": 30, "note": "Final Urea application for size and capsaicin content"},
        ],
        "p_timing": "All P applied at transplanting as DAP",
        "science": (
            "Red pepper (Mareko Fana, Dedo Red) is a high-value crop requiring sustained N supply "
            "across its long growing season. P at transplanting ensures vigorous root establishment "
            "and early flower initiation. Split Urea applications match the three key growth stages. "
            "Irrigation under Awash, Ziway, and Koka command areas allows highest N response."
        ),
    },
    "Fallow": {
        "N_base": 0.0, "N_max": 0.0,
        "P_base": 0.0, "P_max": 0.0,
        "is_legume": False,
        "growth_days": 0,
        "n_timing": [],
        "p_timing": "No fertiliser on fallow land",
        "science": (
            "Fallow land should not receive commercial fertiliser. "
            "Consider green manure, farmyard manure (FYM), or legume cover crops "
            "to restore soil organic matter and biological N fixation for the subsequent crop."
        ),
    },
}

# Soil test critical levels for N and P (adjust recommendations if soil is already rich)
# N critical: soil N > 250 mg/kg → reduce by 30%; > 400 mg/kg → reduce by 50%
# P critical: soil P > 25 ppm → reduce P by 40%; < 8 ppm → increase P by 25%
# pH: < 5.5 → P fixation reduces efficiency; note lime recommendation
# SOC: > 3% → reduce N by 15% (high mineralisation)

class CropFertilizerRecommender:
    """Scientific Urea + DAP fertilizer recommender for all Ethiopian crop groups.
    
    Calculates crop-specific elemental N and P requirements adjusted for:
      - Soil available N (mg/kg) — reduces Urea if soil N is adequate
      - Soil available P (mg/kg) — reduces DAP if soil P is adequate
      - Soil pH — flags acid soils where P fixation limits DAP effectiveness
      - Soil Organic Carbon (SOC%) — reduces Urea where mineralisation is high
    Then converts to commercial Urea (46% N) and DAP (18% N, 46% P₂O₅ ≈ 20.1% elemental P) kg/ha.
    """

    UREA_N_CONTENT = 0.46          # 46% N
    DAP_P_CONTENT  = 0.201         # 20.1% elemental P (= 46% P₂O₅ / 2.29)
    DAP_N_CONTENT  = 0.18          # 18% N (co-nutrient in DAP)
    BAG_KG         = 50.0          # Standard Ethiopian fertiliser bag

    def __init__(self, use_ml: bool = True):
        self.rate_model = None
        self.rate_meta: Dict[str, Any] = {}
        self.rate_model_loaded = False
        if use_ml:
            self._load_rate_model()

    def _load_rate_model(self) -> None:
        path = MODELS_DIR / "fertilizer_rate_model.pkl"
        meta_path = MODELS_DIR / "fertilizer_rate_metadata.json"
        if joblib and path.exists():
            try:
                bundle = joblib.load(path)
                if isinstance(bundle, dict):
                    self.rate_model = bundle.get("model")
                    self.rate_meta = {k: v for k, v in bundle.items() if k != "model"}
                else:
                    self.rate_model = bundle
                self.rate_model_loaded = self.rate_model is not None
                if self.rate_model_loaded:
                    print(f"[EthioSoil] Loaded fertilizer rate model from {path}")
            except Exception as e:
                print(f"[EthioSoil] Warning: Could not load fertilizer rate model: {e}")
                self.rate_model = None
                self.rate_model_loaded = False
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    file_meta = json.load(f)
                self.rate_meta = {**file_meta, **self.rate_meta}
            except Exception:
                pass

    def _predict_np_ml(
        self,
        crop_key: str,
        soil_N_ppm: float,
        soil_P_ppm: float,
        soil_K_ppm: float,
        soil_pH: float,
        soil_SOC_pct: float,
        temperature: Optional[float],
        humidity: Optional[float],
        rainfall: Optional[float],
        altitude_m: Optional[float],
    ) -> Optional[Tuple[float, float]]:
        if not self.rate_model_loaded or self.rate_model is None:
            return None
        numeric = self.rate_meta.get("numeric_features") or [
            "soil_N_ppm", "soil_P_ppm", "soil_K_ppm", "soil_pH", "soil_SOC_pct",
            "temperature", "humidity", "rainfall", "altitude_m",
        ]
        row = {
            "soil_N_ppm": soil_N_ppm,
            "soil_P_ppm": soil_P_ppm,
            "soil_K_ppm": soil_K_ppm,
            "soil_pH": soil_pH,
            "soil_SOC_pct": soil_SOC_pct,
            "temperature": 18.0 if temperature is None else temperature,
            "humidity": 60.0 if humidity is None else humidity,
            "rainfall": 25.0 if rainfall is None else rainfall,
            "altitude_m": 2200.0 if altitude_m is None else altitude_m,
            "crop": crop_key,
        }
        X = pd.DataFrame([{k: row.get(k) for k in list(numeric) + ["crop"]}])
        pred = np.asarray(self.rate_model.predict(X), dtype=float).reshape(-1)
        if pred.size < 2:
            return None
        return float(max(0.0, pred[0])), float(max(0.0, pred[1]))

    def recommend(
        self,
        crop: str,
        soil_N_ppm: float = 150.0,
        soil_P_ppm: float = 15.0,
        soil_K_ppm: float = 280.0,
        soil_pH: float = 6.5,
        soil_SOC_pct: float = 2.0,
        strategy: str = "balanced",   # "low" | "balanced" | "high"
        adm1: str = "Oromia",
        temperature: Optional[float] = None,
        humidity: Optional[float] = None,
        rainfall: Optional[float] = None,
        altitude_m: Optional[float] = None,
        use_ml: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Return full Urea + DAP recommendation for the given crop and soil state."""

        # ── 1. Resolve crop name (handle group names) ──────────────────────
        crop_key = self._resolve_crop(crop)
        req = CROP_NUTRIENT_REQUIREMENTS.get(crop_key)
        if req is None:
            # Generic fallback
            req = CROP_NUTRIENT_REQUIREMENTS["Wheat"]
            crop_key = "Wheat (generic)"

        want_ml = self.rate_model_loaded if use_ml is None else bool(use_ml and self.rate_model_loaded)
        ml_np = None
        if want_ml:
            ml_np = self._predict_np_ml(
                crop_key=crop_key if crop_key != "Wheat (generic)" else "Wheat",
                soil_N_ppm=soil_N_ppm,
                soil_P_ppm=soil_P_ppm,
                soil_K_ppm=soil_K_ppm,
                soil_pH=soil_pH,
                soil_SOC_pct=soil_SOC_pct,
                temperature=temperature,
                humidity=humidity,
                rainfall=rainfall,
                altitude_m=altitude_m,
            )

        # ── 2. Base nutrient rates from strategy ───────────────────────────
        if strategy == "low":
            N_target = req["N_base"] * 0.75
            P_target = req["P_base"] * 0.80
        elif strategy == "high":
            N_target = req["N_max"]
            P_target = req["P_max"]
        else:  # balanced (default)
            N_target = (req["N_base"] + req["N_max"]) / 2.0
            P_target = (req["P_base"] + req["P_max"]) / 2.0

        # ── 3. Soil N adjustment ────────────────────────────────────────────
        N_adj_factor = 1.0
        N_adj_reason = "No soil N adjustment."
        if soil_N_ppm > 400:
            N_adj_factor = 0.50
            N_adj_reason = f"Soil N is very high ({soil_N_ppm:.0f} mg/kg). Reduce Urea by 50% — risk of over-fertilisation."
        elif soil_N_ppm > 250:
            N_adj_factor = 0.70
            N_adj_reason = f"Soil N is moderately high ({soil_N_ppm:.0f} mg/kg). Reduce Urea by 30%."
        elif soil_N_ppm < 80:
            N_adj_factor = 1.20
            N_adj_reason = f"Soil N is low ({soil_N_ppm:.0f} mg/kg). Increase Urea by 20% to compensate."
            N_adj_factor = min(N_adj_factor, req["N_max"] / max(N_target, 1.0))

        # SOC high → mineralisation supplies N
        if soil_SOC_pct > 3.0:
            N_adj_factor *= 0.85
            N_adj_reason += f" High SOC ({soil_SOC_pct:.1f}%) — mineralisation reduces synthetic N need by 15%."

        N_rec = max(0.0, N_target * N_adj_factor)
        if req["is_legume"]:
            N_rec = min(N_rec, req["N_max"])  # Hard cap for legumes

        # ── 4. Soil P adjustment ────────────────────────────────────────────
        P_adj_factor = 1.0
        P_adj_reason = "No soil P adjustment."
        if soil_P_ppm > 40:
            P_adj_factor = 0.50
            P_adj_reason = f"Soil P is very high ({soil_P_ppm:.0f} ppm). Reduce DAP by 50%."
        elif soil_P_ppm > 25:
            P_adj_factor = 0.60
            P_adj_reason = f"Soil P is high ({soil_P_ppm:.0f} ppm). Reduce DAP by 40%."
        elif soil_P_ppm < 8:
            P_adj_factor = 1.25
            P_adj_reason = f"Soil P is critically low ({soil_P_ppm:.0f} ppm). Increase DAP by 25%."

        P_rec = max(0.0, P_target * P_adj_factor)

        source = "agronomic_rules"
        if ml_np is not None:
            N_ml, P_ml = ml_np
            n_lo, n_hi = req["N_base"] * 0.45, req["N_max"] * 1.15
            p_lo, p_hi = req["P_base"] * 0.45, req["P_max"] * 1.15
            N_rec = float(np.clip(N_ml, n_lo, n_hi))
            P_rec = float(np.clip(P_ml, p_lo, p_hi))
            if strategy == "low":
                N_rec *= 0.85
                P_rec *= 0.90
            elif strategy == "high":
                N_rec = min(req["N_max"] * 1.05, N_rec * 1.12)
                P_rec = min(req["P_max"] * 1.05, P_rec * 1.10)
            if req["is_legume"]:
                N_rec = min(N_rec, req["N_max"])
            source = "ml_fertilizer_model"
            N_adj_reason = (
                f"N rate predicted by the fertilizer model from soil N ({soil_N_ppm:.0f} mg/kg), "
                f"SOC ({soil_SOC_pct:.1f}%) and crop={crop_key}."
            )
            P_adj_reason = (
                f"P rate predicted by the fertilizer model from soil P ({soil_P_ppm:.0f} ppm) "
                f"and crop={crop_key}."
            )

        # ── 5. pH warnings ──────────────────────────────────────────────────
        ph_warnings = []
        if soil_pH < 5.0:
            ph_warnings.append(
                f"⚠️ Soil pH is very acidic ({soil_pH:.1f}). Phosphorus fixation by Al³⁺/Fe³⁺ "
                "will immobilise up to 80% of applied DAP. Apply agricultural lime (CaCO₃) "
                "at 2–4 t/ha at least 4 weeks before sowing. pH must reach ≥5.5 for DAP to be effective."
            )
        elif soil_pH < 5.5:
            ph_warnings.append(
                f"⚠️ Soil pH is acidic ({soil_pH:.1f}). P fixation will reduce DAP efficiency. "
                "Apply 1–2 t/ha agricultural lime (CaCO₃) and incorporate before planting."
            )
        if soil_pH > 8.0:
            ph_warnings.append(
                f"⚠️ Soil pH is alkaline ({soil_pH:.1f}). Urea hydrolysis to NH₃ accelerates — "
                "place Urea 5–7 cm deep or use urease-inhibitor treated urea to reduce volatilisation."
            )

        k_note = (
            f"Soil K is {soil_K_ppm:.0f} mg/kg. Ethiopian highland soils are typically K-sufficient; "
            "Urea + DAP (N and P only) is the MoA standard package. MOP/KCl is not required."
        )
        if soil_K_ppm < 80:
            k_note = (
                f"⚠️ Soil K is low ({soil_K_ppm:.0f} mg/kg). Urea and DAP do not supply potassium. "
                "Add muriate of potash (MOP, 60% K₂O) at 50 kg/ha if locally available."
            )

        # ── 6. Convert elemental N, P → Urea + DAP kg/ha ──────────────────
        # Step A: DAP supplies P and some N
        dap_kg_ha  = P_rec / self.DAP_P_CONTENT if P_rec > 0 else 0.0
        n_from_dap = dap_kg_ha * self.DAP_N_CONTENT
        # Step B: Urea supplies remaining N
        remaining_N    = max(0.0, N_rec - n_from_dap)
        urea_kg_ha     = remaining_N / self.UREA_N_CONTENT if remaining_N > 0 else 0.0

        # Per-stage Urea splits
        stage_splits = self._split_urea_by_stages(urea_kg_ha, req.get("n_timing", []))

        # Bags
        dap_bags  = dap_kg_ha  / self.BAG_KG
        urea_bags = urea_kg_ha / self.BAG_KG

        # ── 7. DAP ratio proportion ─────────────────────────────────────────
        total_fert = dap_kg_ha + urea_kg_ha
        dap_pct  = round(dap_kg_ha  / total_fert * 100, 1) if total_fert > 0 else 0.0
        urea_pct = round(urea_kg_ha / total_fert * 100, 1) if total_fert > 0 else 0.0

        return {
            "crop": crop_key,
            "strategy": strategy,
            "region": adm1,
            "source": source,
            "nutrient_targets": {
                "N_recommended_kg_ha": round(N_rec, 1),
                "P_recommended_kg_ha": round(P_rec, 1),
                "N_from_DAP_kg_ha": round(n_from_dap, 1),
                "N_from_Urea_kg_ha": round(remaining_N, 1),
            },
            "commercial_products": {
                "DAP": {
                    "formula": "DAP — (NH₄)₂HPO₄ | 18% N · 46% P₂O₅ (20.1% P)",
                    "rate_kg_ha": round(dap_kg_ha, 1),
                    "bags_50kg_ha": round(dap_bags, 2),
                    "bags_integer": int(np.ceil(dap_bags)),
                    "proportion_pct": dap_pct,
                    "timing": req.get("p_timing", "Apply at sowing as basal"),
                },
                "Urea": {
                    "formula": "Urea — CO(NH₂)₂ | 46% N",
                    "total_rate_kg_ha": round(urea_kg_ha, 1),
                    "bags_50kg_ha": round(urea_bags, 2),
                    "bags_integer": int(np.ceil(urea_bags)),
                    "proportion_pct": urea_pct,
                    "split_application": stage_splits,
                },
            },
            "combination_ratio": {
                "description": f"For every {round(dap_bags, 1)} bag(s) of DAP use {round(urea_bags, 1)} bag(s) of Urea per hectare",
                "dap_to_urea_by_weight": f"{round(dap_kg_ha, 0):.0f} kg DAP : {round(urea_kg_ha, 0):.0f} kg Urea",
                "total_fertiliser_kg_ha": round(total_fert, 1),
                "total_bags_ha": round((dap_kg_ha + urea_kg_ha) / self.BAG_KG, 2),
            },
            "soil_adjustments": {
                "N_adjustment": N_adj_reason,
                "P_adjustment": P_adj_reason,
                "K_note": k_note,
                "pH_warnings": ph_warnings,
            },
            "is_legume": req["is_legume"],
            "growth_days": req["growth_days"],
            "science_note": req["science"],
        }

    @staticmethod
    def _resolve_crop(crop: str) -> str:
        """Map crop group names or variant spellings to CROP_NUTRIENT_REQUIREMENTS keys."""
        mapping = {
            "major_cereals": "Wheat",   # default representative
            "cereals": "Sorghum",
            "pulses": "Bean",
            "specialty": "Potato",
            "wheat": "Wheat",
            "maize": "Maize",
            "teff": "Teff",
            "barley": "Barley",
            "sorghum": "Sorghum",
            "dagussa": "Dagussa",
            "bean": "Bean",
            "pea": "Pea",
            "niger seed": "Niger seed",
            "niger": "Niger seed",
            "potato": "Potato",
            "red pepper": "Red Pepper",
            "pepper": "Red Pepper",
            "fallow": "Fallow",
        }
        return mapping.get(crop.lower().strip(), crop)

    @staticmethod
    def _split_urea_by_stages(total_urea_kg: float, timings: List[Dict]) -> List[Dict]:
        """Split Urea kg/ha across application stages by percentage."""
        splits = []
        for t in timings:
            pct = t.get("pct", 0)
            kg  = round(total_urea_kg * pct / 100.0, 1)
            bags = round(kg / 50.0, 2)
            splits.append({
                "stage": t["stage"],
                "percentage": pct,
                "urea_kg_ha": kg,
                "urea_bags_50kg": bags,
                "instruction": t.get("note", ""),
            })
        return splits


# Singleton for reuse
crop_fertilizer_recommender = CropFertilizerRecommender()
