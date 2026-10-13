#!/usr/bin/env python3
"""
EthioSoil-AI: Trial-Aware Wheat Fertilizer-Response Model Training Pipeline.

Trains and validates regression models for wheat grain yield (kg/ha) using
strict trial-aware GroupKFold validation to prevent data leakage across agricultural trials.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EthioSoil-WheatTraining")


def find_data_path(custom_path: str = None) -> Path:
    if custom_path and Path(custom_path).exists():
        return Path(custom_path)
    candidates = [
        Path("data/processed/ethiosoil_wheat_grain_trials.csv"),
        Path("../data/processed/ethiosoil_wheat_grain_trials.csv"),
        Path(__file__).resolve().parent.parent / "data" / "processed" / "ethiosoil_wheat_grain_trials.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("Could not locate ethiosoil_wheat_grain_trials.csv in data/processed/")


def preprocess_wheat_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean records, impute known coordinate-based metadata, and engineer agronomic interactions."""
    data = df.copy()

    # Fill unassigned adm1 if coordinates fall inside known Amhara research stations
    # (e.g. Minale Liben trials at ~11.5°N, 37.3°E around South Gondar / Gojjam)
    missing_adm1 = data["adm1"].isna()
    if missing_adm1.sum() > 0:
        data.loc[missing_adm1 & (data["latitude"] >= 10.5), "adm1"] = "Amhara"
        data.loc[missing_adm1 & (data["latitude"] < 10.5), "adm1"] = "Oromia"

    # Fill soil_type missingness with explicit 'Unknown' token
    data["soil_type"] = data["soil_type"].fillna("Unknown").astype(str).str.strip()
    data["adm1"] = data["adm1"].fillna("Unknown").astype(str).str.strip()

    # Feature Engineering: Fertilizer interaction and polynomial terms
    # Reflects diminishing marginal returns and NP synergy
    data["NP_interaction"] = data["N_fertilizer"] * data["P_fertilizer"]
    data["N_sq"] = data["N_fertilizer"] ** 2
    data["P_sq"] = data["P_fertilizer"] ** 2

    return data


def build_pipelines(
    numeric_features: List[str], categorical_features: List[str]
) -> Dict[str, Pipeline]:
    """Build preprocessor and model pipelines."""
    # Preprocessor for Linear/Tree models needing explicit imputation
    num_imputer_scaler = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ])

    cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor_standard = ColumnTransformer(
        transformers=[
            ("num", num_imputer_scaler, numeric_features),
            ("cat", cat_encoder, categorical_features),
        ],
        remainder="drop",
    )

    # Preprocessor for HistGradientBoosting (handles NaNs natively in numeric columns)
    preprocessor_hist = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ],
        remainder="drop",
    )

    pipelines = {
        "Dummy (Mean)": Pipeline([("model", DummyRegressor(strategy="mean"))]),
        "Ridge Regression": Pipeline([
            ("prep", preprocessor_standard),
            ("model", Ridge(alpha=10.0)),
        ]),
        "Random Forest Regressor": Pipeline([
            ("prep", preprocessor_standard),
            ("model", RandomForestRegressor(
                n_estimators=150, max_depth=10, min_samples_leaf=4, random_state=42, n_jobs=-1
            )),
        ]),
        "HistGradientBoosting": Pipeline([
            ("prep", preprocessor_hist),
            ("model", HistGradientBoostingRegressor(
                max_iter=150, max_depth=6, min_samples_leaf=15, l2_regularization=1.5, random_state=42
            )),
        ]),
    }
    return pipelines


def evaluate_trial_aware_cv(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    pipelines: Dict[str, Pipeline],
    n_splits: int = 5,
) -> Dict[str, Dict[str, float]]:
    """Strict GroupKFold Cross Validation grouped by trial_id."""
    gkf = GroupKFold(n_splits=n_splits)
    results = {}

    for name, pipe in pipelines.items():
        rmses, maes, r2s, peTop = [], [], [], []

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups), 1):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_val)

            rmse = np.sqrt(mean_squared_error(y_val, preds))
            mae = mean_absolute_error(y_val, preds)
            r2 = r2_score(y_val, preds)

            # Pearson correlation if variance exists
            if np.std(preds) > 1e-4 and np.std(y_val) > 1e-4:
                corr, _ = pearsonr(y_val, preds)
            else:
                corr = 0.0

            rmses.append(rmse)
            maes.append(mae)
            r2s.append(r2)
            peTop.append(corr)

        results[name] = {
            "cv_rmse_mean": float(np.mean(rmses)),
            "cv_rmse_std": float(np.std(rmses)),
            "cv_mae_mean": float(np.mean(maes)),
            "cv_r2_mean": float(np.mean(r2s)),
            "cv_pearson_r_mean": float(np.mean(peTop)),
        }
        logger.info(
            f"[{name}] Trial-CV RMSE: {np.mean(rmses):.1f} ± {np.std(rmses):.1f} kg/ha | "
            f"MAE: {np.mean(maes):.1f} | R²: {np.mean(r2s):.3f} | Pearson r: {np.mean(peTop):.3f}"
        )

    return results


def verify_response_sanity(model: Pipeline) -> Dict[str, Any]:
    """Test agronomic sanity of the trained model."""
    test_rates = [0.0, 23.0, 46.0, 69.0, 92.0, 115.0, 138.0]
    rows = []
    for n in test_rates:
        rows.append({
            "N_fertilizer": n,
            "P_fertilizer": 20.0,
            "K_fertilizer": 0.0,
            "S_fertilizer": 0.0,
            "latitude": 8.5,
            "longitude": 39.2,
            "adm1": "Oromia",
            "soil_pH": 6.5,
            "soil_SOC": 2.0,
            "soil_P": 15.0,
            "soil_clay": 45.0,
            "soil_sand": 20.0,
            "soil_type": "Vertisols",
            "NP_interaction": n * 20.0,
            "N_sq": n**2,
            "P_sq": 20.0**2,
        })
    preds = model.predict(pd.DataFrame(rows))
    control_yield = preds[0]
    yield_gains = [float(p - control_yield) for p in preds]

    logger.info("Agronomic N-Response Sanity Check (P=20 kg/ha, Oromia Vertisol):")
    for n, y, g in zip(test_rates, preds, yield_gains):
        logger.info(f"  N = {n:3.0f} kg/ha -> Predicted Yield: {y:5.0f} kg/ha (Δ: +{g:4.0f} kg/ha)")

    # Return summary dict
    return {
        "test_n_rates": test_rates,
        "predicted_yields": [float(round(p, 1)) for p in preds],
        "yield_gains": [float(round(g, 1)) for g in yield_gains],
        "sanity_check_passed": bool(yield_gains[3] > 0 and yield_gains[-1] >= yield_gains[1]),
    }


def main():
    parser = argparse.ArgumentParser(description="Train EthioSoil-AI Wheat Fertilizer Model")
    parser.add_argument("--data-path", type=str, default=None, help="Path to wheat trial CSV")
    parser.add_argument("--output-model", type=str, default=None, help="Output .pkl file")
    parser.add_argument("--output-metadata", type=str, default=None, help="Output .json metadata file")
    args = parser.parse_args()

    data_file = find_data_path(args.data_path)
    base_dir = data_file.parent.parent.parent
    model_out = Path(args.output_model) if args.output_model else base_dir / "models" / "wheat_fertilizer_model.pkl"
    meta_out = Path(args.output_metadata) if args.output_metadata else base_dir / "models" / "wheat_fertilizer_metadata.json"
    model_out.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading wheat trial dataset from: {data_file}")
    df_raw = pd.read_csv(data_file)
    logger.info(f"Raw wheat grain records: {len(df_raw)} rows, {df_raw['trial_id'].nunique()} distinct trials.")

    df = preprocess_wheat_data(df_raw)

    numeric_features = [
        "N_fertilizer", "P_fertilizer", "K_fertilizer", "S_fertilizer",
        "latitude", "longitude", "soil_pH", "soil_SOC", "soil_P",
        "soil_clay", "soil_sand", "NP_interaction", "N_sq", "P_sq"
    ]
    categorical_features = ["adm1", "soil_type"]
    feature_cols = numeric_features + categorical_features

    X = df[feature_cols]
    y = df["yield"]
    groups = df["trial_id"]

    logger.info(f"Total features: {len(feature_cols)} ({len(numeric_features)} numeric, {len(categorical_features)} categorical)")
    logger.info("Executing 5-fold trial-aware GroupKFold validation...")

    pipelines = build_pipelines(numeric_features, categorical_features)
    cv_metrics = evaluate_trial_aware_cv(X, y, groups, pipelines, n_splits=5)

    # Select best model based on trial-aware RMSE
    best_name = min(
        [k for k in cv_metrics if k != "Dummy (Mean)"],
        key=lambda k: cv_metrics[k]["cv_rmse_mean"]
    )
    logger.info(f"Selected best model architecture: '{best_name}'")
    best_pipeline = pipelines[best_name]

    # Fit best model on all available trial data
    logger.info("Fitting best model on full wheat trial dataset...")
    best_pipeline.fit(X, y)

    # Sanity check
    sanity = verify_response_sanity(best_pipeline)

    # Save model and metadata
    logger.info(f"Saving final trained model to: {model_out}")
    joblib.dump(best_pipeline, model_out)

    metadata = {
        "model_name": best_name,
        "target": "yield",
        "target_unit": "kg/ha (grain yield at standard storage moisture)",
        "nutrient_units": "kg/ha elemental N, P, K, S, Zn",
        "total_records": len(df),
        "total_trials": int(df["trial_id"].nunique()),
        "validation_strategy": "5-fold GroupKFold by trial_id (zero data leakage)",
        "features": {
            "numeric": numeric_features,
            "categorical": categorical_features,
        },
        "group_cv_metrics": cv_metrics,
        "sanity_check": sanity,
        "dataset_source": "Harvard Dataverse doi:10.7910/DVN/RKUMXB (Tamene et al. / EIAR / CIAT)",
    }

    with open(meta_out, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved model metadata to: {meta_out}")
    logger.info("Wheat fertilizer model training pipeline completed successfully!")


if __name__ == "__main__":
    main()
