# Ethiopian Crop Recommendation ML System – Implementation Summary

## 1. Project Overview

The system recommends suitable **crop groups** for Ethiopian farms from soil and climate data. It uses:

- Ethiopian soil/climate data (MERRA-2 and soil lab data)
- 11 engineered features
- A classification model to predict one of 4 crop groups

---

## 2. Data Pipeline

### 2.1 Raw Data (`Crop_recommendation_ethiopia_real.csv`)

- **Size**: 3,867 rows × 29 columns
- **Target**: `label` – crop type
- **Inputs**: Soil (N, P, K, Ph, Zn, S, soil color), MERRA-2 climate (seasonal T2M, QV2M, PRECTOTCORR, etc.), plus PS, GWETTOP, etc.
- **Original crops**: Teff, Maize, Wheat, Barley, Bean, Pea, Sorghum, Dagussa, Niger seed, Potato, Red Pepper, Fallow

### 2.2 Feature Engineering (Notebook 02)

Engineered 11 features:

| Feature        | Description                                           |
|----------------|-------------------------------------------------------|
| **N, P, K**    | Nitrogen, phosphorus, potassium (mg/kg)               |
| **ph**         | Soil pH                                               |
| **temperature**| Mean of seasonal T2M (max/min)                        |
| **humidity**   | Mean of QV2M (specific humidity proxy)                |
| **rainfall**   | Sum of seasonal PRECTOTCORR                           |
| **altitude_m** | Derived from PS via barometric formula                |
| **Zn, S**      | Zinc, sulfur                                          |
| **soil_moisture** | GWETTOP (topsoil wetness, 0–1)                    |

Soil color was one-hot encoded but later dropped; the final model uses only these 11 numeric features.

---

## 3. Preprocessing and Target Grouping (Notebook 04)

### 3.1 Label Merging

Labels are merged in stages to reduce class imbalance and produce usable groups:

1. **First merge**: Pea, Bean → Pulses; Niger seed → Oilseeds; Potato, Red Pepper, Fallow → Other_Specialty; Dagussa, Sorghum → Minor_Cereals
2. **Second merge**: Minor_Cereals → Cereals; Oilseeds, Other_Specialty → Specialty
3. **Major cereals merge**: Teff, Maize, Wheat, Barley → Major_Cereals

### 3.2 Final Target Classes

| Class           | Crops Included                          |
|-----------------|------------------------------------------|
| **Major_Cereals** | Teff, Maize, Wheat, Barley             |
| **Cereals**       | Dagussa, Sorghum                       |
| **Pulses**        | Bean, Pea                              |
| **Specialty**     | Niger seed, Potato, Red Pepper, Fallow |

### 3.3 Augmentation

- Rare classes (&lt; 200 samples): oversampled to 400 with 5% Gaussian noise on numeric features
- Classes augmented: Cereals (143→400), Specialty (167→400)
- Total size after augmentation: **4,667 rows**

### 3.4 Train/Val/Test Split

- Stratified split: 70% train / 15% val / 15% test
- Train: 3,266 | Val: 700 | Test: 701
- StandardScaler fit on train and applied to all splits
- LabelEncoder used for crop groups

---

## 4. Model Training (Notebook 05)

### 4.1 Model Comparison

| Model               | CV F1 Macro | Val Accuracy | Val F1 Macro |
|---------------------|-------------|--------------|--------------|
| **XGBoost**         | 0.592       | 81.9%        | 0.607        |
| Random Forest       | 0.586       | 81.3%        | 0.556        |
| Decision Tree       | 0.515       | 71.3%        | 0.524        |
| Logistic Regression | 0.289       | 28.1%        | 0.264        |

### 4.2 Selected Model

**XGBoost + SMOTE** was chosen and tuned with:

- SMOTE (`k_neighbors=3`) to handle class imbalance
- RandomizedSearchCV over hyperparameters
- Stratified 5-fold CV with F1-macro

### 4.3 Final Test Results

- **Test accuracy**: ~80.5%
- **Test F1 macro**: ~0.68
- **Test F1 weighted**: ~0.81

---

## 5. Deployment (FastAPI + Static Frontend)

### 5.1 API (`src/app.py`)

- **Framework**: FastAPI
- **Endpoint**: `POST /predict` – takes soil/climate inputs and returns crop recommendations
- **Input schema** (`CropInput`): N, P, K, ph, temperature, humidity, rainfall, altitude_m, Zn, S, soil_moisture (with validation ranges)
- **Output**: Best crop group, crops in that group, explanation, confidence, and top-3 alternatives with probabilities
- **Models**: `best_crop_model.pkl`, `scaler_merged.pkl`, `label_encoder_merged.pkl`

### 5.2 Prediction Flow

1. Validate input
2. Build feature vector (11 values)
3. Scale with `scaler.transform()`
4. `model.predict_proba()`
5. Return top-1 and top-3 crop groups with probabilities

### 5.3 Web UI (`static/index.html`)

- Form with the 11 inputs
- Sends `POST /predict` and shows:
  - Recommended crop group
  - Crops in that group
  - Short explanation
  - Confidence
  - Top-3 alternatives with probability bars

---

## 6. Project Structure

```
crop_recommendation/
├── data/                     # Raw and processed data
│   ├── Crop_recommendation_ethiopia_real.csv
│   └── processed/            # engineered, merged, aug, splits
├── models/                   # Saved models
│   ├── best_crop_model.pkl   # XGBoost + SMOTE pipeline
│   ├── scaler_merged.pkl
│   ├── label_encoder_merged.pkl
│   └── feature_names.pkl
├── notebooks/                # 01–05: exploration, feature eng, EDA, preprocessing, training
├── reports/figures/          # Confusion matrix, feature importance, etc.
├── src/app.py               # FastAPI application
├── static/index.html        # Web interface
└── requirements.txt         # Dependencies (e.g., FastAPI, XGBoost, scikit-learn)
```

---

## 7. Summary

- **Data**: Ethiopian soil and MERRA-2 climate data, 12 original crops merged into 4 groups.
- **Features**: 11 engineered inputs (NPK, pH, temperature, humidity, rainfall, altitude, Zn, S, soil moisture).
- **Handling imbalance**: SMOTE, `class_weight='balanced'`, and oversampling of rare classes.
- **Model**: XGBoost with SMOTE, ~80.5% accuracy and ~0.68 F1-macro on held-out test data.
- **Deployment**: FastAPI backend with a static web UI for interactive recommendations.
